"""Unit and integration tests for IngestionService, lifecycle states, idempotency, and sequence-length alignment."""
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pymupdf
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from transformers import AutoTokenizer

from app.core.config import settings
from app.core.database import Base, create_session_factory, get_db
from app.core.security import create_access_token
from app.main import app
from app.models.chunk import Chunk
from app.models.document import Document, ProcessingStatus
from app.models.user import User
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService, EncodingError
from app.services.ingestion_service import IngestionService
from app.services.pdf_service import ExtractedPage, PDFExtractionResult, PDFService
from app.services.vector_store import VectorStore, VectorStoreError


def create_sample_pdf(file_path: Path, pages_text: list[str]) -> Path:
    """Helper utility to create a PDF file on disk."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
    doc.save(str(file_path))
    doc.close()
    return file_path


@pytest.fixture
def db_engine():
    """Provides an isolated in-memory SQLite engine with foreign keys enabled."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @sa.event.listens_for(engine, "connect")
    def _enable_fk(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    try:
        yield engine
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def session_factory(db_engine):
    """Provides a sessionmaker for the test engine."""
    return create_session_factory(db_engine)


@pytest.fixture
def db_session(session_factory):
    """Provides an active test database session."""
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_dirs(tmp_path: Path, monkeypatch):
    """Configures isolated test directories for uploads and vector indices."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    index_dir = tmp_path / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "UPLOAD_DIR", str(upload_dir))
    monkeypatch.setattr(settings, "VECTOR_INDEX_DIR", str(index_dir))
    return {"upload_dir": upload_dir, "index_dir": index_dir}


@pytest.fixture
def test_user(db_session: Session) -> User:
    """Creates a test user in the database."""
    user = User(
        id=uuid4(),
        email="researcher@example.com",
        hashed_password="hashed_pw_xyz",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def other_user(db_session: Session) -> User:
    """Creates a second test user for cross-tenant isolation testing."""
    user = User(
        id=uuid4(),
        email="other_researcher@example.com",
        hashed_password="hashed_pw_abc",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    """Provides a fast mock EmbeddingService generating normalized 384-d vectors without model download."""
    service = MagicMock(spec=EmbeddingService)
    service.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    service.dimension = 384

    def _embed_batch(texts: list[str]) -> list[list[float]]:
        embeddings = []
        for i, text in enumerate(texts):
            # Deterministic reproducible float vector
            vec = np.zeros(384, dtype=np.float32)
            seed_val = (hash(text) % 1000) / 1000.0
            vec[0] = 1.0 + seed_val
            vec[1] = 0.5
            norm = np.linalg.norm(vec)
            vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings

    service.embed_batch.side_effect = _embed_batch
    return service


@pytest.fixture
def test_vector_store(test_dirs) -> VectorStore:
    """Provides a VectorStore persisting into test_dirs['index_dir']."""
    return VectorStore(
        dimension=384,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_dir=test_dirs["index_dir"],
    )


@pytest.fixture
def ingestion_service(
    session_factory, mock_embedding_service, test_vector_store
) -> IngestionService:
    """Provides an IngestionService configured with test dependencies."""
    return IngestionService(
        pdf_service=PDFService(),
        chunking_service=ChunkingService(target_tokens=200, overlap_tokens=30),
        embedding_service=mock_embedding_service,
        vector_store=test_vector_store,
        session_factory=session_factory,
    )


class TestIngestionSuccess:
    """Tests successful document ingestion end-to-end."""

    def test_successful_ingestion_end_to_end(
        self,
        db_session: Session,
        test_user: User,
        test_dirs: dict,
        ingestion_service: IngestionService,
        test_vector_store: VectorStore,
    ):
        pdf_path = test_dirs["upload_dir"] / "paper.pdf"
        create_sample_pdf(
            pdf_path,
            [
                "Introduction to Transformers. Self-attention mechanisms map queries and key-value pairs to outputs.",
                "Empirical Evaluation. We benchmark against state of the art convolutional models on ImageNet.",
            ],
        )

        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="paper.pdf",
            title="Transformer Overview",
            file_path=str(pdf_path),
            file_hash_sha256="test_sha256_hash_1",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db_session.add(doc)
        db_session.commit()

        # Execute ingestion pipeline
        result = ingestion_service.process_document(doc.id)

        assert result is not None
        assert result.processing_status == ProcessingStatus.READY.value
        assert result.error_message is None
        assert result.page_count == 2

        # Verify PostgreSQL chunks
        db_session.expire_all()
        persisted_chunks = (
            db_session.query(Chunk)
            .filter(Chunk.document_id == doc.id)
            .order_by(Chunk.chunk_index)
            .all()
        )
        assert len(persisted_chunks) == 2
        assert persisted_chunks[0].page_number == 1
        assert "Self-attention" in persisted_chunks[0].content
        assert persisted_chunks[1].page_number == 2
        assert "Empirical Evaluation" in persisted_chunks[1].content
        assert persisted_chunks[0].embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert persisted_chunks[0].token_count > 0

        # Verify FAISS vector store
        chunk_ids = [c.id for c in persisted_chunks]
        assert test_vector_store.total_vectors == len(persisted_chunks)
        for cid in chunk_ids:
            assert test_vector_store.contains_id(cid)

        # Verify atomic disk persistence
        assert test_vector_store.index_path.exists()
        assert test_vector_store.metadata_path.exists()


class TestIngestionFailuresAndRollbacks:
    """Tests failure paths, error sanitization, and atomic cleanup."""

    def test_empty_or_scanned_pdf_fails_without_creating_chunks_or_vectors(
        self,
        db_session: Session,
        test_user: User,
        test_dirs: dict,
        ingestion_service: IngestionService,
        test_vector_store: VectorStore,
    ):
        pdf_path = test_dirs["upload_dir"] / "blank.pdf"
        create_sample_pdf(pdf_path, ["", "   \n\t  "])  # 2 blank pages

        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="blank.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="blank_hash",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db_session.add(doc)
        db_session.commit()

        result = ingestion_service.process_document(doc.id)

        assert result is not None
        assert result.processing_status == ProcessingStatus.FAILED.value
        assert "No extractable text found" in result.error_message

        # Verify no orphan chunks or vectors
        db_session.expire_all()
        chunks = db_session.query(Chunk).filter(Chunk.document_id == doc.id).all()
        assert len(chunks) == 0
        assert test_vector_store.total_vectors == 0

    def test_corrupted_pdf_fails_with_sanitized_message(
        self,
        db_session: Session,
        test_user: User,
        test_dirs: dict,
        ingestion_service: IngestionService,
        test_vector_store: VectorStore,
    ):
        corrupted_path = test_dirs["upload_dir"] / "corrupt.pdf"
        corrupted_path.write_bytes(b"%PDF-1.4\nINVALID_CORRUPT_BYTES")

        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="corrupt.pdf",
            file_path=str(corrupted_path),
            file_hash_sha256="corrupt_hash",
            file_size_bytes=corrupted_path.stat().st_size,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db_session.add(doc)
        db_session.commit()

        result = ingestion_service.process_document(doc.id)

        assert result is not None
        assert result.processing_status == ProcessingStatus.FAILED.value
        assert result.error_message == "Corrupted or invalid PDF file."
        # Verify internal paths did not leak into client error
        assert str(test_dirs["upload_dir"]) not in result.error_message
        assert test_vector_store.total_vectors == 0

    def test_missing_file_fails_safely(
        self,
        db_session: Session,
        test_user: User,
        ingestion_service: IngestionService,
    ):
        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="missing.pdf",
            file_path="/tmp/nonexistent_file_path_12345.pdf",
            file_hash_sha256="missing_hash",
            file_size_bytes=100,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db_session.add(doc)
        db_session.commit()

        result = ingestion_service.process_document(doc.id)
        assert result is not None
        assert result.processing_status == ProcessingStatus.FAILED.value
        assert result.error_message == "Document file not found on disk."

    def test_embedding_failure_cleans_up_database_chunks(
        self,
        db_session: Session,
        test_user: User,
        test_dirs: dict,
        ingestion_service: IngestionService,
        mock_embedding_service: EmbeddingService,
        test_vector_store: VectorStore,
    ):
        pdf_path = test_dirs["upload_dir"] / "embed_fail.pdf"
        create_sample_pdf(pdf_path, ["Valid passage that fails during embedding."])

        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="embed_fail.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="embed_fail_hash",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db_session.add(doc)
        db_session.commit()

        # Simulate embedding failure
        mock_embedding_service.embed_batch.side_effect = EncodingError("Inference out of memory")

        result = ingestion_service.process_document(doc.id)

        assert result is not None
        assert result.processing_status == ProcessingStatus.FAILED.value
        assert "Failed to generate vector embeddings" in result.error_message

        # Crucial consistency assertion: chunks must NOT be left in DB without vectors
        db_session.expire_all()
        chunks = db_session.query(Chunk).filter(Chunk.document_id == doc.id).all()
        assert len(chunks) == 0
        assert test_vector_store.total_vectors == 0

    def test_vector_indexing_failure_cleans_up_chunks_and_partial_vectors(
        self,
        db_session: Session,
        test_user: User,
        test_dirs: dict,
        ingestion_service: IngestionService,
        test_vector_store: VectorStore,
    ):
        pdf_path = test_dirs["upload_dir"] / "vector_fail.pdf"
        create_sample_pdf(pdf_path, ["Valid passage that fails in vector store."])

        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="vector_fail.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="vector_fail_hash",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db_session.add(doc)
        db_session.commit()

        # Simulate vector store failure
        with patch.object(test_vector_store, "add_vectors", side_effect=VectorStoreError("FAISS disk error")):
            result = ingestion_service.process_document(doc.id)

        assert result is not None
        assert result.processing_status == ProcessingStatus.FAILED.value
        assert "Failed to index vector embeddings" in result.error_message

        # Crucial consistency assertion: DB chunks rolled back, FAISS empty
        db_session.expire_all()
        chunks = db_session.query(Chunk).filter(Chunk.document_id == doc.id).all()
        assert len(chunks) == 0
        assert test_vector_store.total_vectors == 0


class TestIdempotencyAndRetry:
    """Tests idempotency, re-ingestion, and retry behavior."""

    def test_reingestion_is_strictly_idempotent(
        self,
        db_session: Session,
        test_user: User,
        test_dirs: dict,
        ingestion_service: IngestionService,
        test_vector_store: VectorStore,
    ):
        pdf_path = test_dirs["upload_dir"] / "retry_paper.pdf"
        create_sample_pdf(
            pdf_path,
            ["Page 1 text on gradient descent.", "Page 2 text on loss landscape."],
        )

        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="retry_paper.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="retry_hash",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db_session.add(doc)
        db_session.commit()

        # Run 1: initial ingestion
        r1 = ingestion_service.process_document(doc.id)
        assert r1.processing_status == ProcessingStatus.READY.value

        db_session.expire_all()
        chunks_1 = db_session.query(Chunk).filter(Chunk.document_id == doc.id).all()
        chunk_count_1 = len(chunks_1)
        assert chunk_count_1 == 2
        assert test_vector_store.total_vectors == 2

        # Run 2: re-ingest same document
        r2 = ingestion_service.process_document(doc.id)
        assert r2.processing_status == ProcessingStatus.READY.value

        db_session.expire_all()
        chunks_2 = db_session.query(Chunk).filter(Chunk.document_id == doc.id).all()
        assert len(chunks_2) == 2  # NOT 4!
        assert test_vector_store.total_vectors == 2  # NOT 4!

        # Vectors in FAISS map strictly to new chunk IDs
        for c in chunks_2:
            assert test_vector_store.contains_id(c.id)

    def test_retry_from_failed_state_recovers_to_ready(
        self,
        db_session: Session,
        test_user: User,
        test_dirs: dict,
        ingestion_service: IngestionService,
        mock_embedding_service: EmbeddingService,
        test_vector_store: VectorStore,
    ):
        pdf_path = test_dirs["upload_dir"] / "recoverable.pdf"
        create_sample_pdf(pdf_path, ["Passage to recover."])

        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="recoverable.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="recoverable_hash",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db_session.add(doc)
        db_session.commit()

        # Fail on run 1
        mock_embedding_service.embed_batch.side_effect = EncodingError("Temporary GPU OOM")
        r1 = ingestion_service.process_document(doc.id)
        assert r1.processing_status == ProcessingStatus.FAILED.value
        assert test_vector_store.total_vectors == 0

        # Succeed on retry
        mock_embedding_service.embed_batch.side_effect = None
        # restore deterministic embed function
        mock_embedding_service.embed_batch.side_effect = lambda texts: [
            (np.ones(384, dtype=np.float32) / np.sqrt(384)).tolist() for _ in texts
        ]

        r2 = ingestion_service.retry_document(doc.id)
        assert r2.processing_status == ProcessingStatus.READY.value
        assert r2.error_message is None

        db_session.expire_all()
        chunks = db_session.query(Chunk).filter(Chunk.document_id == doc.id).all()
        assert len(chunks) == 1
        assert test_vector_store.total_vectors == 1
        assert test_vector_store.contains_id(chunks[0].id)


class TestSequenceLengthAlignment:
    """Tests chunk sizing alignment with all-MiniLM-L6-v2's 256 WordPiece token limit."""

    def test_chunk_token_limit_eliminates_silent_truncation(self):
        """Proves that default chunking targets passages that never exceed the 256 WordPiece sequence length."""
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2", local_files_only=True
        )

        # Realistic academic text with technical terms, subwords, and punctuation
        academic_text = (
            "In this study, we investigate the empirical behavior of transformer-based architectures "
            "under extreme sequence length constraints. Specifically, cross-entropy loss functions and "
            "regularized weight decay algorithms interact non-linearly when processing high-entropy "
            "scientific corpora. Furthermore, attention heads exhibit sparse activation patterns across "
            "intermediate feed-forward representations, indicating redundant parameter allocations in "
            "overparameterized neural language models. We evaluate our hypotheses across three standardized "
            "benchmarks: GLUE, SuperGLUE, and scientific passage retrieval suites. "
        ) * 5  # ~3,500 characters

        page = ExtractedPage(page_number=1, text=academic_text)

        # 1. New default chunker (~200 target tokens, ~30 overlap)
        chunker_aligned = ChunkingService(target_tokens=200, overlap_tokens=30)
        chunks_aligned = chunker_aligned.chunk_page(page)

        assert len(chunks_aligned) >= 4
        for idx, chunk in enumerate(chunks_aligned):
            # Inspect actual WordPiece tokens produced by the model tokenizer
            tokens = tokenizer.encode(chunk.text, add_special_tokens=True)
            token_count = len(tokens)

            # CRITICAL GUARANTEE: Must strictly fit under the model's 256 token limit!
            assert token_count <= 256, (
                f"Chunk {idx} has {token_count} WordPiece tokens, which exceeds the 256 limit! "
                f"Silent truncation would occur!"
            )
            # Character count is comfortably bounded by target_chars (200 * 4 = 800)
            assert len(chunk.text) <= 800

        # 2. Compare against legacy unaligned chunker (~400 tokens)
        chunker_legacy = ChunkingService(target_tokens=400, overlap_tokens=50)
        chunks_legacy = chunker_legacy.chunk_page(page)

        # Legacy chunker produces chunks that exceed 256 tokens (proving the truncation risk existed)
        legacy_lengths = [
            len(tokenizer.encode(c.text, add_special_tokens=True))
            for c in chunks_legacy
        ]
        assert any(l > 256 for l in legacy_lengths), "Legacy chunker was expected to exceed 256 tokens."


class TestAPIRetryEndpointAndIsolation:
    """Tests the POST /api/v1/documents/{id}/retry endpoint and multi-tenant isolation."""

    @pytest.fixture
    def client(self, db_session: Session, test_dirs: dict):
        def _override_get_db():
            yield db_session

        app.dependency_overrides[get_db] = _override_get_db
        with TestClient(app) as test_client:
            yield test_client
        app.dependency_overrides.clear()

    @pytest.fixture
    def auth_headers(self, test_user: User) -> dict:
        token = create_access_token(subject=test_user.id)
        return {"Authorization": f"Bearer {token}"}

    @pytest.fixture
    def other_auth_headers(self, other_user: User) -> dict:
        token = create_access_token(subject=other_user.id)
        return {"Authorization": f"Bearer {token}"}

    def test_retry_endpoint_owner_initiates_processing(
        self,
        client: TestClient,
        auth_headers: dict,
        db_session: Session,
        test_user: User,
        test_dirs: dict,
        mock_embedding_service: EmbeddingService,
    ):
        pdf_path = test_dirs["upload_dir"] / "retry_api.pdf"
        create_sample_pdf(pdf_path, ["Passage to retry via API."])

        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="retry_api.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="api_retry_hash",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.FAILED.value,
            error_message="Previous transient error",
        )
        db_session.add(doc)
        db_session.commit()

        with patch("app.services.ingestion_service.get_embedding_service", return_value=mock_embedding_service):
            response = client.post(
                f"/api/v1/documents/{doc.id}/retry",
                headers=auth_headers,
            )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(doc.id)

        # In TestClient, background tasks execute synchronously before return
        db_session.expire_all()
        reloaded = db_session.query(Document).filter(Document.id == doc.id).first()
        assert reloaded.processing_status == ProcessingStatus.READY.value
        assert reloaded.error_message is None

    def test_retry_endpoint_other_user_forbidden_returns_404(
        self,
        client: TestClient,
        other_auth_headers: dict,
        db_session: Session,
        test_user: User,
        test_dirs: dict,
    ):
        pdf_path = test_dirs["upload_dir"] / "private.pdf"
        create_sample_pdf(pdf_path, ["Private research."])

        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,  # Owned by test_user
            filename="private.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="private_hash",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.FAILED.value,
        )
        db_session.add(doc)
        db_session.commit()

        # other_user attempts to retry test_user's document
        response = client.post(
            f"/api/v1/documents/{doc.id}/retry",
            headers=other_auth_headers,
        )
        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]

    def test_retry_endpoint_nonexistent_document_returns_404(
        self,
        client: TestClient,
        auth_headers: dict,
    ):
        response = client.post(
            f"/api/v1/documents/{uuid4()}/retry",
            headers=auth_headers,
        )
        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]
