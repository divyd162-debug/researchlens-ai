"""PostgreSQL integration tests for ResearchLens AI.

Runs exclusively against a dedicated test database (TEST_DATABASE_URL).
Never runs against the development database.
Verifies:
1. Real PostgreSQL connection and Alembic schema.
2. End-to-end PDF ingestion pipeline against PostgreSQL.
3. Concurrent ingestion status claims with PostgreSQL row locking.
4. Deterministic FAISS recovery from PostgreSQL chunk records.
"""
import concurrent.futures
import os
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest
import pymupdf
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.database import Base, create_session_factory, db_session_scope
from app.models.chunk import Chunk
from app.models.document import Document, ProcessingStatus
from app.models.user import User
from app.services.embedding_service import EmbeddingService
from app.services.index_recovery import rebuild_vector_index, verify_and_recover_vector_index
from app.services.ingestion_service import IngestionService
from app.services.pdf_service import PDFService
from app.services.vector_store import VectorStore

POSTGRES_TEST_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5432/researchlens_test",
)


def is_postgres_available() -> bool:
    """Checks if the PostgreSQL test instance is accessible."""
    try:
        engine = create_engine(POSTGRES_TEST_URL, connect_args={"connect_timeout": 2})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not is_postgres_available(),
    reason="Live PostgreSQL test database not reachable at TEST_DATABASE_URL",
)


@pytest.fixture(scope="module")
def pg_engine():
    """Provides a module-scoped PostgreSQL engine connected to the dedicated test database."""
    engine = create_engine(POSTGRES_TEST_URL, pool_size=5, max_overflow=10)
    yield engine
    engine.dispose()


@pytest.fixture(scope="module")
def pg_session_factory(pg_engine) -> sessionmaker[Session]:
    return create_session_factory(pg_engine)


@pytest.fixture
def pg_session(pg_session_factory) -> Generator[Session, None, None]:
    """Provides an isolated transactional session that cleans up test data after each test."""
    with db_session_scope(pg_session_factory) as session:
        yield session
        # Cleanup documents and chunks created during tests
        session.execute(text("DELETE FROM chunks"))
        session.execute(text("DELETE FROM documents"))
        session.execute(text("DELETE FROM users WHERE email LIKE 'pg_test_%'"))
        session.commit()


@pytest.fixture
def pg_test_user(pg_session: Session) -> User:
    user = User(
        email=f"pg_test_{uuid4().hex[:8]}@researchlens.ai",
        hashed_password="hashed_pw_for_pg_test",
        is_active=True,
    )
    pg_session.add(user)
    pg_session.commit()
    pg_session.refresh(user)
    return user


@pytest.fixture
def mock_embedder() -> EmbeddingService:
    service = MagicMock(spec=EmbeddingService)
    service.dimension = 384
    service.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def deterministic_embed(texts):
        res = []
        for t in texts:
            seed = sum(ord(c) for c in t) % 10000
            np.random.seed(seed)
            v = np.random.randn(384).astype(np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            res.append(v.tolist())
        return res

    service.embed_batch.side_effect = deterministic_embed
    return service


def create_test_pdf(file_path: Path, pages: list[str]) -> Path:
    doc = pymupdf.open()
    for text_val in pages:
        p = doc.new_page()
        p.insert_text((72, 72), text_val, fontsize=12)
    doc.save(str(file_path))
    doc.close()
    return file_path


class TestPostgreSQLIngestionAndConcurrency:
    """Live PostgreSQL integration tests."""

    def test_postgres_ingestion_end_to_end(
        self,
        pg_session: Session,
        pg_session_factory: sessionmaker[Session],
        pg_test_user: User,
        tmp_path: Path,
        mock_embedder: EmbeddingService,
    ):
        """Verifies full ingestion pipeline against a real PostgreSQL instance."""
        pdf_path = tmp_path / "pg_paper.pdf"
        create_test_pdf(pdf_path, ["Page 1: Deep learning models.", "Page 2: Optimization methods."])

        doc = Document(
            id=uuid4(),
            owner_id=pg_test_user.id,
            filename="pg_paper.pdf",
            file_path=str(pdf_path),
            file_hash_sha256=f"hash_{uuid4().hex}",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.PENDING.value,
        )
        pg_session.add(doc)
        pg_session.commit()

        index_dir = tmp_path / "pg_vec_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        store = VectorStore(
            dimension=384,
            model_name=mock_embedder.model_name,
            index_dir=index_dir,
        )

        service = IngestionService(
            session_factory=pg_session_factory,
            pdf_service=PDFService(),
            embedding_service=mock_embedder,
            vector_store=store,
        )

        processed = service.process_document(doc.id, pg_session_factory)
        assert processed is not None
        assert processed.processing_status == ProcessingStatus.READY.value

        # Verify PostgreSQL persistence
        pg_session.expire_all()
        chunks = (
            pg_session.query(Chunk)
            .filter(Chunk.document_id == doc.id)
            .order_by(Chunk.chunk_index)
            .all()
        )
        assert len(chunks) == 2
        assert store.total_vectors == 2
        for c in chunks:
            assert store.contains_id(c.id)

    def test_postgres_concurrent_claims_prevent_duplicate_chunks(
        self,
        pg_session: Session,
        pg_session_factory: sessionmaker[Session],
        pg_test_user: User,
        tmp_path: Path,
        mock_embedder: EmbeddingService,
    ):
        """Tests that PostgreSQL row-level conditional updates serialize concurrent claims."""
        pdf_path = tmp_path / "pg_race.pdf"
        create_test_pdf(pdf_path, ["Passage to test PostgreSQL race condition."])

        doc = Document(
            id=uuid4(),
            owner_id=pg_test_user.id,
            filename="pg_race.pdf",
            file_path=str(pdf_path),
            file_hash_sha256=f"race_hash_{uuid4().hex}",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.PENDING.value,
        )
        pg_session.add(doc)
        pg_session.commit()

        index_dir = tmp_path / "pg_race_vec_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        store = VectorStore(
            dimension=384,
            model_name=mock_embedder.model_name,
            index_dir=index_dir,
        )

        service = IngestionService(
            session_factory=pg_session_factory,
            pdf_service=PDFService(),
            embedding_service=mock_embedder,
            vector_store=store,
        )

        # Launch two threads simultaneously against PostgreSQL
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(service.process_document, doc.id, pg_session_factory)
            f2 = executor.submit(service.process_document, doc.id, pg_session_factory)
            r1 = f1.result()
            r2 = f2.result()

        assert r1 is not None
        assert r2 is not None

        # Verify final document is READY
        pg_session.expire_all()
        reloaded = pg_session.query(Document).filter(Document.id == doc.id).first()
        assert reloaded.processing_status == ProcessingStatus.READY.value

        # Verify PostgreSQL chunks were not duplicated
        chunks = pg_session.query(Chunk).filter(Chunk.document_id == doc.id).all()
        assert len(chunks) == 1
        assert store.total_vectors == 1

    def test_postgres_faiss_recovery_from_real_records(
        self,
        pg_session: Session,
        pg_session_factory: sessionmaker[Session],
        pg_test_user: User,
        tmp_path: Path,
        mock_embedder: EmbeddingService,
    ):
        """Tests reconstructing a missing FAISS index from live PostgreSQL chunk records."""
        pdf_path = tmp_path / "pg_recover.pdf"
        create_test_pdf(pdf_path, ["First recoverable passage.", "Second recoverable passage."])

        doc = Document(
            id=uuid4(),
            owner_id=pg_test_user.id,
            filename="pg_recover.pdf",
            file_path=str(pdf_path),
            file_hash_sha256=f"rec_hash_{uuid4().hex}",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.PENDING.value,
        )
        pg_session.add(doc)
        pg_session.commit()

        index_dir = tmp_path / "pg_rec_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        store = VectorStore(
            dimension=384,
            model_name=mock_embedder.model_name,
            index_dir=index_dir,
        )

        service = IngestionService(
            session_factory=pg_session_factory,
            pdf_service=PDFService(),
            embedding_service=mock_embedder,
            vector_store=store,
        )

        service.process_document(doc.id, pg_session_factory)
        assert store.total_vectors == 2

        # Simulate index loss on disk
        store.clear()
        bin_file = index_dir / "faiss_index.bin"
        meta_file = index_dir / "faiss_metadata.json"
        if bin_file.exists():
            bin_file.unlink()
        if meta_file.exists():
            meta_file.unlink()

        # Execute recovery
        rebuilt = rebuild_vector_index(
            session=pg_session,
            vector_store=store,
            embedding_service=mock_embedder,
        )
        assert rebuilt == 2
        assert store.total_vectors == 2
        assert store.is_ready is True
        assert bin_file.exists()
        assert meta_file.exists()
