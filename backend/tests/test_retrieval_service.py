"""Unit and integration tests for RetrievalService multi-tenant vector retrieval, scoping, and metadata."""
import uuid
import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from pydantic import ValidationError

from app.core.database import Base
from app.models.chat import ChatScopeType
from app.models.chunk import Chunk
from app.models.collection import Collection, CollectionDocument
from app.models.document import Document, ProcessingStatus
from app.models.user import User
from app.schemas.rag import RetrievedContextChunk
from app.services.retrieval_service import DocumentNotReadyError, RetrievalService
from app.services.vector_store import VectorStore


class DummyEmbeddingService:
    """Deterministic fast embedding service for testing without model weight downloads."""

    def __init__(self, dimension: int = 4):
        self._dimension = dimension
        self.model_name = "dummy-test-embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> np.ndarray:
        # Generate distinct components from text hash
        h = abs(hash(text))
        v0 = float((h % 97) + 1)
        v1 = float(((h // 97) % 89) + 1)
        v2 = float(((h // (97 * 89)) % 83) + 1)
        v3 = float(((h // (97 * 89 * 83)) % 79) + 1)
        vec = np.array([v0, v1, v2, v3], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32)


@pytest.fixture
def db_session():
    """Provides an isolated in-memory SQLite database session with foreign keys enabled."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @sa.event.listens_for(engine, "connect")
    def _enable_fk(conn, _):
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)
    session: Session = session_factory()
    yield session
    session.rollback()
    session.close()
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def dummy_embedder():
    return DummyEmbeddingService(dimension=4)


@pytest.fixture
def vector_store():
    return VectorStore(dimension=4, model_name="dummy-test-embedder")


@pytest.fixture
def test_user(db_session: Session) -> User:
    user = User(email="researcher@example.com", hashed_password="pw")
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def second_user(db_session: Session) -> User:
    user = User(email="second_user@example.com", hashed_password="pw")
    db_session.add(user)
    db_session.commit()
    return user


class TestRetrievalServiceMetadataAndOffsets:
    """Tests verifying accurate retrieval of passage metadata, page numbers, and exact character offsets."""

    def test_retrieval_passage_metadata_and_exact_offsets(
        self, db_session, test_user, dummy_embedder, vector_store
    ):
        """Verifies char_start and char_end are accurately populated from persisted Chunk records."""
        doc = Document(
            owner_id=test_user.id,
            filename="paper.pdf",
            title="Transformer Research Paper",
            file_path="uploads/paper.pdf",
            file_hash_sha256="hash_metadata_test",
            file_size_bytes=2048,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.commit()

        # Chunk with distinct, non-zero offsets referring to physical page text slice
        c1 = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=3,
            content="Self-attention allows the model to associate each word with all other words in the sentence.",
            char_start=120,
            char_end=213,
            token_count=15,
        )
        c2 = Chunk(
            document_id=doc.id,
            chunk_index=1,
            page_number=3,
            content="Feed-forward networks apply two linear transformations with a ReLU activation in between.",
            char_start=214,
            char_end=304,
            token_count=14,
        )
        db_session.add_all([c1, c2])
        db_session.commit()

        v1 = dummy_embedder.embed_text(c1.content)
        v2 = dummy_embedder.embed_text(c2.content)
        vector_store.add_vectors([c1.id, c2.id], [v1, v2])

        retrieval = RetrievalService(dummy_embedder, vector_store)
        results = retrieval.retrieve_relevant_chunks(
            db=db_session,
            query="self-attention mechanism",
            user_id=test_user.id,
            top_k=5,
        )

        assert len(results) == 2
        chunk_map = {r.chunk_id: r for r in results}

        # Validate Chunk 1 metadata and offsets
        r1 = chunk_map[c1.id]
        assert r1.document_id == doc.id
        assert r1.document_title == "Transformer Research Paper"
        assert r1.page_number == 3
        assert r1.content == c1.content
        assert r1.char_start == 120
        assert r1.char_end == 213
        assert r1.char_end > r1.char_start
        assert r1.similarity_score > 0.0

        # Validate Chunk 2 metadata and offsets
        r2 = chunk_map[c2.id]
        assert r2.document_id == doc.id
        assert r2.document_title == "Transformer Research Paper"
        assert r2.page_number == 3
        assert r2.content == c2.content
        assert r2.char_start == 214
        assert r2.char_end == 304
        assert r2.char_end > r2.char_start

    def test_retrieved_context_chunk_offset_validation(self):
        """Verifies RetrievedContextChunk schema enforces non-negative offsets and char_end >= char_start."""
        doc_id = uuid.uuid4()

        # Valid instantiation
        valid_chunk = RetrievedContextChunk(
            chunk_id=1,
            document_id=doc_id,
            content="Valid content",
            page_number=1,
            document_title="Title",
            similarity_score=0.95,
            char_start=10,
            char_end=23,
        )
        assert valid_chunk.char_start == 10
        assert valid_chunk.char_end == 23

        # Inverted offsets rejected
        with pytest.raises(ValidationError):
            RetrievedContextChunk(
                chunk_id=2,
                document_id=doc_id,
                content="Invalid offsets",
                page_number=1,
                document_title="Title",
                similarity_score=0.95,
                char_start=50,
                char_end=20,  # char_end < char_start
            )

        # Negative char_start rejected
        with pytest.raises(ValidationError):
            RetrievedContextChunk(
                chunk_id=3,
                document_id=doc_id,
                content="Negative start",
                page_number=1,
                document_title="Title",
                similarity_score=0.95,
                char_start=-1,
                char_end=10,
            )

    def test_retrieval_title_prefers_title_then_filename_then_fallback(
        self, db_session, test_user, dummy_embedder, vector_store
    ):
        """Verifies title resolution: title if present, filename if title is None/empty, 'Untitled Document' if both missing."""
        doc1 = Document(
            owner_id=test_user.id,
            filename="named_paper.pdf",
            title="Explicit Paper Title",
            file_path="uploads/named_paper.pdf",
            file_hash_sha256="hash_named_paper",
            file_size_bytes=1024,
            processing_status=ProcessingStatus.READY.value,
        )
        doc2 = Document(
            owner_id=test_user.id,
            filename="WAD_EXP_8_experiment.pdf",
            title=None,
            file_path="uploads/wad.pdf",
            file_hash_sha256="hash_wad",
            file_size_bytes=1024,
            processing_status=ProcessingStatus.READY.value,
        )
        doc3 = Document(
            owner_id=test_user.id,
            filename="whitespace_title.pdf",
            title="   ",
            file_path="uploads/ws.pdf",
            file_hash_sha256="hash_ws",
            file_size_bytes=1024,
            processing_status=ProcessingStatus.READY.value,
        )
        doc4 = Document(
            owner_id=test_user.id,
            filename="",
            title=None,
            file_path="uploads/blank.pdf",
            file_hash_sha256="hash_blank",
            file_size_bytes=1024,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add_all([doc1, doc2, doc3, doc4])
        db_session.commit()

        c1 = Chunk(document_id=doc1.id, chunk_index=0, page_number=1, content="alpha", char_start=0, char_end=5, token_count=1)
        c2 = Chunk(document_id=doc2.id, chunk_index=0, page_number=1, content="beta", char_start=0, char_end=4, token_count=1)
        c3 = Chunk(document_id=doc3.id, chunk_index=0, page_number=1, content="gamma", char_start=0, char_end=5, token_count=1)
        c4 = Chunk(document_id=doc4.id, chunk_index=0, page_number=1, content="delta", char_start=0, char_end=5, token_count=1)
        db_session.add_all([c1, c2, c3, c4])
        db_session.commit()

        v1 = dummy_embedder.embed_text(c1.content)
        v2 = dummy_embedder.embed_text(c2.content)
        v3 = dummy_embedder.embed_text(c3.content)
        v4 = dummy_embedder.embed_text(c4.content)
        vector_store.add_vectors([c1.id, c2.id, c3.id, c4.id], [v1, v2, v3, v4])

        retrieval = RetrievalService(dummy_embedder, vector_store)
        results = retrieval.retrieve_relevant_chunks(
            db=db_session,
            query="test query",
            user_id=test_user.id,
            top_k=10,
        )

        title_map = {r.chunk_id: r.document_title for r in results}
        assert title_map[c1.id] == "Explicit Paper Title"
        assert title_map[c2.id] == "WAD_EXP_8_experiment.pdf"
        assert title_map[c3.id] == "whitespace_title.pdf"
        assert title_map[c4.id] == "Untitled Document"


class TestRetrievalServiceRankingAndFiltering:
    """Tests verifying similarity ranking, threshold filtering, and top-k bounding."""

    def test_ranking_and_top_k(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies chunks are sorted strictly by descending score and sliced to top_k."""
        doc = Document(
            owner_id=test_user.id,
            filename="ranking.pdf",
            title="Ranking Doc",
            file_path="uploads/ranking.pdf",
            file_hash_sha256="hash_rank",
            file_size_bytes=1000,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.commit()

        chunks = [
            Chunk(
                document_id=doc.id,
                chunk_index=i,
                page_number=1,
                content=f"Passage numbered {i} with distinct text content.",
                char_start=i * 50,
                char_end=(i + 1) * 50,
                token_count=10,
            )
            for i in range(5)
        ]
        db_session.add_all(chunks)
        db_session.commit()

        for c in chunks:
            v = dummy_embedder.embed_text(c.content)
            vector_store.add_vectors([c.id], [v])

        retrieval = RetrievalService(dummy_embedder, vector_store)
        top_2 = retrieval.retrieve_relevant_chunks(
            db=db_session,
            query=chunks[2].content,
            user_id=test_user.id,
            top_k=2,
        )

        assert len(top_2) == 2
        # Verify descending similarity scores
        assert top_2[0].similarity_score >= top_2[1].similarity_score
        # Exact match should be top
        assert top_2[0].chunk_id == chunks[2].id

    def test_similarity_threshold_filtering(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies chunks below similarity_threshold are excluded."""
        doc = Document(
            owner_id=test_user.id,
            filename="threshold.pdf",
            file_path="uploads/threshold.pdf",
            file_hash_sha256="hash_thresh",
            file_size_bytes=1000,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.commit()

        c = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content="General text content.",
            char_start=0,
            char_end=21,
            token_count=5,
        )
        db_session.add(c)
        db_session.commit()
        v = dummy_embedder.embed_text(c.content)
        vector_store.add_vectors([c.id], [v])

        retrieval = RetrievalService(dummy_embedder, vector_store)

        # High threshold eliminates result
        results = retrieval.retrieve_relevant_chunks(
            db=db_session,
            query="Unrelated search query",
            user_id=test_user.id,
            similarity_threshold=0.9999,
        )
        assert results == []


class TestRetrievalServiceSecurityAndIsolation:
    """Tests verifying multi-tenant isolation, document readiness, and scope enforcement."""

    def test_cross_tenant_isolation(self, db_session, test_user, second_user, dummy_embedder, vector_store):
        """Verifies User A cannot retrieve User B's chunks under any query."""
        doc_a = Document(
            owner_id=test_user.id,
            filename="doc_a.pdf",
            file_path="doc_a.pdf",
            file_hash_sha256="ha",
            file_size_bytes=100,
            processing_status=ProcessingStatus.READY.value,
        )
        doc_b = Document(
            owner_id=second_user.id,
            filename="doc_b.pdf",
            file_path="doc_b.pdf",
            file_hash_sha256="hb",
            file_size_bytes=100,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add_all([doc_a, doc_b])
        db_session.commit()

        c_a = Chunk(document_id=doc_a.id, chunk_index=0, page_number=1, content="User A confidential text", char_start=0, char_end=25, token_count=5)
        c_b = Chunk(document_id=doc_b.id, chunk_index=0, page_number=1, content="User B secret proprietary formula", char_start=0, char_end=34, token_count=5)
        db_session.add_all([c_a, c_b])
        db_session.commit()

        vector_store.add_vectors([c_a.id, c_b.id], [dummy_embedder.embed_text(c_a.content), dummy_embedder.embed_text(c_b.content)])

        retrieval = RetrievalService(dummy_embedder, vector_store)

        # User A searches specifically for User B's secrets
        results = retrieval.retrieve_relevant_chunks(
            db=db_session,
            query="secret proprietary formula",
            user_id=test_user.id,
        )

        for r in results:
            assert r.document_id != doc_b.id
            assert r.chunk_id != c_b.id

    def test_document_scoped_query(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies DOCUMENT scope returns only chunks from the specified document."""
        doc1 = Document(owner_id=test_user.id, filename="d1.pdf", file_path="d1.pdf", file_hash_sha256="h1", file_size_bytes=100, processing_status=ProcessingStatus.READY.value)
        doc2 = Document(owner_id=test_user.id, filename="d2.pdf", file_path="d2.pdf", file_hash_sha256="h2", file_size_bytes=100, processing_status=ProcessingStatus.READY.value)
        db_session.add_all([doc1, doc2])
        db_session.commit()

        c1 = Chunk(document_id=doc1.id, chunk_index=0, page_number=1, content="Doc 1 text", char_start=0, char_end=10, token_count=3)
        c2 = Chunk(document_id=doc2.id, chunk_index=0, page_number=1, content="Doc 2 text", char_start=0, char_end=10, token_count=3)
        db_session.add_all([c1, c2])
        db_session.commit()

        vector_store.add_vectors([c1.id, c2.id], [dummy_embedder.embed_text(c1.content), dummy_embedder.embed_text(c2.content)])

        retrieval = RetrievalService(dummy_embedder, vector_store)
        results = retrieval.retrieve_relevant_chunks(
            db=db_session,
            query="text",
            user_id=test_user.id,
            scope_type=ChatScopeType.DOCUMENT,
            scope_id=doc1.id,
        )

        assert len(results) == 1
        assert results[0].document_id == doc1.id

    def test_document_scoped_unready_raises_document_not_ready_error(
        self, db_session, test_user, dummy_embedder, vector_store
    ):
        """Verifies querying an unready document raises DocumentNotReadyError with status."""
        doc = Document(
            owner_id=test_user.id,
            filename="unready.pdf",
            file_path="unready.pdf",
            file_hash_sha256="hunready",
            file_size_bytes=100,
            processing_status=ProcessingStatus.EXTRACTING.value,
        )
        db_session.add(doc)
        db_session.commit()

        retrieval = RetrievalService(dummy_embedder, vector_store)
        with pytest.raises(DocumentNotReadyError) as exc_info:
            retrieval.retrieve_relevant_chunks(
                db=db_session,
                query="query",
                user_id=test_user.id,
                scope_type=ChatScopeType.DOCUMENT,
                scope_id=doc.id,
            )

        assert exc_info.value.filename == "unready.pdf"
        assert exc_info.value.processing_status == ProcessingStatus.EXTRACTING.value

    def test_document_scoped_unowned_raises_value_error(
        self, db_session, test_user, second_user, dummy_embedder, vector_store
    ):
        """Verifies querying another user's document raises ValueError (not found or not owned)."""
        doc_b = Document(
            owner_id=second_user.id,
            filename="b.pdf",
            file_path="b.pdf",
            file_hash_sha256="hb",
            file_size_bytes=100,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc_b)
        db_session.commit()

        retrieval = RetrievalService(dummy_embedder, vector_store)
        with pytest.raises(ValueError, match="not found or not owned"):
            retrieval.retrieve_relevant_chunks(
                db=db_session,
                query="query",
                user_id=test_user.id,
                scope_type=ChatScopeType.DOCUMENT,
                scope_id=doc_b.id,
            )

    def test_collection_scoped_query(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies COLLECTION scope returns only chunks from documents inside the collection."""
        col = Collection(owner_id=test_user.id, name="AI Research")
        doc_in = Document(owner_id=test_user.id, filename="in.pdf", file_path="in.pdf", file_hash_sha256="hin", file_size_bytes=100, processing_status=ProcessingStatus.READY.value)
        doc_out = Document(owner_id=test_user.id, filename="out.pdf", file_path="out.pdf", file_hash_sha256="hout", file_size_bytes=100, processing_status=ProcessingStatus.READY.value)
        db_session.add_all([col, doc_in, doc_out])
        db_session.commit()

        link = CollectionDocument(collection_id=col.id, document_id=doc_in.id)
        c_in = Chunk(document_id=doc_in.id, chunk_index=0, page_number=1, content="In collection text", char_start=0, char_end=18, token_count=4)
        c_out = Chunk(document_id=doc_out.id, chunk_index=0, page_number=1, content="Out of collection text", char_start=0, char_end=22, token_count=5)
        db_session.add_all([link, c_in, c_out])
        db_session.commit()

        vector_store.add_vectors([c_in.id, c_out.id], [dummy_embedder.embed_text(c_in.content), dummy_embedder.embed_text(c_out.content)])

        retrieval = RetrievalService(dummy_embedder, vector_store)
        results = retrieval.retrieve_relevant_chunks(
            db=db_session,
            query="text",
            user_id=test_user.id,
            scope_type=ChatScopeType.COLLECTION,
            scope_id=col.id,
        )

        assert len(results) == 1
        assert results[0].document_id == doc_in.id

    def test_collection_scoped_unowned_raises_value_error(
        self, db_session, test_user, second_user, dummy_embedder, vector_store
    ):
        """Verifies querying another user's collection raises ValueError."""
        col_b = Collection(owner_id=second_user.id, name="Other Collection")
        db_session.add(col_b)
        db_session.commit()

        retrieval = RetrievalService(dummy_embedder, vector_store)
        with pytest.raises(ValueError, match="not found or not owned"):
            retrieval.retrieve_relevant_chunks(
                db=db_session,
                query="query",
                user_id=test_user.id,
                scope_type=ChatScopeType.COLLECTION,
                scope_id=col_b.id,
            )

    def test_global_scope_ignores_unready_and_dangling(
        self, db_session, test_user, dummy_embedder, vector_store
    ):
        """Verifies chunks from unready documents and dangling FAISS IDs are excluded in GLOBAL retrieval."""
        doc_pending = Document(owner_id=test_user.id, filename="pending.pdf", file_path="p.pdf", file_hash_sha256="hp", file_size_bytes=100, processing_status=ProcessingStatus.PENDING.value)
        db_session.add(doc_pending)
        db_session.commit()

        c = Chunk(document_id=doc_pending.id, chunk_index=0, page_number=1, content="Pending chunk text", char_start=0, char_end=18, token_count=4)
        db_session.add(c)
        db_session.commit()

        # Add c.id (unready) and 99999 (dangling, no DB record)
        v = dummy_embedder.embed_text(c.content)
        vector_store.add_vectors([c.id, 99999], [v, v])

        retrieval = RetrievalService(dummy_embedder, vector_store)
        results = retrieval.retrieve_relevant_chunks(
            db=db_session,
            query="chunk text",
            user_id=test_user.id,
        )
        assert results == []

    def test_scope_and_input_validation(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies invalid queries, invalid scope combinations, and invalid IDs raise ValueError."""
        retrieval = RetrievalService(dummy_embedder, vector_store)

        with pytest.raises(ValueError, match="non-empty string"):
            retrieval.retrieve_relevant_chunks(db=db_session, query="   ", user_id=test_user.id)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            retrieval.retrieve_relevant_chunks(db=db_session, query="query", user_id=test_user.id, top_k=0)

        with pytest.raises(ValueError, match="similarity_threshold must be a float"):
            retrieval.retrieve_relevant_chunks(db=db_session, query="query", user_id=test_user.id, similarity_threshold="invalid")

        with pytest.raises(ValueError, match="user_id cannot be None"):
            retrieval.retrieve_relevant_chunks(db=db_session, query="query", user_id=None)

        with pytest.raises(ValueError, match="scope_id must be None when scope_type is GLOBAL"):
            retrieval.retrieve_relevant_chunks(
                db=db_session, query="query", user_id=test_user.id, scope_type=ChatScopeType.GLOBAL, scope_id=uuid.uuid4()
            )

        with pytest.raises(ValueError, match="scope_id is required"):
            retrieval.retrieve_relevant_chunks(
                db=db_session, query="query", user_id=test_user.id, scope_type=ChatScopeType.DOCUMENT, scope_id=None
            )


class TestRetrievalServiceDisambiguation:
    """Tests verifying constraint extraction, structural entity promotion, and table/dataset disambiguation."""

    def test_table_and_dataset_disambiguation_ranks_target_chunk_first(
        self, db_session, test_user, dummy_embedder, vector_store
    ):
        """Verifies a query for Table 6 on CICIDS2017 without SMOTE ranks Chunk 1 over Chunk 2."""
        doc = Document(
            owner_id=test_user.id,
            filename="paper.pdf",
            file_path="paper.pdf",
            file_hash_sha256="h123",
            file_size_bytes=5000,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.commit()

        # Chunk 1: Target Table 6 on CICIDS2017 without SMOTE
        c1 = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=7,
            content=(
                "Table 6. Performance metrics of all models on the CICIDS2017 dataset (without SMOTE).\n"
                "Naïve Bayes Accuracy 0.8565 Precision 0.6364 Recall 0.9742 FAR 0.1820"
            ),
            char_start=0,
            char_end=150,
            token_count=35,
        )
        # Chunk 2: Conflicting Table 9 on NSL-KDD
        c2 = Chunk(
            document_id=doc.id,
            chunk_index=1,
            page_number=16,
            content=(
                "Table 9. Statistical summary on NSL-KDD dataset (without SMOTE).\n"
                "Naïve Bayes Mean 0.5649 FAR 0.0065"
            ),
            char_start=0,
            char_end=120,
            token_count=30,
        )
        db_session.add_all([c1, c2])
        db_session.commit()

        v1 = dummy_embedder.embed_text(c1.content)
        v2 = dummy_embedder.embed_text(c2.content)
        vector_store.add_vectors([c1.id, c2.id], [v1, v2])

        retrieval = RetrievalService(dummy_embedder, vector_store)
        results = retrieval.retrieve_relevant_chunks(
            db=db_session,
            query="According to Table 6, what is the FAR of Naïve Bayes on CICIDS2017 without SMOTE?",
            user_id=test_user.id,
            top_k=2,
        )

        assert len(results) == 2
        assert results[0].chunk_id == c1.id
        assert "Table 6" in results[0].content
        assert "0.1820" in results[0].content

    def test_experimental_condition_disambiguation(
        self, db_session, test_user, dummy_embedder, vector_store
    ):
        """Verifies query specifying 'without SMOTE' ranks negative condition chunk above positive condition chunk."""
        doc = Document(
            owner_id=test_user.id,
            filename="eval.pdf",
            file_path="eval.pdf",
            file_hash_sha256="h456",
            file_size_bytes=4000,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.commit()

        c_with = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=2,
            content="Evaluation with SMOTE data balancing yields high precision of 0.9850 and low FAR of 0.0065.",
            char_start=0,
            char_end=95,
            token_count=20,
        )
        c_without = Chunk(
            document_id=doc.id,
            chunk_index=1,
            page_number=3,
            content="Evaluation without SMOTE data balancing shows lower precision of 0.6364 and high FAR of 0.1820.",
            char_start=0,
            char_end=98,
            token_count=20,
        )
        db_session.add_all([c_with, c_without])
        db_session.commit()

        v_with = dummy_embedder.embed_text(c_with.content)
        v_without = dummy_embedder.embed_text(c_without.content)
        vector_store.add_vectors([c_with.id, c_without.id], [v_with, v_without])

        retrieval = RetrievalService(dummy_embedder, vector_store)
        results = retrieval.retrieve_relevant_chunks(
            db=db_session,
            query="What is the precision and FAR without SMOTE?",
            user_id=test_user.id,
            top_k=2,
        )

        assert len(results) == 2
        assert results[0].chunk_id == c_without.id
        assert "without SMOTE" in results[0].content

    def test_structural_entity_lexical_expansion_retrieves_unindexed_table(
        self, db_session, test_user, dummy_embedder, vector_store
    ):
        """Verifies lexical expansion finds a chunk explicitly mentioning Table 12 even if not in vector index."""
        doc = Document(
            owner_id=test_user.id,
            filename="tables.pdf",
            file_path="tables.pdf",
            file_hash_sha256="h789",
            file_size_bytes=3000,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.commit()

        c_indexed = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content="General overview of intrusion detection models and benchmark architectures.",
            char_start=0,
            char_end=80,
            token_count=15,
        )
        c_unindexed = Chunk(
            document_id=doc.id,
            chunk_index=1,
            page_number=5,
            content="Table 12. Cross-dataset hyperparameter configurations and training epochs.",
            char_start=0,
            char_end=76,
            token_count=15,
        )
        db_session.add_all([c_indexed, c_unindexed])
        db_session.commit()

        # Only add c_indexed to vector store
        v_indexed = dummy_embedder.embed_text(c_indexed.content)
        vector_store.add_vectors([c_indexed.id], [v_indexed])

        retrieval = RetrievalService(dummy_embedder, vector_store)
        results = retrieval.retrieve_relevant_chunks(
            db=db_session,
            query="According to Table 12, what are the training epochs?",
            user_id=test_user.id,
            top_k=2,
        )

        assert len(results) >= 1
        assert results[0].chunk_id == c_unindexed.id
        assert "Table 12" in results[0].content

    def test_structural_entity_lexical_expansion_matches_punctuation_variants(
        self, db_session, test_user, dummy_embedder, vector_store
    ):
        """Verifies lexical expansion matches structural entity formats like 'Table. 6' and 'Table6'."""
        doc = Document(
            owner_id=test_user.id,
            filename="tables_punct.pdf",
            file_path="tables_punct.pdf",
            file_hash_sha256="hpunct",
            file_size_bytes=3000,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.commit()

        c_punct = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=6,
            content="Table. 6: Performance metrics on CICIDS2017 dataset without SMOTE shows FAR of 0.1820.",
            char_start=0,
            char_end=88,
            token_count=18,
        )
        db_session.add(c_punct)
        db_session.commit()

        # Do not add to vector index to strictly test lexical path
        retrieval = RetrievalService(dummy_embedder, vector_store)
        results = retrieval.retrieve_relevant_chunks(
            db=db_session,
            query="What is the FAR in Table 6?",
            user_id=test_user.id,
            top_k=2,
        )

        assert len(results) >= 1
        assert results[0].chunk_id == c_punct.id
        assert "Table. 6" in results[0].content

    def test_global_scope_prevents_faiss_candidate_dilution_across_tenants(
        self, db_session, test_user, second_user, dummy_embedder, vector_store
    ):
        """Verifies that in GLOBAL scope, FAISS candidate selection is restricted to the requesting user's

        chunks, preventing high-similarity chunks from other tenants from crowding out the user's candidates.
        """
        # User B (other tenant) has 45 chunks with identical high-scoring vectors
        doc_b = Document(
            owner_id=second_user.id,
            filename="user_b.pdf",
            file_path="user_b.pdf",
            file_hash_sha256="hb_dilution",
            file_size_bytes=5000,
            processing_status=ProcessingStatus.READY.value,
        )
        # User A (test user) has 1 chunk
        doc_a = Document(
            owner_id=test_user.id,
            filename="user_a.pdf",
            file_path="user_a.pdf",
            file_hash_sha256="ha_dilution",
            file_size_bytes=2000,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add_all([doc_b, doc_a])
        db_session.commit()

        b_chunks = [
            Chunk(
                document_id=doc_b.id,
                chunk_index=i,
                page_number=1,
                content=f"Intrusion detection benchmark query target vector {i}",
                char_start=0,
                char_end=50,
                token_count=10,
            )
            for i in range(45)
        ]
        a_chunk = Chunk(
            document_id=doc_a.id,
            chunk_index=0,
            page_number=1,
            content="Intrusion detection benchmark query target vector owned by User A",
            char_start=0,
            char_end=65,
            token_count=12,
        )
        db_session.add_all(b_chunks + [a_chunk])
        db_session.commit()

        # Add all 45 of User B's chunks and User A's chunk to the shared FAISS vector store
        b_ids = [c.id for c in b_chunks]
        b_vecs = [dummy_embedder.embed_text(c.content) for c in b_chunks]
        vector_store.add_vectors(b_ids, b_vecs)

        a_id = a_chunk.id
        a_vec = dummy_embedder.embed_text(a_chunk.content)
        vector_store.add_vectors([a_id], [a_vec])

        retrieval = RetrievalService(dummy_embedder, vector_store)

        # In GLOBAL scope, User A must retrieve their own chunk despite 45 candidates from User B
        results = retrieval.retrieve_relevant_chunks(
            db=db_session,
            query="Intrusion detection benchmark query target vector",
            user_id=test_user.id,
            scope_type=ChatScopeType.GLOBAL,
            top_k=5,
        )

        assert len(results) == 1
        assert results[0].chunk_id == a_id
        assert results[0].document_id == doc_a.id
