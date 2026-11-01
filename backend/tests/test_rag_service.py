"""Unit and integration tests for RAGService orchestration, multi-tenant isolation, and scoping."""
import uuid
import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.chat import ChatScopeType
from app.models.chunk import Chunk
from app.models.collection import Collection, CollectionDocument
from app.models.document import Document, ProcessingStatus
from app.models.user import User
from app.services.embedding_service import EmbeddingService
from app.services.llm.exceptions import LLMProviderError, LLMQuotaExceededError
from app.services.llm.mock_provider import MockLLMProvider
from app.services.rag_service import RAGService
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
        # Deterministic normalized vector derived from text hash
        val = float(abs(hash(text)) % 100 + 1)
        vec = np.array([val, val * 0.5, val * 0.25, val * 0.1], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32)


@pytest.fixture
def db_session():
    """Provides an isolated SQLite database session with foreign keys enabled."""
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
    # In-memory FAISS store with 4 dimensions matching dummy embedder
    return VectorStore(dimension=4, model_name="dummy-test-embedder")


@pytest.fixture
def test_user(db_session: Session) -> User:
    user = User(email="researcher@example.com", hashed_password="pw")
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def second_user(db_session: Session) -> User:
    user = User(email="intruder@example.com", hashed_password="pw")
    db_session.add(user)
    db_session.commit()
    return user


class TestRAGServiceHappyPath:
    """Tests verifying the standard grounded RAG workflow."""

    def test_happy_path_rag_query(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies end-to-end RAG query retrieval, synthesis, and citation snapshot creation."""
        # 1. Setup Document and Chunks in DB
        doc = Document(
            owner_id=test_user.id,
            filename="transformer.pdf",
            title="Attention Is All You Need",
            file_path="uploads/transformer.pdf",
            file_hash_sha256="hash123",
            file_size_bytes=1000,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.commit()

        c1 = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content="The Transformer architecture is based entirely on attention mechanisms.",
            char_start=0,
            char_end=72,
            token_count=10,
        )
        c2 = Chunk(
            document_id=doc.id,
            chunk_index=1,
            page_number=2,
            content="Multi-head attention yields significant empirical gains in translation.",
            char_start=73,
            char_end=145,
            token_count=10,
        )
        db_session.add_all([c1, c2])
        db_session.commit()

        # 2. Add vectors to FAISS index
        v1 = dummy_embedder.embed_text(c1.content)
        v2 = dummy_embedder.embed_text(c2.content)
        vector_store.add_vectors([c1.id, c2.id], [v1, v2])

        # 3. Create RAGService with MockLLMProvider
        llm_provider = MockLLMProvider()
        rag_service = RAGService(
            embedding_service=dummy_embedder,
            vector_store=vector_store,
            llm_provider=llm_provider,
        )

        # 4. Execute query
        response = rag_service.answer_query(
            db=db_session,
            query="How does the transformer attention mechanism work?",
            user_id=test_user.id,
            top_k=2,
        )

        # 5. Verify result
        assert response.status == "SUCCESS"
        assert response.retrieved_chunk_count == 2
        assert len(response.citations) == 2
        assert response.unknown_chunk_ids == []
        assert response.malformed_markers == []

        # Check citations correspond to c1 and c2
        citation_ids = {c.chunk_id for c in response.citations}
        assert citation_ids == {c1.id, c2.id}
        assert response.citations[0].document_title == "Attention Is All You Need"
        assert response.citations[0].page_number in (1, 2)
        assert response.citations[0].document_id == doc.id
        assert response.citations[1].document_id == doc.id


class TestRAGServiceTenantIsolationAndScoping:
    """Tests verifying multi-tenant isolation and conversational scope filtering."""

    def test_cross_tenant_isolation(self, db_session, test_user, second_user, dummy_embedder, vector_store):
        """Verifies User A never retrieves or cites chunks belonging to User B."""
        # Doc 1 for User A
        doc_a = Document(
            owner_id=test_user.id,
            filename="user_a.pdf",
            title="User A Paper",
            file_path="uploads/user_a.pdf",
            file_hash_sha256="hash_a",
            file_size_bytes=1000,
            processing_status=ProcessingStatus.READY.value,
        )
        # Doc 2 for User B
        doc_b = Document(
            owner_id=second_user.id,
            filename="user_b.pdf",
            title="User B Secret",
            file_path="uploads/user_b.pdf",
            file_hash_sha256="hash_b",
            file_size_bytes=1000,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add_all([doc_a, doc_b])
        db_session.commit()

        c_a = Chunk(
            document_id=doc_a.id,
            chunk_index=0,
            page_number=1,
            content="User A public passage.",
            char_start=0,
            char_end=22,
            token_count=5,
        )
        c_b = Chunk(
            document_id=doc_b.id,
            chunk_index=0,
            page_number=1,
            content="User B confidential passage.",
            char_start=0,
            char_end=29,
            token_count=5,
        )
        db_session.add_all([c_a, c_b])
        db_session.commit()

        # Add both to shared vector store
        v_a = dummy_embedder.embed_text(c_a.content)
        v_b = dummy_embedder.embed_text(c_b.content)
        vector_store.add_vectors([c_a.id, c_b.id], [v_a, v_b])

        rag_service = RAGService(
            embedding_service=dummy_embedder,
            vector_store=vector_store,
            llm_provider=MockLLMProvider(),
        )

        # Query as User A
        resp_a = rag_service.answer_query(
            db=db_session,
            query="Tell me about confidential passage",
            user_id=test_user.id,
        )

        # Only User A's chunk must be returned
        assert resp_a.retrieved_chunk_count == 1
        assert len(resp_a.citations) == 1
        assert resp_a.citations[0].chunk_id == c_a.id
        assert "User B" not in resp_a.answer

    def test_document_scoped_query(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies DOCUMENT scope only retrieves chunks from the specified document."""
        doc1 = Document(
            owner_id=test_user.id,
            filename="doc1.pdf",
            title="Doc 1",
            file_path="doc1.pdf",
            file_hash_sha256="h1",
            file_size_bytes=100,
            processing_status=ProcessingStatus.READY.value,
        )
        doc2 = Document(
            owner_id=test_user.id,
            filename="doc2.pdf",
            title="Doc 2",
            file_path="doc2.pdf",
            file_hash_sha256="h2",
            file_size_bytes=100,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add_all([doc1, doc2])
        db_session.commit()

        c1 = Chunk(document_id=doc1.id, chunk_index=0, page_number=1, content="Doc 1 text", char_start=0, char_end=10, token_count=3)
        c2 = Chunk(document_id=doc2.id, chunk_index=0, page_number=1, content="Doc 2 text", char_start=0, char_end=10, token_count=3)
        db_session.add_all([c1, c2])
        db_session.commit()

        v1 = dummy_embedder.embed_text(c1.content)
        v2 = dummy_embedder.embed_text(c2.content)
        vector_store.add_vectors([c1.id, c2.id], [v1, v2])

        rag_service = RAGService(
            embedding_service=dummy_embedder,
            vector_store=vector_store,
            llm_provider=MockLLMProvider(),
        )

        # Query scoped to doc1
        resp = rag_service.answer_query(
            db=db_session,
            query="test",
            user_id=test_user.id,
            scope_type=ChatScopeType.DOCUMENT,
            scope_id=doc1.id,
        )

        assert resp.retrieved_chunk_count == 1
        assert resp.citations[0].chunk_id == c1.id
        assert resp.citations[0].document_id == doc1.id

    def test_collection_scoped_query(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies COLLECTION scope only retrieves documents associated with that collection."""
        col = Collection(owner_id=test_user.id, name="NLP Collection")
        doc_in = Document(owner_id=test_user.id, filename="in.pdf", file_path="in.pdf", file_hash_sha256="hin", file_size_bytes=100, processing_status=ProcessingStatus.READY.value)
        doc_out = Document(owner_id=test_user.id, filename="out.pdf", file_path="out.pdf", file_hash_sha256="hout", file_size_bytes=100, processing_status=ProcessingStatus.READY.value)
        db_session.add_all([col, doc_in, doc_out])
        db_session.commit()

        assoc = CollectionDocument(collection_id=col.id, document_id=doc_in.id)
        db_session.add(assoc)
        db_session.commit()

        c_in = Chunk(document_id=doc_in.id, chunk_index=0, page_number=1, content="In collection", char_start=0, char_end=13, token_count=3)
        c_out = Chunk(document_id=doc_out.id, chunk_index=0, page_number=1, content="Out of collection", char_start=0, char_end=17, token_count=4)
        db_session.add_all([c_in, c_out])
        db_session.commit()

        v_in = dummy_embedder.embed_text(c_in.content)
        v_out = dummy_embedder.embed_text(c_out.content)
        vector_store.add_vectors([c_in.id, c_out.id], [v_in, v_out])

        rag_service = RAGService(
            embedding_service=dummy_embedder,
            vector_store=vector_store,
            llm_provider=MockLLMProvider(),
        )

        resp = rag_service.answer_query(
            db=db_session,
            query="find text",
            user_id=test_user.id,
            scope_type=ChatScopeType.COLLECTION,
            scope_id=col.id,
        )

        assert resp.retrieved_chunk_count == 1
        assert resp.citations[0].chunk_id == c_in.id
        assert resp.citations[0].document_id == doc_in.id


class TestRAGServiceEarlyReturnAndErrors:
    """Tests verifying early exit without provider calls, parameter validation, and error propagation."""

    def test_empty_vector_store_early_return_zero_llm_calls(self, db_session, test_user, dummy_embedder, vector_store, monkeypatch):
        """Verifies early exit returns fallback with ZERO LLM calls when index is empty."""
        llm = MockLLMProvider()
        called = False

        def mock_generate(*args, **kwargs):
            nonlocal called
            called = True
            return llm.generate(*args, **kwargs)

        monkeypatch.setattr(llm, "generate", mock_generate)

        rag_service = RAGService(
            embedding_service=dummy_embedder,
            vector_store=vector_store,
            llm_provider=llm,
        )

        resp = rag_service.answer_query(
            db=db_session,
            query="Query against empty index",
            user_id=test_user.id,
        )

        assert resp.status == "NO_RELEVANT_CONTEXT"
        assert resp.citations == []
        assert resp.retrieved_chunk_count == 0
        assert not called, "LLM provider generate() should NOT be called when retrieval is empty"

    def test_similarity_threshold_filtering(self, db_session, test_user, dummy_embedder, vector_store, monkeypatch):
        """Verifies chunks below similarity threshold are filtered out; if all filtered, early returns."""
        doc = Document(
            owner_id=test_user.id,
            filename="paper.pdf",
            file_path="paper.pdf",
            file_hash_sha256="hp",
            file_size_bytes=100,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.commit()

        c = Chunk(document_id=doc.id, chunk_index=0, page_number=1, content="Low similarity text", char_start=0, char_end=19, token_count=4)
        db_session.add(c)
        db_session.commit()

        v = dummy_embedder.embed_text(c.content)
        vector_store.add_vectors([c.id], [v])

        llm = MockLLMProvider()
        called = False

        def mock_generate(*args, **kwargs):
            nonlocal called
            called = True
            return llm.generate(*args, **kwargs)

        monkeypatch.setattr(llm, "generate", mock_generate)

        rag_service = RAGService(
            embedding_service=dummy_embedder,
            vector_store=vector_store,
            llm_provider=llm,
        )

        # Set an impossibly high similarity threshold (e.g. 1.5)
        resp = rag_service.answer_query(
            db=db_session,
            query="Some query",
            user_id=test_user.id,
            similarity_threshold=1.5,
        )

        assert resp.status == "NO_RELEVANT_CONTEXT"
        assert not called, "LLM provider must not be invoked when all chunks fall below threshold"

    def test_provider_error_propagation(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies typed LLM provider errors (e.g. quota, provider failure) bubble up cleanly."""
        doc = Document(
            owner_id=test_user.id,
            filename="doc.pdf",
            file_path="doc.pdf",
            file_hash_sha256="hd",
            file_size_bytes=100,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.commit()

        c = Chunk(document_id=doc.id, chunk_index=0, page_number=1, content="Some text", char_start=0, char_end=9, token_count=2)
        db_session.add(c)
        db_session.commit()

        v = dummy_embedder.embed_text(c.content)
        vector_store.add_vectors([c.id], [v])

        # Provider simulated quota failure
        quota_llm = MockLLMProvider(simulate_error="quota")
        rag_service = RAGService(
            embedding_service=dummy_embedder,
            vector_store=vector_store,
            llm_provider=quota_llm,
        )

        with pytest.raises(LLMQuotaExceededError):
            rag_service.answer_query(db=db_session, query="query", user_id=test_user.id)

    def test_invalid_parameters_raise_value_errors(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies invalid query, top_k, or missing scope_id raise ValueError."""
        rag_service = RAGService(
            embedding_service=dummy_embedder,
            vector_store=vector_store,
            llm_provider=MockLLMProvider(),
        )

        with pytest.raises(ValueError, match="Query must be a non-empty string"):
            rag_service.answer_query(db=db_session, query="   ", user_id=test_user.id)

        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            rag_service.answer_query(db=db_session, query="query", user_id=test_user.id, top_k=0)

        with pytest.raises(ValueError, match="scope_id is required"):
            rag_service.answer_query(db=db_session, query="query", user_id=test_user.id, scope_type=ChatScopeType.DOCUMENT, scope_id=None)


class TestRAGServiceSecurityRegressions:
    """Security regression tests verifying scope validation, user_id validation, and candidate authorization."""

    def test_invalid_scope_string_rejected_not_global(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies invalid scope string is rejected and never falls through to GLOBAL."""
        rag = RAGService(dummy_embedder, vector_store, MockLLMProvider())

        with pytest.raises(ValueError, match="Invalid scope_type"):
            rag.answer_query(db=db_session, query="test", user_id=test_user.id, scope_type="INVALID")

        with pytest.raises(ValueError, match="scope_type cannot be None"):
            rag.answer_query(db=db_session, query="test", user_id=test_user.id, scope_type=None)

        with pytest.raises(ValueError, match="Unsupported scope_type type"):
            rag.answer_query(db=db_session, query="test", user_id=test_user.id, scope_type=123)

    def test_valid_scope_string_converted_to_enum(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies string values matching ChatScopeType are accepted and converted."""
        rag = RAGService(dummy_embedder, vector_store, MockLLMProvider())
        resp = rag.answer_query(db=db_session, query="test", user_id=test_user.id, scope_type="GLOBAL")
        assert resp.status == "NO_RELEVANT_CONTEXT"

    def test_global_scope_with_scope_id_rejected(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies supplying scope_id for GLOBAL scope is rejected."""
        rag = RAGService(dummy_embedder, vector_store, MockLLMProvider())
        with pytest.raises(ValueError, match="scope_id must be None when scope_type is GLOBAL"):
            rag.answer_query(
                db=db_session,
                query="test",
                user_id=test_user.id,
                scope_type=ChatScopeType.GLOBAL,
                scope_id=uuid.uuid4(),
            )

    def test_scoped_query_malformed_scope_ids_rejected(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies malformed string scope IDs are rejected."""
        rag = RAGService(dummy_embedder, vector_store, MockLLMProvider())
        with pytest.raises(ValueError, match="scope_id must be a valid UUID"):
            rag.answer_query(
                db=db_session,
                query="test",
                user_id=test_user.id,
                scope_type=ChatScopeType.DOCUMENT,
                scope_id="bad-uuid",
            )
        with pytest.raises(ValueError, match="scope_id must be a valid UUID"):
            rag.answer_query(
                db=db_session,
                query="test",
                user_id=test_user.id,
                scope_type=ChatScopeType.COLLECTION,
                scope_id="bad-uuid",
            )

    def test_other_user_document_denied(self, db_session, test_user, second_user, dummy_embedder, vector_store):
        """Verifies User A cannot query with User B's document ID."""
        doc_b = Document(
            owner_id=second_user.id,
            filename="secret_b.pdf",
            file_path="secret_b.pdf",
            file_hash_sha256="hb",
            file_size_bytes=100,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc_b)
        db_session.commit()

        rag = RAGService(dummy_embedder, vector_store, MockLLMProvider())
        with pytest.raises(ValueError, match="not found or not owned by user"):
            rag.answer_query(
                db=db_session,
                query="test",
                user_id=test_user.id,
                scope_type=ChatScopeType.DOCUMENT,
                scope_id=doc_b.id,
            )

    def test_other_user_collection_denied(self, db_session, test_user, second_user, dummy_embedder, vector_store):
        """Verifies User A cannot query with User B's collection ID."""
        col_b = Collection(owner_id=second_user.id, name="User B Collection")
        db_session.add(col_b)
        db_session.commit()

        rag = RAGService(dummy_embedder, vector_store, MockLLMProvider())
        with pytest.raises(ValueError, match="not found or not owned by user"):
            rag.answer_query(
                db=db_session,
                query="test",
                user_id=test_user.id,
                scope_type=ChatScopeType.COLLECTION,
                scope_id=col_b.id,
            )

    def test_nonexistent_targets_denied(self, db_session, test_user, dummy_embedder, vector_store):
        """Verifies non-existent UUIDs are safely rejected."""
        rag = RAGService(dummy_embedder, vector_store, MockLLMProvider())
        missing_id = uuid.uuid4()
        with pytest.raises(ValueError, match="not found or not owned by user"):
            rag.answer_query(
                db=db_session,
                query="test",
                user_id=test_user.id,
                scope_type=ChatScopeType.DOCUMENT,
                scope_id=missing_id,
            )
        with pytest.raises(ValueError, match="not found or not owned by user"):
            rag.answer_query(
                db=db_session,
                query="test",
                user_id=test_user.id,
                scope_type=ChatScopeType.COLLECTION,
                scope_id=missing_id,
            )

    def test_invalid_user_id_rejected_early(self, db_session, dummy_embedder, vector_store, monkeypatch):
        """Verifies invalid user IDs are rejected before embedding or LLM calls."""
        embed_called = False
        def mock_embed(*args, **kwargs):
            nonlocal embed_called
            embed_called = True
            return dummy_embedder.embed_text(*args, **kwargs)

        monkeypatch.setattr(dummy_embedder, "embed_text", mock_embed)
        rag = RAGService(dummy_embedder, vector_store, MockLLMProvider())

        with pytest.raises(ValueError, match="user_id cannot be None"):
            rag.answer_query(db=db_session, query="test", user_id=None)

        with pytest.raises(ValueError, match="user_id must be a valid UUID"):
            rag.answer_query(db=db_session, query="test", user_id="not-a-uuid")

        with pytest.raises(ValueError, match="user_id must be a valid UUID"):
            rag.answer_query(db=db_session, query="test", user_id=12345)

        assert not embed_called, "embed_text must not be called when user_id is invalid"

    def test_dangling_faiss_vector_not_sent_to_llm(self, db_session, test_user, dummy_embedder, vector_store, monkeypatch):
        """Verifies a chunk indexed in FAISS but deleted from DB is never sent to the LLM."""
        doc = Document(
            owner_id=test_user.id,
            filename="transient.pdf",
            file_path="transient.pdf",
            file_hash_sha256="ht",
            file_size_bytes=100,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.commit()

        c = Chunk(document_id=doc.id, chunk_index=0, page_number=1, content="Transient text", char_start=0, char_end=14, token_count=3)
        db_session.add(c)
        db_session.commit()

        v = dummy_embedder.embed_text(c.content)
        vector_store.add_vectors([c.id], [v])

        # Delete the chunk from the DB to simulate dangling FAISS index entry
        db_session.delete(c)
        db_session.commit()

        llm = MockLLMProvider()
        generate_called = False
        def mock_generate(*args, **kwargs):
            nonlocal generate_called
            generate_called = True
            return llm.generate(*args, **kwargs)

        monkeypatch.setattr(llm, "generate", mock_generate)
        rag = RAGService(dummy_embedder, vector_store, llm)

        resp = rag.answer_query(db=db_session, query="query", user_id=test_user.id)
        assert resp.status == "NO_RELEVANT_CONTEXT"
        assert resp.retrieved_chunk_count == 0
        assert not generate_called, "LLM must not be called with dangling FAISS chunks missing from DB"

    def test_unready_documents_excluded_from_global_retrieval(self, db_session, test_user, dummy_embedder, vector_store, monkeypatch):
        """Verifies chunks from PENDING or FAILED documents are excluded from GLOBAL retrieval."""
        doc_pending = Document(
            owner_id=test_user.id,
            filename="pending.pdf",
            file_path="pending.pdf",
            file_hash_sha256="hpending",
            file_size_bytes=100,
            processing_status=ProcessingStatus.PENDING.value,
        )
        doc_failed = Document(
            owner_id=test_user.id,
            filename="failed.pdf",
            file_path="failed.pdf",
            file_hash_sha256="hfailed",
            file_size_bytes=100,
            processing_status=ProcessingStatus.FAILED.value,
        )
        db_session.add_all([doc_pending, doc_failed])
        db_session.commit()

        c_p = Chunk(document_id=doc_pending.id, chunk_index=0, page_number=1, content="Pending text", char_start=0, char_end=12, token_count=3)
        c_f = Chunk(document_id=doc_failed.id, chunk_index=0, page_number=1, content="Failed text", char_start=0, char_end=11, token_count=3)
        db_session.add_all([c_p, c_f])
        db_session.commit()

        v_p = dummy_embedder.embed_text(c_p.content)
        v_f = dummy_embedder.embed_text(c_f.content)
        vector_store.add_vectors([c_p.id, c_f.id], [v_p, v_f])

        llm = MockLLMProvider()
        generate_called = False
        def mock_generate(*args, **kwargs):
            nonlocal generate_called
            generate_called = True
            return llm.generate(*args, **kwargs)

        monkeypatch.setattr(llm, "generate", mock_generate)
        rag = RAGService(dummy_embedder, vector_store, llm)

        resp = rag.answer_query(db=db_session, query="query", user_id=test_user.id)
        assert resp.status == "NO_RELEVANT_CONTEXT"
        assert resp.retrieved_chunk_count == 0
        assert not generate_called, "LLM must not be called when only unready documents match"


class TestRAGServiceDisambiguationAndGrounding:
    """Tests verifying end-to-end table disambiguation, citation validation, and bracket normalization."""

    def test_rag_query_disambiguates_table_and_validates_citations(
        self, db_session, test_user, dummy_embedder, vector_store, monkeypatch
    ):
        """Verifies target Table 6 chunk is retrieved and cited over Table 9 chunk."""
        doc = Document(
            owner_id=test_user.id,
            filename="benchmark.pdf",
            file_path="benchmark.pdf",
            file_hash_sha256="hbench",
            file_size_bytes=10000,
            processing_status=ProcessingStatus.READY.value,
            title="Benchmark Intrusion Detection Study",
        )
        db_session.add(doc)
        db_session.commit()

        c_target = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=7,
            content=(
                "Table 6. Performance metrics of all models on the CICIDS2017 dataset (without SMOTE).\n"
                "Naïve Bayes Accuracy 0.8565 Precision 0.6364 Recall 0.9742 FAR 0.1820 AUC 0.9711"
            ),
            char_start=0,
            char_end=175,
            token_count=40,
        )
        c_conflicting = Chunk(
            document_id=doc.id,
            chunk_index=1,
            page_number=17,
            content=(
                "Table 9. Statistical summary on NSL-KDD dataset (without SMOTE).\n"
                "Naïve Bayes Mean FAR 0.0065 Range [0.0065, 0.0065]"
            ),
            char_start=0,
            char_end=130,
            token_count=30,
        )
        db_session.add_all([c_target, c_conflicting])
        db_session.commit()

        v1 = dummy_embedder.embed_text(c_target.content)
        v2 = dummy_embedder.embed_text(c_conflicting.content)
        vector_store.add_vectors([c_target.id, c_conflicting.id], [v1, v2])

        # Configure LLM provider to answer based on provided context
        llm = MockLLMProvider()
        def mock_generate(req):
            from app.schemas.llm import LLMResponse
            first_chunk = req.context_chunks[0]
            ans = f"According to Table 6, the FAR of Naïve Bayes on CICIDS2017 without SMOTE is 0.1820 [CHUNK:{first_chunk.chunk_id}]."
            return LLMResponse(
                content=ans,
                model_name="test-model",
                provider_name="test-provider",
                finish_reason="stop",
                prompt_tokens=100,
                completion_tokens=30,
                total_tokens=130,
            )

        monkeypatch.setattr(llm, "generate", mock_generate)
        rag = RAGService(dummy_embedder, vector_store, llm)

        query = "According to Table 6, what is the FAR of Naïve Bayes on CICIDS2017 without SMOTE?"
        resp = rag.answer_query(db=db_session, query=query, user_id=test_user.id)

        assert resp.status == "SUCCESS"
        assert len(resp.citations) == 1
        assert resp.citations[0].chunk_id == c_target.id
        assert resp.citations[0].page_number == 7
        assert "Table 6" in resp.citations[0].quoted_text
        assert resp.unsupported_numbers == []

    def test_rag_query_handles_unicode_lenticular_brackets_from_llm(
        self, db_session, test_user, dummy_embedder, vector_store, monkeypatch
    ):
        """Verifies lenticular brackets emitted by LLMs (e.g. 【CHUNK:253】) are parsed into verified snapshots."""
        doc = Document(
            owner_id=test_user.id,
            filename="eval.pdf",
            file_path="eval.pdf",
            file_hash_sha256="heval",
            file_size_bytes=5000,
            processing_status=ProcessingStatus.READY.value,
            title="Evaluation Report",
        )
        db_session.add(doc)
        db_session.commit()

        c = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=7,
            content="Table 6. Performance of models on CICIDS2017 without SMOTE. Naïve Bayes FAR: 0.1820.",
            char_start=0,
            char_end=85,
            token_count=20,
        )
        db_session.add(c)
        db_session.commit()

        v = dummy_embedder.embed_text(c.content)
        vector_store.add_vectors([c.id], [v])

        llm = MockLLMProvider()
        def mock_generate(req):
            from app.schemas.llm import LLMResponse
            # Model emits Unicode lenticular brackets
            ans = f"Naïve Bayes FAR without SMOTE is 0.1820【CHUNK:{c.id}】."
            return LLMResponse(
                content=ans,
                model_name="test-model",
                provider_name="test-provider",
                finish_reason="stop",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
            )

        monkeypatch.setattr(llm, "generate", mock_generate)
        rag = RAGService(dummy_embedder, vector_store, llm)

        query = "According to Table 6, what is the FAR of Naïve Bayes on CICIDS2017 without SMOTE?"
        resp = rag.answer_query(db=db_session, query=query, user_id=test_user.id)

        assert resp.status == "SUCCESS"
        assert len(resp.citations) == 1
        assert resp.citations[0].chunk_id == c.id
        assert resp.citations[0].page_number == 7
        assert resp.citations[0].document_title == "Evaluation Report"
        assert resp.unsupported_numbers == []
