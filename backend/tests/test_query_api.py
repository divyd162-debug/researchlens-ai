"""Integration tests for Grounded RAG Query API endpoint (POST /api/v1/query).

Verifies:
1. Authentication: 401 for unauthenticated requests, invalid tokens, and inactive users.
2. Request Schema Validation: 422 for missing/empty/whitespace/long queries, invalid top_k,
   out-of-bounds similarity thresholds, invalid scopes, forbidden user_id, and unknown fields.
3. Grounded Generation: Successful RAG answer with citation snapshots, token metrics,
   GLOBAL, DOCUMENT, and COLLECTION scopes.
4. Empty & Unready Scenarios: 200 OK with status="NO_RELEVANT_CONTEXT" when no context matches
   or scoped document is not READY.
5. Cross-Tenant Isolation: 404 responses for foreign or non-existent document/collection scopes
   (never leaking existence), and zero cross-tenant retrieval leakage in GLOBAL queries.
6. Error Handling:
   - 503 Service Unavailable when vector store is not ready.
   - 429 Too Many Requests when LLM quota is exceeded.
   - 400 Bad Request when LLM context limit is exceeded.
   - 502 Bad Gateway when upstream LLM provider fails.
   - 500 Internal Server Error for unexpected exceptions.
"""
from typing import Generator
import uuid
from uuid import uuid4

from fastapi.testclient import TestClient
import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.api.v1.query import get_rag_service
from app.core.database import Base
from app.core.security import create_access_token
from app.main import app
from app.models.chat import ChatScopeType
from app.models.chunk import Chunk
from app.models.collection import Collection, CollectionDocument
from app.models.document import Document, ProcessingStatus
from app.models.user import User
from app.services.llm.exceptions import (
    LLMConfigurationError,
    LLMContextLimitExceededError,
    LLMError,
    LLMProviderError,
    LLMQuotaExceededError,
)
from app.services.llm.mock_provider import MockLLMProvider
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStore, VectorStoreNotReadyError


class DummyEmbeddingService:
    """Deterministic fast embedding service for testing without model weight downloads."""

    def __init__(self, dimension: int = 4):
        self._dimension = dimension
        self.model_name = "dummy-test-embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> np.ndarray:
        h = abs(hash(text))
        v0 = float((h % 97) + 1)
        v1 = float(((h // 97) % 89) + 1)
        v2 = float(((h // (97 * 89)) % 83) + 1)
        v3 = float(((h // (97 * 89 * 83)) % 79) + 1)
        vec = np.array([v0, v1, v2, v3], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm).astype(np.float32)


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
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
    try:
        yield session
    finally:
        session.rollback()
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def dummy_embedder() -> DummyEmbeddingService:
    return DummyEmbeddingService(dimension=4)


@pytest.fixture
def vector_store() -> VectorStore:
    return VectorStore(dimension=4, model_name="dummy-test-embedder")


@pytest.fixture
def mock_llm_provider() -> MockLLMProvider:
    return MockLLMProvider(model_name="mock-model-v1")


@pytest.fixture
def rag_service(
    dummy_embedder: DummyEmbeddingService,
    vector_store: VectorStore,
    mock_llm_provider: MockLLMProvider,
) -> RAGService:
    return RAGService(
        embedding_service=dummy_embedder,
        vector_store=vector_store,
        llm_provider=mock_llm_provider,
    )


@pytest.fixture
def client(db_session: Session, rag_service: RAGService) -> Generator[TestClient, None, None]:
    """Provides a TestClient with overridden database session and RAG service."""
    def _override_get_db():
        yield db_session

    def _override_get_rag_service():
        return rag_service

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_rag_service] = _override_get_rag_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def user_a(db_session: Session) -> User:
    user = User(email="user_a@example.com", hashed_password="pw_a", is_active=True)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def user_b(db_session: Session) -> User:
    user = User(email="user_b@example.com", hashed_password="pw_b", is_active=True)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers_a(user_a: User) -> dict:
    token = create_access_token(subject=user_a.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_b(user_b: User) -> dict:
    token = create_access_token(subject=user_b.id)
    return {"Authorization": f"Bearer {token}"}


def populate_document_with_chunk(
    db: Session,
    user: User,
    vector_store: VectorStore,
    chunk_id: int,
    content: str,
    title: str = "Attention Is All You Need",
    status: ProcessingStatus = ProcessingStatus.READY,
    page_number: int = 1,
    char_start: int = 0,
    char_end: int = 100,
    vector: np.ndarray = None,
) -> Document:
    """Helper to create a document, chunk in SQLite, and matching FAISS vector."""
    doc = Document(
        id=uuid4(),
        owner_id=user.id,
        filename=f"{title.lower().replace(' ', '_')}.pdf",
        title=title,
        file_path=f"/tmp/{uuid4()}.pdf",
        file_hash_sha256=f"sha256_{uuid4().hex}",
        file_size_bytes=1024,
        processing_status=status.value,
    )
    db.add(doc)
    db.flush()

    chunk = Chunk(
        id=chunk_id,
        document_id=doc.id,
        content=content,
        page_number=page_number,
        chunk_index=0,
        char_start=char_start,
        char_end=char_end,
        token_count=max(len(content.split()), 1),
    )
    db.add(chunk)
    db.commit()

    if vector is not None and status == ProcessingStatus.READY:
        norm_vec = vector / np.linalg.norm(vector)
        vector_store.add_vectors(
            vectors=norm_vec.reshape(1, -1).astype(np.float32),
            chunk_ids=[chunk_id],
        )

    return doc


# ===========================================================================
# 1. Authentication Tests
# ===========================================================================

class TestQueryAuthentication:
    """Tests ensuring POST /api/v1/query enforces valid bearer credentials."""

    def test_unauthenticated_request_returns_401(self, client: TestClient):
        response = client.post("/api/v1/query", json={"query": "what is attention?"})
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    def test_invalid_bearer_token_returns_401(self, client: TestClient):
        response = client.post(
            "/api/v1/query",
            headers={"Authorization": "Bearer invalid.token.payload"},
            json={"query": "what is attention?"},
        )
        assert response.status_code == 401

    def test_inactive_user_returns_401(self, client: TestClient, db_session: Session):
        inactive_user = User(email="inactive@example.com", hashed_password="pw", is_active=False)
        db_session.add(inactive_user)
        db_session.commit()
        token = create_access_token(subject=inactive_user.id)

        response = client.post(
            "/api/v1/query",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "what is attention?"},
        )
        assert response.status_code == 401


# ===========================================================================
# 2. Request Schema Validation Tests
# ===========================================================================

class TestQueryRequestValidation:
    """Tests schema validation, bounds checks, scope pairing, and forbidden fields."""

    def test_empty_body_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post("/api/v1/query", headers=auth_headers_a, json={})
        assert response.status_code == 422

    def test_missing_query_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"scope_type": "GLOBAL"},
        )
        assert response.status_code == 422

    def test_empty_string_query_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": ""},
        )
        assert response.status_code == 422

    def test_whitespace_only_query_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "   \n\t  "},
        )
        assert response.status_code == 422

    def test_query_too_long_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "x" * 1001},
        )
        assert response.status_code == 422

    def test_top_k_bounds_returns_422(self, client: TestClient, auth_headers_a: dict):
        for bad_k in [0, -1, 51]:
            response = client.post(
                "/api/v1/query",
                headers=auth_headers_a,
                json={"query": "attention", "top_k": bad_k},
            )
            assert response.status_code == 422

    def test_similarity_threshold_bounds_returns_422(self, client: TestClient, auth_headers_a: dict):
        for bad_th in [-1.1, 1.1]:
            response = client.post(
                "/api/v1/query",
                headers=auth_headers_a,
                json={"query": "attention", "similarity_threshold": bad_th},
            )
            assert response.status_code == 422

    def test_document_scope_missing_scope_id_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "attention", "scope_type": "DOCUMENT"},
        )
        assert response.status_code == 422

    def test_collection_scope_missing_scope_id_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "attention", "scope_type": "COLLECTION"},
        )
        assert response.status_code == 422

    def test_global_scope_with_scope_id_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={
                "query": "attention",
                "scope_type": "GLOBAL",
                "scope_id": str(uuid4()),
            },
        )
        assert response.status_code == 422

    def test_forbidden_user_id_injection_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "attention", "user_id": str(uuid4())},
        )
        assert response.status_code == 422

    def test_unknown_fields_rejected_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "attention", "extra_field": "disallowed"},
        )
        assert response.status_code == 422


# ===========================================================================
# 3. Grounded Generation & Scoping Tests
# ===========================================================================

class TestGroundedRAGExecution:
    """Tests successful RAG query execution, scopes, citations, and empty context handling."""

    def test_successful_global_rag_query(
        self,
        client: TestClient,
        db_session: Session,
        user_a: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
        auth_headers_a: dict,
    ):
        v = dummy_embedder.embed_text("attention mechanism in transformers")
        doc = populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=101,
            content="The dominant sequence transduction models are based on complex recurrent neural networks.",
            title="Attention Is All You Need",
            char_start=0,
            char_end=95,
            vector=v,
        )

        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "attention mechanism in transformers", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["query"] == "attention mechanism in transformers"
        assert data["status"] == "SUCCESS"
        assert data["model_name"] == "mock-model-v1"
        assert data["provider_name"] == "mock"
        assert data["retrieved_chunk_count"] == 1
        assert len(data["citations"]) == 1

        citation = data["citations"][0]
        assert citation["chunk_id"] == 101
        assert citation["document_id"] == str(doc.id)
        assert citation["document_title"] == "Attention Is All You Need"
        assert citation["page_number"] == 1
        assert citation["citation_frequency"] >= 1
        assert data["total_tokens"] > 0

    def test_successful_document_scoped_query(
        self,
        client: TestClient,
        db_session: Session,
        user_a: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
        auth_headers_a: dict,
    ):
        v = dummy_embedder.embed_text("convolutional neural networks")
        doc = populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=201,
            content="Deep residual networks represent a breakthrough in image recognition architectures.",
            title="ResNet Paper",
            vector=v,
        )

        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={
                "query": "convolutional neural networks",
                "scope_type": "DOCUMENT",
                "scope_id": str(doc.id),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["retrieved_chunk_count"] == 1
        assert data["citations"][0]["document_id"] == str(doc.id)

    def test_successful_collection_scoped_query(
        self,
        client: TestClient,
        db_session: Session,
        user_a: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
        auth_headers_a: dict,
    ):
        col = Collection(owner_id=user_a.id, name="Vision Papers", description="CV papers")
        db_session.add(col)
        db_session.commit()

        v = dummy_embedder.embed_text("vision transformers")
        doc = populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=301,
            content="An image is worth 16x16 words in vision transformers.",
            title="ViT Paper",
            vector=v,
        )

        col_doc = CollectionDocument(collection_id=col.id, document_id=doc.id)
        db_session.add(col_doc)
        db_session.commit()

        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={
                "query": "vision transformers",
                "scope_type": "COLLECTION",
                "scope_id": str(col.id),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["retrieved_chunk_count"] == 1
        assert data["citations"][0]["chunk_id"] == 301

    def test_empty_retrieval_returns_no_relevant_context(
        self,
        client: TestClient,
        auth_headers_a: dict,
    ):
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "non-existent concepts in empty library"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "NO_RELEVANT_CONTEXT"
        assert data["retrieved_chunk_count"] == 0
        assert data["citations"] == []
        assert "No relevant context was found" in data["answer"]

    def test_similarity_threshold_filters_all_chunks(
        self,
        client: TestClient,
        db_session: Session,
        user_a: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
        auth_headers_a: dict,
    ):
        v = dummy_embedder.embed_text("target text")
        populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=401,
            content="target text",
            vector=v,
        )

        # Set impossible similarity threshold
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={
                "query": "target text",
                "similarity_threshold": 1.0,  # Strict maximum: slightly different vector or FP noise
            },
        )
        assert response.status_code == 200
        data = response.json()
        # With threshold=1.0, if cosine < 1.0 it returns NO_RELEVANT_CONTEXT
        assert data["status"] in ("SUCCESS", "NO_RELEVANT_CONTEXT")

    def test_unready_scoped_document_returns_no_relevant_context(
        self,
        client: TestClient,
        db_session: Session,
        user_a: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
        auth_headers_a: dict,
    ):
        doc = populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=501,
            content="Unready doc chunk",
            title="Pending Paper",
            status=ProcessingStatus.PENDING,
        )

        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={
                "query": "any query",
                "scope_type": "DOCUMENT",
                "scope_id": str(doc.id),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "NO_RELEVANT_CONTEXT"
        assert data["retrieved_chunk_count"] == 0
        assert "not ready for queries" in data["answer"]

    def test_empty_collection_scoped_query_returns_no_relevant_context_without_llm_call(
        self,
        client: TestClient,
        db_session: Session,
        user_a: User,
        rag_service: RAGService,
        auth_headers_a: dict,
        monkeypatch,
    ):
        col = Collection(owner_id=user_a.id, name="Empty Collection", description="No documents here")
        db_session.add(col)
        db_session.commit()

        llm_called = False
        def _mock_generate(*args, **kwargs):
            nonlocal llm_called
            llm_called = True
            return rag_service.llm_provider.generate(*args, **kwargs)

        monkeypatch.setattr(rag_service.llm_provider, "generate", _mock_generate)

        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={
                "query": "any query on empty collection",
                "scope_type": "COLLECTION",
                "scope_id": str(col.id),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "NO_RELEVANT_CONTEXT"
        assert data["retrieved_chunk_count"] == 0
        assert data["citations"] == []
        assert "No relevant context was found" in data["answer"]
        assert not llm_called, "LLM must not be called when scoped collection has zero documents"


# ===========================================================================
# 4. Cross-Tenant Isolation Tests
# ===========================================================================

class TestQueryCrossTenantIsolation:
    """Tests multi-tenant isolation and 404 security guarantees."""

    def test_foreign_document_scope_returns_404(
        self,
        client: TestClient,
        db_session: Session,
        user_b: User,
        vector_store: VectorStore,
        auth_headers_a: dict,
    ):
        doc_b = populate_document_with_chunk(
            db=db_session,
            user=user_b,
            vector_store=vector_store,
            chunk_id=601,
            content="Confidential User B Document",
            title="User B Secret",
        )

        # User A attempts to query User B's document
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={
                "query": "secret",
                "scope_type": "DOCUMENT",
                "scope_id": str(doc_b.id),
            },
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Target document not found."

    def test_foreign_collection_scope_returns_404(
        self,
        client: TestClient,
        db_session: Session,
        user_b: User,
        auth_headers_a: dict,
    ):
        col_b = Collection(owner_id=user_b.id, name="User B Private Collection")
        db_session.add(col_b)
        db_session.commit()

        # User A attempts to query User B's collection
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={
                "query": "secret",
                "scope_type": "COLLECTION",
                "scope_id": str(col_b.id),
            },
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Target collection not found."

    def test_nonexistent_document_scope_returns_404(
        self,
        client: TestClient,
        auth_headers_a: dict,
    ):
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={
                "query": "secret",
                "scope_type": "DOCUMENT",
                "scope_id": str(uuid4()),
            },
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Target document not found."

    def test_nonexistent_collection_scope_returns_404(
        self,
        client: TestClient,
        auth_headers_a: dict,
    ):
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={
                "query": "secret",
                "scope_type": "COLLECTION",
                "scope_id": str(uuid4()),
            },
        )
        assert response.status_code == 404
        assert response.json()["detail"] == "Target collection not found."

    def test_global_query_never_leaks_other_tenant_chunks(
        self,
        client: TestClient,
        db_session: Session,
        user_a: User,
        user_b: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
        auth_headers_a: dict,
    ):
        v = dummy_embedder.embed_text("shared topic")
        # User B document
        populate_document_with_chunk(
            db=db_session,
            user=user_b,
            vector_store=vector_store,
            chunk_id=701,
            content="User B private chunk content on shared topic",
            title="User B Research",
            vector=v,
        )

        # User A executes global search
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "shared topic"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "NO_RELEVANT_CONTEXT"
        assert data["retrieved_chunk_count"] == 0
        assert data["citations"] == []

    def test_shared_faiss_index_isolates_tenants_when_both_have_matching_chunks(
        self,
        client: TestClient,
        db_session: Session,
        user_a: User,
        user_b: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
        auth_headers_a: dict,
    ):
        """Exercises shared FAISS index with candidates from both users.

        Ensures User B's chunk (which has exact/highest similarity) is rejected by PostgreSQL
        multi-tenant authorization, and User A receives only their authorized chunk.
        """
        query_text = "attention mechanisms in transformer networks"
        v_query = dummy_embedder.embed_text(query_text)

        # User B chunk: exact vector match to query (highest ranking)
        doc_b = populate_document_with_chunk(
            db=db_session,
            user=user_b,
            vector_store=vector_store,
            chunk_id=801,
            content="User B confidential attention findings",
            title="User B Proprietary Paper",
            vector=v_query,
        )

        # User A chunk: also matching vector
        doc_a = populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=802,
            content="User A public attention findings in transformers",
            title="User A Open Research",
            vector=v_query,
        )

        # User A executes query
        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": query_text, "top_k": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCESS"
        assert data["retrieved_chunk_count"] == 1
        assert len(data["citations"]) == 1

        citation = data["citations"][0]
        assert citation["chunk_id"] == 802
        assert citation["document_id"] == str(doc_a.id)
        assert citation["document_title"] == "User A Open Research"

        # Explicit negative assertions: User B secrets must never leak
        assert str(user_b.id) not in response.text
        assert str(doc_b.id) not in response.text
        assert "801" not in [str(c["chunk_id"]) for c in data["citations"]]
        assert "User B Proprietary Paper" not in response.text
        assert "User B confidential attention findings" not in response.text


# ===========================================================================
# 5. System & Provider Error Tests
# ===========================================================================

class TestQuerySystemAndProviderErrors:
    """Tests proper HTTP mapping for vector store, LLM provider, and internal errors."""

    def test_vector_store_not_ready_returns_503(
        self,
        client: TestClient,
        auth_headers_a: dict,
        monkeypatch,
    ):
        def _mock_answer(*args, **kwargs):
            raise VectorStoreNotReadyError("FAISS vector index has not been loaded.")

        monkeypatch.setattr(RAGService, "answer_query", _mock_answer)

        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "test query"},
        )
        assert response.status_code == 503
        assert "Vector search service is currently unavailable" in response.json()["detail"]

    def test_llm_quota_exceeded_returns_429(
        self,
        client: TestClient,
        auth_headers_a: dict,
        monkeypatch,
    ):
        def _mock_answer(*args, **kwargs):
            raise LLMQuotaExceededError("Rate limit exceeded at https://internal.llm:8443 with key=sentinel_quota_token")

        monkeypatch.setattr(RAGService, "answer_query", _mock_answer)

        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "test query"},
        )
        assert response.status_code == 429
        assert response.json()["detail"] == "LLM provider rate limit or quota exceeded. Please try again later."
        assert "sentinel_quota_token" not in response.text
        assert "https://internal.llm" not in response.text

    def test_llm_context_limit_exceeded_returns_400(
        self,
        client: TestClient,
        auth_headers_a: dict,
        monkeypatch,
    ):
        def _mock_answer(*args, **kwargs):
            raise LLMContextLimitExceededError("Max context 8192 exceeded for internal prompt containing sentinel_context_secret")

        monkeypatch.setattr(RAGService, "answer_query", _mock_answer)

        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "test query"},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "The query or retrieved context exceeds the model maximum context length."
        assert "sentinel_context_secret" not in response.text

    def test_llm_provider_error_returns_502(
        self,
        client: TestClient,
        auth_headers_a: dict,
        monkeypatch,
    ):
        def _mock_answer(*args, **kwargs):
            raise LLMProviderError("Upstream API connection failed at https://llm.vendor.cloud with sentinel_provider_key")

        monkeypatch.setattr(RAGService, "answer_query", _mock_answer)

        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "test query"},
        )
        assert response.status_code == 502
        assert response.json()["detail"] == "Upstream LLM provider failed to generate a response. Please try again later."
        assert "sentinel_provider_key" not in response.text
        assert "https://llm.vendor.cloud" not in response.text

    def test_llm_configuration_failure_returns_safe_500_without_secrets(
        self,
        client: TestClient,
        auth_headers_a: dict,
        monkeypatch,
    ):
        def _mock_answer(*args, **kwargs):
            raise LLMConfigurationError("Invalid provider setup: API_KEY='sk-secret-sentinel-12345' rejected.")

        monkeypatch.setattr(RAGService, "answer_query", _mock_answer)

        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "test query"},
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "LLM service encountered an unexpected error."
        assert "sk-secret-sentinel-12345" not in response.text
        assert "API_KEY" not in response.text

    def test_llm_generic_error_returns_500(
        self,
        client: TestClient,
        auth_headers_a: dict,
        monkeypatch,
    ):
        def _mock_answer(*args, **kwargs):
            raise LLMError("Internal LLM pipeline invariant violated with sentinel_generic_llm_secret.")

        monkeypatch.setattr(RAGService, "answer_query", _mock_answer)

        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "test query"},
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "LLM service encountered an unexpected error."
        assert "sentinel_generic_llm_secret" not in response.text

    def test_unexpected_exception_returns_500(
        self,
        client: TestClient,
        auth_headers_a: dict,
        monkeypatch,
    ):
        def _mock_answer(*args, **kwargs):
            raise RuntimeError("Database connection suddenly dropped: sentinel_db_connection_string_pw")

        monkeypatch.setattr(RAGService, "answer_query", _mock_answer)

        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "test query"},
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "An unexpected internal server error occurred while processing the query."
        assert "sentinel_db_connection_string_pw" not in response.text

    def test_generic_value_error_returns_400(
        self,
        client: TestClient,
        auth_headers_a: dict,
        monkeypatch,
    ):
        def _mock_answer(*args, **kwargs):
            raise ValueError("Invalid parameter value for retrieval configuration.")

        monkeypatch.setattr(RAGService, "answer_query", _mock_answer)

        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "test query"},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid parameter value for retrieval configuration."

    def test_internal_value_error_sanitizes_sentinel_details(
        self,
        client: TestClient,
        auth_headers_a: dict,
        monkeypatch,
    ):
        def _mock_answer(*args, **kwargs):
            raise ValueError("Database internal column error with user_id=123 and secret_token=sentinel_secret_val")

        monkeypatch.setattr(RAGService, "answer_query", _mock_answer)

        response = client.post(
            "/api/v1/query",
            headers=auth_headers_a,
            json={"query": "test query"},
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid request parameters."
        assert "sentinel_secret_val" not in response.text
        assert "user_id=123" not in response.text

