"""Integration tests for Semantic Search API endpoint (POST /api/v1/search).

Verifies:
1. Authentication: 401 for unauthenticated, invalid tokens, and inactive users.
2. Request Schema Validation: 422 for missing/empty/long queries, invalid top_k,
   out-of-bounds similarity thresholds, invalid scopes, and client-supplied user_id.
3. Ranked Retrieval: Successful passage retrieval with chunk metadata, exact character offsets,
   descending score ranking, top_k limits, and threshold filtering.
4. Empty Results: 200 OK with empty result list when no matching vectors/chunks found.
5. Scope Enforcement: GLOBAL, DOCUMENT, and COLLECTION scoping rules.
6. Cross-Tenant Isolation: Complete isolation preventing retrieval of other users' documents,
   and 404 responses for foreign document/collection scopes (never leaking existence).
7. System Errors: 400 for unready scoped documents, 503 when vector store is not ready.
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
from app.api.v1.search import get_retrieval_service
from app.core.database import Base
from app.core.security import create_access_token
from app.main import app
from app.models.chat import ChatScopeType
from app.models.chunk import Chunk
from app.models.collection import Collection, CollectionDocument
from app.models.document import Document, ProcessingStatus
from app.models.user import User
from app.services.retrieval_service import RetrievalService
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
def retrieval_service(dummy_embedder: DummyEmbeddingService, vector_store: VectorStore) -> RetrievalService:
    return RetrievalService(embedding_service=dummy_embedder, vector_store=vector_store)


@pytest.fixture
def client(db_session: Session, retrieval_service: RetrievalService) -> Generator[TestClient, None, None]:
    """Provides a TestClient with overridden database session and retrieval service."""
    def _override_get_db():
        yield db_session

    def _override_get_retrieval_service():
        return retrieval_service

    app.dependency_overrides[get_db] = _override_get_db
    app.dependency_overrides[get_retrieval_service] = _override_get_retrieval_service
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

class TestSearchAuthentication:
    """Tests ensuring POST /api/v1/search enforces valid bearer credentials."""

    def test_unauthenticated_request_returns_401(self, client: TestClient):
        response = client.post("/api/v1/search", json={"query": "neural networks"})
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"

    def test_invalid_bearer_token_returns_401(self, client: TestClient):
        response = client.post(
            "/api/v1/search",
            headers={"Authorization": "Bearer invalid.token.payload"},
            json={"query": "neural networks"},
        )
        assert response.status_code == 401

    def test_inactive_user_returns_401(self, client: TestClient, db_session: Session):
        inactive_user = User(email="inactive@example.com", hashed_password="pw", is_active=False)
        db_session.add(inactive_user)
        db_session.commit()
        token = create_access_token(subject=inactive_user.id)

        response = client.post(
            "/api/v1/search",
            headers={"Authorization": f"Bearer {token}"},
            json={"query": "neural networks"},
        )
        assert response.status_code == 401


# ===========================================================================
# 2. Request Schema Validation Tests
# ===========================================================================

class TestSearchRequestValidation:
    """Tests schema validation, bounds checks, and forbidden parameter injection."""

    def test_empty_body_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post("/api/v1/search", headers=auth_headers_a, json={})
        assert response.status_code == 422

    def test_missing_query_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"scope_type": "GLOBAL"},
        )
        assert response.status_code == 422

    def test_empty_query_string_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": ""},
        )
        assert response.status_code == 422

    def test_whitespace_only_query_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": "    \t\n  "},
        )
        assert response.status_code == 422

    def test_query_exceeding_1000_chars_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": "a" * 1001},
        )
        assert response.status_code == 422

    def test_valid_query_is_trimmed_in_response(
        self, client: TestClient, auth_headers_a: dict, dummy_embedder: DummyEmbeddingService
    ):
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": "   transformers in vision   "},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "transformers in vision"

    def test_top_k_bounds(self, client: TestClient, auth_headers_a: dict):
        # top_k < 1 -> 422
        resp_low = client.post("/api/v1/search", headers=auth_headers_a, json={"query": "test", "top_k": 0})
        assert resp_low.status_code == 422

        # top_k > 50 -> 422
        resp_high = client.post("/api/v1/search", headers=auth_headers_a, json={"query": "test", "top_k": 51})
        assert resp_high.status_code == 422

    def test_similarity_threshold_bounds(self, client: TestClient, auth_headers_a: dict):
        # threshold < -1.0 -> 422
        resp_low = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": "test", "similarity_threshold": -1.1},
        )
        assert resp_low.status_code == 422

        # threshold > 1.0 -> 422
        resp_high = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": "test", "similarity_threshold": 1.1},
        )
        assert resp_high.status_code == 422

    def test_invalid_scope_type_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": "test", "scope_type": "UNKNOWN_SCOPE"},
        )
        assert response.status_code == 422

    def test_global_scope_with_scope_id_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": "test", "scope_type": "GLOBAL", "scope_id": str(uuid4())},
        )
        assert response.status_code == 422

    def test_document_scope_without_scope_id_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": "test", "scope_type": "DOCUMENT"},
        )
        assert response.status_code == 422

    def test_collection_scope_without_scope_id_returns_422(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": "test", "scope_type": "COLLECTION"},
        )
        assert response.status_code == 422

    def test_client_supplied_user_id_rejected_with_422(self, client: TestClient, auth_headers_a: dict):
        """Never accept user_id from the client request body (ConfigDict extra='forbid')."""
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": "test", "user_id": str(uuid4())},
        )
        assert response.status_code == 422


# ===========================================================================
# 3. Retrieval & Ranking Tests
# ===========================================================================

class TestSearchRetrievalAndRanking:
    """Tests successful ranked passage retrieval and ordering."""

    def test_successful_search_returns_ranked_passages(
        self,
        client: TestClient,
        auth_headers_a: dict,
        db_session: Session,
        user_a: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
    ):
        # Query vector
        query_text = "attention mechanisms"
        q_vec = dummy_embedder.embed_text(query_text)

        # Create two matching chunks: chunk 101 very close, chunk 102 slightly less close
        vec_101 = q_vec.copy()
        vec_102 = np.array([q_vec[1], q_vec[0], q_vec[2], q_vec[3]], dtype=np.float32)

        doc1 = populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=101,
            content="Self-attention connects different positions of a single sequence.",
            title="Attention Is All You Need",
            page_number=2,
            char_start=45,
            char_end=112,
            vector=vec_101,
        )
        doc2 = populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=102,
            content="Recurrent neural networks compute hidden states sequentially.",
            title="Seq2Seq Models",
            page_number=1,
            char_start=0,
            char_end=62,
            vector=vec_102,
        )

        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": query_text, "top_k": 5, "similarity_threshold": 0.0},
        )
        assert response.status_code == 200
        data = response.json()

        assert data["query"] == query_text
        assert data["scope_type"] == "GLOBAL"
        assert data["scope_id"] is None
        assert data["total_results"] == 2
        assert data["result_count"] == 2
        assert len(data["results"]) == 2

        # Verify descending order of similarity score
        top_passage = data["results"][0]
        second_passage = data["results"][1]
        assert top_passage["similarity_score"] >= second_passage["similarity_score"]

        # Verify passage 101 exact metadata
        assert top_passage["chunk_id"] == 101
        assert top_passage["document_id"] == str(doc1.id)
        assert top_passage["document_title"] == "Attention Is All You Need"
        assert top_passage["page_number"] == 2
        assert top_passage["content"] == "Self-attention connects different positions of a single sequence."
        assert top_passage["char_start"] == 45
        assert top_passage["char_end"] == 112

    def test_search_returns_filename_when_document_title_is_none(
        self,
        client: TestClient,
        auth_headers_a: dict,
        db_session: Session,
        user_a: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
    ):
        """Verifies search returns the document filename when title is None."""
        query_text = "computer vision techniques"
        q_vec = dummy_embedder.embed_text(query_text)

        doc = Document(
            id=uuid4(),
            owner_id=user_a.id,
            filename="cv_transformer_paper.pdf",
            title=None,
            file_path="/tmp/cv.pdf",
            file_hash_sha256=f"sha256_{uuid4().hex}",
            file_size_bytes=1024,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.flush()

        chunk = Chunk(
            id=501,
            document_id=doc.id,
            content="Vision transformers apply self-attention to image patches.",
            page_number=1,
            chunk_index=0,
            char_start=0,
            char_end=58,
            token_count=8,
        )
        db_session.add(chunk)
        db_session.commit()

        vector_store.add_vectors(
            vectors=(q_vec / np.linalg.norm(q_vec)).reshape(1, -1).astype(np.float32),
            chunk_ids=[501],
        )

        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": query_text, "top_k": 1, "similarity_threshold": 0.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["document_title"] == "cv_transformer_paper.pdf"

    def test_top_k_limits_results(
        self,
        client: TestClient,
        auth_headers_a: dict,
        db_session: Session,
        user_a: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
    ):
        query_text = "deep learning"
        q_vec = dummy_embedder.embed_text(query_text)

        for i in range(1, 6):
            populate_document_with_chunk(
                db=db_session,
                user=user_a,
                vector_store=vector_store,
                chunk_id=200 + i,
                content=f"Deep learning chunk passage number {i}.",
                title=f"Paper {i}",
                vector=q_vec,
            )

        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": query_text, "top_k": 2},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 2
        assert data["result_count"] == 2
        assert len(data["results"]) == 2

    def test_similarity_threshold_filters_results(
        self,
        client: TestClient,
        auth_headers_a: dict,
        db_session: Session,
        user_a: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
    ):
        query_text = "sparse representations"
        q_vec = dummy_embedder.embed_text(query_text)

        # High similarity vector (identical to query, score ~ 1.0)
        populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=301,
            content="High match passage.",
            vector=q_vec,
        )

        # Low similarity / orthogonal vector
        low_vec = np.array([-q_vec[1], q_vec[0], -q_vec[3], q_vec[2]], dtype=np.float32)
        populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=302,
            content="Low match passage.",
            vector=low_vec,
        )

        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": query_text, "similarity_threshold": 0.95},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 1
        assert data["results"][0]["chunk_id"] == 301

    def test_empty_results_returns_200_with_empty_list(
        self,
        client: TestClient,
        auth_headers_a: dict,
    ):
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": "unindexed topic with no documents"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 0
        assert data["result_count"] == 0
        assert data["results"] == []


# ===========================================================================
# 4. Scope Enforcement Tests
# ===========================================================================

class TestSearchScopeEnforcement:
    """Tests scoping queries to DOCUMENT or COLLECTION."""

    def test_document_scoped_search_filters_other_documents(
        self,
        client: TestClient,
        auth_headers_a: dict,
        db_session: Session,
        user_a: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
    ):
        query_text = "evaluation methodology"
        q_vec = dummy_embedder.embed_text(query_text)

        doc1 = populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=401,
            content="Target document evaluation section.",
            title="Doc 1",
            vector=q_vec,
        )
        doc2 = populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=402,
            content="Other document evaluation section.",
            title="Doc 2",
            vector=q_vec,
        )

        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={
                "query": query_text,
                "scope_type": "DOCUMENT",
                "scope_id": str(doc1.id),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["scope_type"] == "DOCUMENT"
        assert data["scope_id"] == str(doc1.id)
        assert data["total_results"] == 1
        assert data["results"][0]["chunk_id"] == 401
        assert data["results"][0]["document_id"] == str(doc1.id)

    def test_document_scoped_search_non_existent_returns_404(
        self,
        client: TestClient,
        auth_headers_a: dict,
    ):
        random_id = str(uuid4())
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={
                "query": "any query",
                "scope_type": "DOCUMENT",
                "scope_id": random_id,
            },
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_collection_scoped_search_filters_documents(
        self,
        client: TestClient,
        auth_headers_a: dict,
        db_session: Session,
        user_a: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
    ):
        query_text = "collection topic"
        q_vec = dummy_embedder.embed_text(query_text)

        col = Collection(name="NLP Research", owner_id=user_a.id)
        db_session.add(col)
        db_session.commit()

        doc_in_col = populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=501,
            content="Document included in collection.",
            vector=q_vec,
        )
        doc_outside_col = populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=502,
            content="Document outside collection.",
            vector=q_vec,
        )

        # Associate doc_in_col with collection
        col_doc = CollectionDocument(collection_id=col.id, document_id=doc_in_col.id)
        db_session.add(col_doc)
        db_session.commit()

        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={
                "query": query_text,
                "scope_type": "COLLECTION",
                "scope_id": str(col.id),
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["scope_type"] == "COLLECTION"
        assert data["scope_id"] == str(col.id)
        assert data["total_results"] == 1
        assert data["results"][0]["chunk_id"] == 501

    def test_collection_scoped_search_non_existent_returns_404(
        self,
        client: TestClient,
        auth_headers_a: dict,
    ):
        random_id = str(uuid4())
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={
                "query": "any query",
                "scope_type": "COLLECTION",
                "scope_id": random_id,
            },
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


# ===========================================================================
# 5. Cross-Tenant Isolation Tests
# ===========================================================================

class TestSearchCrossTenantIsolation:
    """Verifies that tenants cannot access or probe other users' documents."""

    def test_global_search_never_returns_other_users_chunks(
        self,
        client: TestClient,
        auth_headers_a: dict,
        auth_headers_b: dict,
        db_session: Session,
        user_a: User,
        user_b: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
    ):
        query_text = "shared research theme"
        q_vec = dummy_embedder.embed_text(query_text)

        populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=601,
            content="User A private research notes.",
            vector=q_vec,
        )
        populate_document_with_chunk(
            db=db_session,
            user=user_b,
            vector_store=vector_store,
            chunk_id=602,
            content="User B private research notes.",
            vector=q_vec,
        )

        # User A searches: must receive chunk 601 and NOT chunk 602
        resp_a = client.post("/api/v1/search", headers=auth_headers_a, json={"query": query_text})
        assert resp_a.status_code == 200
        data_a = resp_a.json()
        assert data_a["total_results"] == 1
        assert data_a["results"][0]["chunk_id"] == 601

        # User B searches: must receive chunk 602 and NOT chunk 601
        resp_b = client.post("/api/v1/search", headers=auth_headers_b, json={"query": query_text})
        assert resp_b.status_code == 200
        data_b = resp_b.json()
        assert data_b["total_results"] == 1
        assert data_b["results"][0]["chunk_id"] == 602

    def test_document_scope_other_users_document_returns_404_without_leaking(
        self,
        client: TestClient,
        auth_headers_a: dict,
        db_session: Session,
        user_b: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
    ):
        q_vec = dummy_embedder.embed_text("test")
        doc_b = populate_document_with_chunk(
            db=db_session,
            user=user_b,
            vector_store=vector_store,
            chunk_id=701,
            content="User B confidential paper.",
            vector=q_vec,
        )

        # User A attempts to scope to User B's document
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={
                "query": "confidential",
                "scope_type": "DOCUMENT",
                "scope_id": str(doc_b.id),
            },
        )
        # Must return 404 Not Found (same as if doc_b did not exist)
        assert response.status_code == 404
        assert response.json()["detail"] == "Target document not found."

    def test_collection_scope_other_users_collection_returns_404_without_leaking(
        self,
        client: TestClient,
        auth_headers_a: dict,
        db_session: Session,
        user_b: User,
    ):
        col_b = Collection(name="User B Private Collection", owner_id=user_b.id)
        db_session.add(col_b)
        db_session.commit()

        # User A attempts to scope to User B's collection
        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={
                "query": "anything",
                "scope_type": "COLLECTION",
                "scope_id": str(col_b.id),
            },
        )
        # Must return 404 Not Found (same as if col_b did not exist)
        assert response.status_code == 404
        assert response.json()["detail"] == "Target collection not found."


# ===========================================================================
# 6. System Errors & Readiness Handling Tests
# ===========================================================================

class TestSearchSystemErrors:
    """Verifies handling of unready documents and vector store unavailability."""

    def test_document_scope_unready_document_returns_400(
        self,
        client: TestClient,
        auth_headers_a: dict,
        db_session: Session,
        user_a: User,
        vector_store: VectorStore,
    ):
        doc = populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=801,
            content="Unprocessed document text.",
            status=ProcessingStatus.EXTRACTING,
        )

        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={
                "query": "unready",
                "scope_type": "DOCUMENT",
                "scope_id": str(doc.id),
            },
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "not ready for queries" in detail
        assert "EXTRACTING" in detail

    def test_global_search_silently_ignores_unready_documents(
        self,
        client: TestClient,
        auth_headers_a: dict,
        db_session: Session,
        user_a: User,
        vector_store: VectorStore,
        dummy_embedder: DummyEmbeddingService,
    ):
        query_text = "quantum computing"
        q_vec = dummy_embedder.embed_text(query_text)

        # Ready document
        populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=901,
            content="Ready quantum paper.",
            status=ProcessingStatus.READY,
            vector=q_vec,
        )

        # Unready document (chunk in DB, but document is CHUNKING)
        unready_doc = populate_document_with_chunk(
            db=db_session,
            user=user_a,
            vector_store=vector_store,
            chunk_id=902,
            content="Unready quantum paper.",
            status=ProcessingStatus.CHUNKING,
        )
        # Force chunk 902 into FAISS index
        vector_store.add_vectors(
            chunk_ids=[902],
            vectors=q_vec.reshape(1, -1).astype(np.float32),
        )

        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": query_text},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] == 1
        assert data["results"][0]["chunk_id"] == 901

    def test_vector_store_not_ready_returns_503(
        self,
        client: TestClient,
        auth_headers_a: dict,
        vector_store: VectorStore,
    ):
        # Simulate broken / recovering vector store
        vector_store._is_ready = False

        response = client.post(
            "/api/v1/search",
            headers=auth_headers_a,
            json={"query": "test query"},
        )
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"].lower()
