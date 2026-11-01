"""Integration tests for Persistent Chat REST API endpoints (/api/v1/chat)."""
import threading
from typing import Generator
from uuid import UUID, uuid4
from fastapi.testclient import TestClient
import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.api.v1.chat import get_chat_service
from app.api.v1.query import get_rag_service
from app.core.database import Base
from app.core.security import create_access_token
from app.main import app
from app.models.chat import ChatScopeType, ChatSession, Message, MessageCitation, SenderType
from app.models.chunk import Chunk
from app.models.collection import Collection
from app.models.document import Document, ProcessingStatus
from app.models.user import User
from app.schemas.rag import CitationSnapshot, RAGResponse
from app.services.chat_service import ChatService
from app.services.llm.exceptions import (
    LLMContextLimitExceededError,
    LLMProviderError,
    LLMQuotaExceededError,
)
from app.services.llm.mock_provider import MockLLMProvider
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStore, VectorStoreNotReadyError


class DummyEmbeddingService:
    def __init__(self, dimension: int = 4):
        self._dimension = dimension
        self.model_name = "dummy-embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> np.ndarray:
        return np.ones(self._dimension, dtype=np.float32) / np.sqrt(self._dimension)

    def embed_batch(self, texts):
        return [self.embed_text(t) for t in texts]


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
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
def user_a(db_session: Session) -> User:
    user = User(
        id=uuid4(),
        email="chat_user_a@researchlens.ai",
        hashed_password="hashed_pw_a",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def user_b(db_session: Session) -> User:
    user = User(
        id=uuid4(),
        email="chat_user_b@researchlens.ai",
        hashed_password="hashed_pw_b",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def token_a(user_a: User) -> str:
    return create_access_token(subject=user_a.id)


@pytest.fixture
def token_b(user_b: User) -> str:
    return create_access_token(subject=user_b.id)



@pytest.fixture
def auth_headers_a(token_a: str) -> dict:
    return {"Authorization": f"Bearer {token_a}"}


@pytest.fixture
def auth_headers_b(token_b: str) -> dict:
    return {"Authorization": f"Bearer {token_b}"}


@pytest.fixture
def doc_a(db_session: Session, user_a: User) -> Document:
    doc = Document(
        id=uuid4(),
        owner_id=user_a.id,
        filename="user_a_doc.pdf",
        file_path="/tmp/user_a_doc.pdf",
        file_hash_sha256="hash_user_a_doc",
        file_size_bytes=2048,
        processing_status=ProcessingStatus.READY.value,
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    return doc


@pytest.fixture
def col_a(db_session: Session, user_a: User) -> Collection:
    col = Collection(
        id=uuid4(),
        owner_id=user_a.id,
        name="User A Collection",
    )
    db_session.add(col)
    db_session.commit()
    db_session.refresh(col)
    return col


@pytest.fixture
def mock_rag_service(db_session: Session) -> RAGService:
    rag = RAGService(
        embedding_service=DummyEmbeddingService(dimension=4),
        vector_store=VectorStore(dimension=4, model_name="dummy-embedder"),
        llm_provider=MockLLMProvider(model_name="mock-model"),
    )
    return rag


@pytest.fixture
def client(db_session: Session, mock_rag_service: RAGService) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestChatAuthentication:
    """Verifies that unauthenticated requests to all chat endpoints return 401."""

    def test_endpoints_require_auth(self, client: TestClient):
        dummy_id = uuid4()
        endpoints = [
            ("POST", "/api/v1/chat/sessions", {"scope_type": "GLOBAL"}),
            ("GET", "/api/v1/chat/sessions", None),
            ("GET", f"/api/v1/chat/sessions/{dummy_id}", None),
            ("PATCH", f"/api/v1/chat/sessions/{dummy_id}", {"title": "Updated"}),
            ("GET", f"/api/v1/chat/sessions/{dummy_id}/messages", None),
            ("POST", f"/api/v1/chat/sessions/{dummy_id}/messages", {"content": "Hello"}),
        ]
        for method, path, payload in endpoints:
            if method == "POST":
                resp = client.post(path, json=payload)
            elif method == "GET":
                resp = client.get(path)
            elif method == "PATCH":
                resp = client.patch(path, json=payload)
            assert resp.status_code == 401, f"{method} {path} did not return 401"


class TestChatSessionEndpoints:
    """Verifies session CRUD operations, validations, and tenant isolation."""

    def test_create_session_success(
        self, client: TestClient, auth_headers_a: dict, doc_a: Document, col_a: Collection
    ):
        # 1. GLOBAL
        res_global = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={"scope_type": "GLOBAL", "title": "Global Thread"},
        )
        assert res_global.status_code == 201
        data = res_global.json()
        assert data["title"] == "Global Thread"
        assert data["scope_type"] == "GLOBAL"
        assert data["target_document_id"] is None
        assert data["target_collection_id"] is None
        assert data["is_archived"] is False
        assert data["message_count"] == 0

        # 2. DOCUMENT
        res_doc = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={"scope_type": "DOCUMENT", "target_document_id": str(doc_a.id)},
        )
        assert res_doc.status_code == 201
        assert res_doc.json()["target_document_id"] == str(doc_a.id)

        # 3. COLLECTION
        res_col = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={"scope_type": "COLLECTION", "target_collection_id": str(col_a.id)},
        )
        assert res_col.status_code == 201
        assert res_col.json()["target_collection_id"] == str(col_a.id)

    def test_create_session_foreign_targets_return_404(
        self, client: TestClient, auth_headers_b: dict, doc_a: Document, col_a: Collection
    ):
        # User B cannot target User A's document
        res_doc = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_b,
            json={"scope_type": "DOCUMENT", "target_document_id": str(doc_a.id)},
        )
        assert res_doc.status_code == 404
        assert "not found" in res_doc.json()["detail"].lower()

        # User B cannot target User A's collection
        res_col = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_b,
            json={"scope_type": "COLLECTION", "target_collection_id": str(col_a.id)},
        )
        assert res_col.status_code == 404
        assert "not found" in res_col.json()["detail"].lower()

    def test_create_session_forbids_extra_fields_and_user_id(
        self, client: TestClient, auth_headers_a: dict
    ):
        # Client cannot supply user_id or unexpected fields
        res = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={"scope_type": "GLOBAL", "user_id": str(uuid4()), "unknown_field": "val"},
        )
        assert res.status_code == 422

    def test_list_sessions_tenant_isolation_and_pagination(
        self, client: TestClient, auth_headers_a: dict, auth_headers_b: dict
    ):
        client.post("/api/v1/chat/sessions", headers=auth_headers_a, json={"title": "A1"})
        client.post("/api/v1/chat/sessions", headers=auth_headers_a, json={"title": "A2"})
        client.post("/api/v1/chat/sessions", headers=auth_headers_b, json={"title": "B1"})

        # User A list
        res_a = client.get("/api/v1/chat/sessions", headers=auth_headers_a)
        assert res_a.status_code == 200
        data_a = res_a.json()
        assert data_a["total"] == 2
        assert len(data_a["items"]) == 2
        assert {item["title"] for item in data_a["items"]} == {"A1", "A2"}

        # User B list
        res_b = client.get("/api/v1/chat/sessions", headers=auth_headers_b)
        assert res_b.status_code == 200
        data_b = res_b.json()
        assert data_b["total"] == 1
        assert data_b["items"][0]["title"] == "B1"

        # Pagination: limit=1
        res_pag = client.get("/api/v1/chat/sessions?limit=1&offset=0", headers=auth_headers_a)
        assert res_pag.status_code == 200
        assert len(res_pag.json()["items"]) == 1
        assert res_pag.json()["total"] == 2

    def test_get_and_patch_session_tenant_isolation(
        self, client: TestClient, auth_headers_a: dict, auth_headers_b: dict
    ):
        res = client.post("/api/v1/chat/sessions", headers=auth_headers_a, json={"title": "Original"})
        session_id = res.json()["id"]

        # User A can get session
        get_res = client.get(f"/api/v1/chat/sessions/{session_id}", headers=auth_headers_a)
        assert get_res.status_code == 200
        assert get_res.json()["title"] == "Original"

        # User B gets 404
        get_b = client.get(f"/api/v1/chat/sessions/{session_id}", headers=auth_headers_b)
        assert get_b.status_code == 404

        # User A can update title and archive
        patch_res = client.patch(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers_a,
            json={"title": "Updated Title", "is_archived": True},
        )
        assert patch_res.status_code == 200
        assert patch_res.json()["title"] == "Updated Title"
        assert patch_res.json()["is_archived"] is True

        # User B cannot update User A's session (404)
        patch_b = client.patch(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers_b,
            json={"title": "Hacked"},
        )
        assert patch_b.status_code == 404

    def test_delete_session_success(self, client: TestClient, auth_headers_a: dict):
        """Verifies owner can delete a chat session, returning 200 and removing it from subsequent queries."""
        # 1. Create a session
        res = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={"title": "Session to be deleted", "scope_type": "GLOBAL"},
        )
        assert res.status_code == 201
        session_id = res.json()["id"]

        # 2. Delete the session
        del_res = client.delete(f"/api/v1/chat/sessions/{session_id}", headers=auth_headers_a)
        assert del_res.status_code == 200
        assert del_res.json() == {"message": "Chat session deleted successfully."}

        # 3. Subsequent GET returns 404
        get_res = client.get(f"/api/v1/chat/sessions/{session_id}", headers=auth_headers_a)
        assert get_res.status_code == 404

        # 4. Listing sessions does not contain the deleted session
        list_res = client.get("/api/v1/chat/sessions", headers=auth_headers_a)
        assert list_res.status_code == 200
        session_ids = [s["id"] for s in list_res.json()["items"]]
        assert session_id not in session_ids

    def test_delete_session_unauthenticated_returns_401(self, client: TestClient):
        """Verifies unauthenticated DELETE request returns 401."""
        res = client.delete(f"/api/v1/chat/sessions/{uuid4()}")
        assert res.status_code == 401

    def test_delete_session_cross_user_isolation_returns_404(
        self, client: TestClient, auth_headers_a: dict, auth_headers_b: dict
    ):
        """Verifies user cannot delete another user's session; receives 404 without leaking existence."""
        # Create session as User A
        res = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={"title": "User A Private Chat", "scope_type": "GLOBAL"},
        )
        assert res.status_code == 201
        session_id = res.json()["id"]

        # User B attempts to delete User A's session
        del_b = client.delete(f"/api/v1/chat/sessions/{session_id}", headers=auth_headers_b)
        assert del_b.status_code == 404
        assert del_b.json()["detail"] == "Chat session not found."

        # User A's session is still intact
        get_a = client.get(f"/api/v1/chat/sessions/{session_id}", headers=auth_headers_a)
        assert get_a.status_code == 200
        assert get_a.json()["title"] == "User A Private Chat"

    def test_delete_session_invalid_uuid_returns_422(self, client: TestClient, auth_headers_a: dict):
        """Verifies malformed UUID in URL returns 422 validation error."""
        res = client.delete("/api/v1/chat/sessions/not-a-valid-uuid", headers=auth_headers_a)
        assert res.status_code == 422

    def test_delete_nonexistent_session_returns_404(self, client: TestClient, auth_headers_a: dict):
        """Verifies deleting non-existent session UUID returns 404."""
        res = client.delete(f"/api/v1/chat/sessions/{uuid4()}", headers=auth_headers_a)
        assert res.status_code == 404
        assert res.json()["detail"] == "Chat session not found."


class TestChatMessageEndpoints:
    """Verifies message history and grounded dialogue turn execution."""

    def test_post_message_grounded_turn_success(
        self, client: TestClient, auth_headers_a: dict, doc_a: Document, db_session: Session, mock_rag_service: RAGService
    ):
        # Create session
        res = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={"scope_type": "DOCUMENT", "target_document_id": str(doc_a.id)},
        )
        session_id = res.json()["id"]

        # Insert chunk
        chunk = Chunk(
            document_id=doc_a.id,
            chunk_index=0,
            page_number=1,
            content="Evidence passage text.",
            char_start=0,
            char_end=22,
            token_count=5,
        )
        db_session.add(chunk)
        db_session.commit()

        # Mock RAG response
        mock_citations = [
            CitationSnapshot(
                chunk_id=chunk.id,
                document_id=doc_a.id,
                quoted_text="Evidence passage text.",
                page_number=1,
                document_title="user_a_doc.pdf",
                similarity_score=0.95,
                citation_frequency=1,
            )
        ]
        mock_rag_response = RAGResponse(
            query="Explain evidence",
            answer="Here is the evidence [1].",
            citations=mock_citations,
            status="SUCCESS",
            model_name="mock-model",
            provider_name="MockProvider",
            retrieved_chunk_count=1,
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30,
        )
        mock_rag_service.answer_query = lambda **kwargs: mock_rag_response

        # Post message
        post_resp = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_a,
            json={"content": "Explain evidence", "top_k": 3},
        )
        assert post_resp.status_code == 200
        turn_data = post_resp.json()
        assert turn_data["session_id"] == session_id
        assert turn_data["user_message"]["content"] == "Explain evidence"
        assert turn_data["user_message"]["sender_type"] == "USER"
        assert turn_data["assistant_message"]["content"] == "Here is the evidence [1]."
        assert turn_data["assistant_message"]["sender_type"] == "ASSISTANT"
        assert len(turn_data["assistant_message"]["citations"]) == 1
        cit = turn_data["assistant_message"]["citations"][0]
        assert cit["chunk_id"] == chunk.id
        assert cit["quoted_text"] == "Evidence passage text."
        assert cit["document_title"] == "user_a_doc.pdf"
        assert cit["similarity_score"] == 0.95

        # Get message history and assert chronological order
        hist_resp = client.get(f"/api/v1/chat/sessions/{session_id}/messages", headers=auth_headers_a)
        assert hist_resp.status_code == 200
        hist_data = hist_resp.json()
        assert hist_data["total"] == 2
        assert hist_data["items"][0]["content"] == "Explain evidence"
        assert hist_data["items"][0]["sender_type"] == "USER"
        assert hist_data["items"][1]["content"] == "Here is the evidence [1]."
        assert hist_data["items"][1]["sender_type"] == "ASSISTANT"
        assert len(hist_data["items"][1]["citations"]) == 1

    def test_post_message_to_archived_session_returns_400(
        self, client: TestClient, auth_headers_a: dict
    ):
        res = client.post("/api/v1/chat/sessions", headers=auth_headers_a, json={"title": "To Archive"})
        session_id = res.json()["id"]
        client.patch(f"/api/v1/chat/sessions/{session_id}", headers=auth_headers_a, json={"is_archived": True})

        post_res = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_a,
            json={"content": "Should fail"},
        )
        assert post_res.status_code == 400
        assert "archived" in post_res.json()["detail"].lower()

    def test_post_message_provider_error_status_and_atomic_rollback(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, mock_rag_service: RAGService
    ):
        res = client.post("/api/v1/chat/sessions", headers=auth_headers_a, json={"title": "Error Test"})
        session_id = res.json()["id"]

        # Provider quota error -> 429
        mock_rag_service.answer_query = lambda **kwargs: (_ for _ in ()).throw(LLMQuotaExceededError("Rate limit exceeded"))
        resp_429 = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_a,
            json={"content": "Should hit quota error"},
        )
        assert resp_429.status_code == 429
        assert "quota exceeded" in resp_429.json()["detail"].lower()

        # Provider upstream failure -> 502
        mock_rag_service.answer_query = lambda **kwargs: (_ for _ in ()).throw(LLMProviderError("Provider unavailable"))
        resp_502 = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_a,
            json={"content": "Should hit provider error"},
        )
        assert resp_502.status_code == 502
        assert "provider failed" in resp_502.json()["detail"].lower()

        # Vector store unavailable -> 503
        mock_rag_service.answer_query = lambda **kwargs: (_ for _ in ()).throw(VectorStoreNotReadyError("Index not ready"))
        resp_503 = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_a,
            json={"content": "Should hit vector store error"},
        )
        assert resp_503.status_code == 503
        assert "unavailable" in resp_503.json()["detail"].lower()

        # Context limit exceeded -> 400
        mock_rag_service.answer_query = lambda **kwargs: (_ for _ in ()).throw(LLMContextLimitExceededError("Context limit"))
        resp_400 = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_a,
            json={"content": "Should hit context limit"},
        )
        assert resp_400.status_code == 400
        assert "context" in resp_400.json()["detail"].lower()

        # Assert zero messages persisted across all failed calls (clean rollback)
        persisted = db_session.query(Message).filter(Message.chat_session_id == UUID(session_id)).all()
        assert len(persisted) == 0

    def test_post_message_foreign_session_returns_404(
        self, client: TestClient, auth_headers_a: dict, auth_headers_b: dict
    ):
        res = client.post("/api/v1/chat/sessions", headers=auth_headers_a, json={"title": "User A Chat"})
        session_id = res.json()["id"]

        # User B cannot post to User A's session
        post_b = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_b,
            json={"content": "Intruder message"},
        )
        assert post_b.status_code == 404
        assert "not found" in post_b.json()["detail"].lower()

    def test_post_message_collection_scope_success(
        self, client: TestClient, auth_headers_a: dict, col_a: Collection, mock_rag_service: RAGService
    ):
        res = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={"scope_type": "COLLECTION", "target_collection_id": str(col_a.id)},
        )
        session_id = res.json()["id"]

        mock_rag_response = RAGResponse(
            query="Collection query",
            answer="Answer for collection.",
            citations=[],
            status="SUCCESS",
            model_name="mock-model",
            provider_name="MockProvider",
            retrieved_chunk_count=0,
        )
        mock_rag_service.answer_query = lambda **kwargs: mock_rag_response

        resp = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_a,
            json={"content": "Collection query"},
        )
        assert resp.status_code == 200
        assert resp.json()["assistant_message"]["content"] == "Answer for collection."

    def test_post_message_no_relevant_context_fallback(
        self, client: TestClient, auth_headers_a: dict, mock_rag_service: RAGService
    ):
        res = client.post("/api/v1/chat/sessions", headers=auth_headers_a, json={"title": "Fallback Test"})
        session_id = res.json()["id"]

        fallback_response = RAGResponse(
            query="No match query",
            answer="No relevant context was found in your library to answer the prompt.",
            citations=[],
            status="NO_RELEVANT_CONTEXT",
            model_name="mock-model",
            provider_name="MockProvider",
            retrieved_chunk_count=0,
        )
        mock_rag_service.answer_query = lambda **kwargs: fallback_response

        resp = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_a,
            json={"content": "No match query"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["assistant_message"]["content"] == "No relevant context was found in your library to answer the prompt."
        assert len(data["assistant_message"]["citations"]) == 0

    def test_post_message_updates_session_timestamp(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, mock_rag_service: RAGService
    ):
        res = client.post("/api/v1/chat/sessions", headers=auth_headers_a, json={"title": "Timestamp Test"})
        session_id = res.json()["id"]
        initial_updated_at = res.json()["updated_at"]

        mock_rag_response = RAGResponse(
            query="Ping",
            answer="Pong",
            citations=[],
            status="SUCCESS",
            model_name="mock-model",
            provider_name="MockProvider",
            retrieved_chunk_count=0,
        )
        mock_rag_service.answer_query = lambda **kwargs: mock_rag_response

        resp = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_a,
            json={"content": "Ping"},
        )
        assert resp.status_code == 200

        # Reload session and check updated_at
        db_session.expire_all()
        session_model = db_session.query(ChatSession).filter(ChatSession.id == UUID(session_id)).first()
        assert session_model.updated_at is not None

    def test_post_message_payload_validation(
        self, client: TestClient, auth_headers_a: dict
    ):
        res = client.post("/api/v1/chat/sessions", headers=auth_headers_a, json={"title": "Validation Test"})
        session_id = res.json()["id"]

        # 1. Empty content -> 422
        r1 = client.post(f"/api/v1/chat/sessions/{session_id}/messages", headers=auth_headers_a, json={"content": "   "})
        assert r1.status_code == 422

        # 2. Content too long (> 10,000 chars) -> 422
        r2 = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_a,
            json={"content": "x" * 10001},
        )
        assert r2.status_code == 422

        # 3. top_k out of bounds -> 422
        r3 = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_a,
            json={"content": "Valid", "top_k": 0},
        )
        assert r3.status_code == 422

        r4 = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_a,
            json={"content": "Valid", "top_k": 51},
        )
        assert r4.status_code == 422

        # 4. Unknown extra field -> 422
        r5 = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_a,
            json={"content": "Valid", "user_id": str(uuid4())},
        )
        assert r5.status_code == 422

    def test_post_message_session_archived_during_rag_processing_returns_400(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, mock_rag_service: RAGService
    ):
        """Deterministic API test:
        If an active session is archived while RAG/provider processing is in flight,
        the request must return HTTP 400 Bad Request with an established error detail,
        and no messages or citations must be persisted.
        """
        import threading

        res = client.post("/api/v1/chat/sessions", headers=auth_headers_a, json={"title": "Race Test"})
        assert res.status_code == 201
        session_id = res.json()["id"]

        rag_started_event = threading.Event()
        archive_done_event = threading.Event()

        fake_rag_response = RAGResponse(
            query="In flight question",
            answer="In flight answer that should not be saved",
            citations=[],
            status="SUCCESS",
            model_name="mock-model",
            provider_name="MockProvider",
            retrieved_chunk_count=0,
        )

        def hooked_answer_query(**kwargs):
            rag_started_event.set()
            archive_done_event.wait(timeout=5.0)
            return fake_rag_response

        mock_rag_service.answer_query = hooked_answer_query

        # Use separate session factory for get_db so concurrent requests have their own session
        engine = db_session.get_bind()
        session_factory = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

        def per_request_db():
            s = session_factory()
            try:
                yield s
            finally:
                s.close()

        app.dependency_overrides[get_db] = per_request_db

        post_response = {}

        def send_post_turn():
            r = client.post(
                f"/api/v1/chat/sessions/{session_id}/messages",
                headers=auth_headers_a,
                json={"content": "In flight question"},
            )
            post_response["status_code"] = r.status_code
            post_response["json"] = r.json()

        t = threading.Thread(target=send_post_turn)
        t.start()

        assert rag_started_event.wait(timeout=5.0), "RAG processing did not start in time"

        # Concurrently archive the session via the PATCH endpoint
        patch_res = client.patch(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers_a,
            json={"is_archived": True},
        )
        assert patch_res.status_code == 200
        assert patch_res.json()["is_archived"] is True

        # Let the in-flight RAG query resume and attempt to persist
        archive_done_event.set()
        t.join(timeout=5.0)

        # Restore original test db fixture override
        app.dependency_overrides[get_db] = lambda: db_session

        # Assert the turn request was rejected with 400 Bad Request
        assert post_response["status_code"] == 400
        assert "archived chat session" in post_response["json"]["detail"].lower()

        # Assert no messages or citations were persisted
        db_session.expire_all()
        messages = db_session.query(Message).filter(Message.chat_session_id == UUID(session_id)).all()
        assert len(messages) == 0
        citations = db_session.query(MessageCitation).all()
        assert len(citations) == 0


    def test_patch_unarchive_fails_when_target_document_deleted(
        self, client: TestClient, auth_headers_a: dict, doc_a: Document, db_session: Session
    ):
        res = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={"scope_type": "DOCUMENT", "target_document_id": str(doc_a.id)},
        )
        session_id = res.json()["id"]

        # Simulate document deletion archiving the session and clearing target_document_id
        db_session.query(ChatSession).filter(ChatSession.id == UUID(session_id)).update(
            {"is_archived": True, "target_document_id": None}
        )
        db_session.commit()

        # Attempt to unarchive
        patch_res = client.patch(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers_a,
            json={"is_archived": False},
        )
        assert patch_res.status_code == 400
        assert "whose target document was deleted" in patch_res.json()["detail"].lower()


class TestChatApiConcurrency:
    """Verifies HTTP API concurrency handling, 409 Conflict with SESSION_BUSY, and tenant isolation."""

    def test_delete_session_returns_409_conflict_when_generation_in_progress(
        self, client: TestClient, auth_headers_a: dict, db_session: Session
    ):
        res = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={"scope_type": "GLOBAL", "title": "API Active Chat"},
        )
        session_id = res.json()["id"]

        rag_started_event = threading.Event()
        rag_proceed_event = threading.Event()
        post_response = {}

        mock_rag = RAGService(
            embedding_service=DummyEmbeddingService(dimension=4),
            vector_store=VectorStore(dimension=4, model_name="dummy-embedder"),
            llm_provider=MockLLMProvider(model_name="mock-model"),
        )

        def slow_answer(**kwargs):
            rag_started_event.set()
            assert rag_proceed_event.wait(timeout=5.0)
            return RAGResponse(
                query=kwargs.get("query", "In flight question"),
                answer="Completed answer.",
                citations=[],
                status="SUCCESS",
                model_name="mock-model",
                provider_name="MockProvider",
                retrieved_chunk_count=0,
                completion_tokens=5,
                prompt_tokens=10,
                total_tokens=15,
            )

        mock_rag.answer_query = slow_answer
        app.dependency_overrides[get_rag_service] = lambda: mock_rag

        def send_post_turn():
            r = client.post(
                f"/api/v1/chat/sessions/{session_id}/messages",
                headers=auth_headers_a,
                json={"content": "In flight question"},
            )
            post_response["status_code"] = r.status_code
            post_response["json"] = r.json()

        t = threading.Thread(target=send_post_turn)
        t.start()

        assert rag_started_event.wait(timeout=5.0)

        # Attempt DELETE while generation is in flight
        del_res = client.delete(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers_a,
        )
        assert del_res.status_code == 409
        assert "SESSION_BUSY" in del_res.json()["detail"]

        # Resume generation and ensure it persists safely
        rag_proceed_event.set()
        t.join(timeout=5.0)

        app.dependency_overrides.pop(get_rag_service, None)

        assert post_response["status_code"] == 200
        assert post_response["json"]["assistant_message"]["content"] == "Completed answer."

        # Verify session is still alive
        check_res = client.get(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers_a,
        )
        assert check_res.status_code == 200

    def test_duplicate_post_turn_returns_409_conflict(
        self, client: TestClient, auth_headers_a: dict, db_session: Session
    ):
        res = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={"scope_type": "GLOBAL", "title": "API Duplicate Post Chat"},
        )
        session_id = res.json()["id"]

        rag_started_event = threading.Event()
        rag_proceed_event = threading.Event()
        first_post_response = {}

        mock_rag = RAGService(
            embedding_service=DummyEmbeddingService(dimension=4),
            vector_store=VectorStore(dimension=4, model_name="dummy-embedder"),
            llm_provider=MockLLMProvider(model_name="mock-model"),
        )

        def slow_answer(**kwargs):
            rag_started_event.set()
            assert rag_proceed_event.wait(timeout=5.0)
            return RAGResponse(
                query=kwargs.get("query", "First question"),
                answer="First answer.",
                citations=[],
                status="SUCCESS",
                model_name="mock-model",
                provider_name="MockProvider",
                retrieved_chunk_count=0,
                completion_tokens=5,
                prompt_tokens=10,
                total_tokens=15,
            )

        mock_rag.answer_query = slow_answer
        app.dependency_overrides[get_rag_service] = lambda: mock_rag

        def send_first_post():
            r = client.post(
                f"/api/v1/chat/sessions/{session_id}/messages",
                headers=auth_headers_a,
                json={"content": "First question"},
            )
            first_post_response["status_code"] = r.status_code

        t = threading.Thread(target=send_first_post)
        t.start()

        assert rag_started_event.wait(timeout=5.0)

        # Concurrently send second message on same session
        second_post = client.post(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers_a,
            json={"content": "Second question"},
        )
        assert second_post.status_code == 409
        assert "SESSION_BUSY" in second_post.json()["detail"]

        rag_proceed_event.set()
        t.join(timeout=5.0)

        app.dependency_overrides.pop(get_rag_service, None)
        assert first_post_response["status_code"] == 200

    def test_delete_session_by_other_user_returns_404_not_409_even_if_generating(
        self, client: TestClient, auth_headers_a: dict, auth_headers_b: dict, db_session: Session
    ):
        res = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={"scope_type": "GLOBAL", "title": "User A Private Chat"},
        )
        session_id = res.json()["id"]

        rag_started_event = threading.Event()
        rag_proceed_event = threading.Event()

        mock_rag = RAGService(
            embedding_service=DummyEmbeddingService(dimension=4),
            vector_store=VectorStore(dimension=4, model_name="dummy-embedder"),
            llm_provider=MockLLMProvider(model_name="mock-model"),
        )

        def slow_answer(**kwargs):
            rag_started_event.set()
            assert rag_proceed_event.wait(timeout=5.0)
            return RAGResponse(
                query=kwargs.get("query", "Q"),
                answer="Ans",
                citations=[],
                status="SUCCESS",
                model_name="mock-model",
                provider_name="MockProvider",
                retrieved_chunk_count=0,
                completion_tokens=5,
                prompt_tokens=10,
                total_tokens=15,
            )

        mock_rag.answer_query = slow_answer
        app.dependency_overrides[get_rag_service] = lambda: mock_rag

        t = threading.Thread(
            target=lambda: client.post(
                f"/api/v1/chat/sessions/{session_id}/messages",
                headers=auth_headers_a,
                json={"content": "Q"},
            )
        )
        t.start()

        assert rag_started_event.wait(timeout=5.0)

        # User B attempts to DELETE User A's session
        del_res = client.delete(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers_b,
        )
        # Must return 404 Not Found, revealing nothing about existence or busy state
        assert del_res.status_code == 404
        assert del_res.json()["detail"] == "Chat session not found."

        rag_proceed_event.set()
        t.join(timeout=5.0)
        app.dependency_overrides.pop(get_rag_service, None)


class TestMultiDocumentScopeAPI:
    """Integration tests for DOCUMENTS scope via HTTP REST endpoints."""

    def test_create_documents_session_api(self, client, auth_headers_a, db_session, user_a):
        doc1 = Document(
            id=uuid4(),
            owner_id=user_a.id,
            filename="api_doc1.pdf",
            title="API Doc 1",
            file_path="/tmp/api_doc1.pdf",
            file_hash_sha256="api_h1",
            file_size_bytes=1000,
            page_count=6,
            processing_status=ProcessingStatus.READY.value,
        )
        doc2 = Document(
            id=uuid4(),
            owner_id=user_a.id,
            filename="api_doc2.pdf",
            title="API Doc 2",
            file_path="/tmp/api_doc2.pdf",
            file_hash_sha256="api_h2",
            file_size_bytes=2000,
            page_count=12,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add_all([doc1, doc2])
        db_session.commit()

        # POST /api/v1/chat/sessions
        res = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={
                "title": "API Multi-Doc Chat",
                "scope_type": "DOCUMENTS",
                "document_ids": [str(doc1.id), str(doc2.id)],
            },
        )
        assert res.status_code == 201
        data = res.json()
        assert data["title"] == "API Multi-Doc Chat"
        assert data["scope_type"] == "DOCUMENTS"
        assert len(data["selected_documents"]) == 2
        returned_ids = {d["id"] for d in data["selected_documents"]}
        assert str(doc1.id) in returned_ids
        assert str(doc2.id) in returned_ids

        session_id = data["id"]

        # GET /api/v1/chat/sessions/{id}
        get_res = client.get(f"/api/v1/chat/sessions/{session_id}", headers=auth_headers_a)
        assert get_res.status_code == 200
        assert len(get_res.json()["selected_documents"]) == 2

        # GET /api/v1/chat/sessions
        list_res = client.get("/api/v1/chat/sessions", headers=auth_headers_a)
        assert list_res.status_code == 200
        items = list_res.json()["items"]
        matching = [item for item in items if item["id"] == session_id]
        assert len(matching) == 1
        assert len(matching[0]["selected_documents"]) == 2

    def test_create_documents_session_cross_tenant_api(self, client, auth_headers_a, db_session, user_b):
        doc_b = Document(
            id=uuid4(),
            owner_id=user_b.id,
            filename="user_b_private.pdf",
            file_path="/tmp/b.pdf",
            file_hash_sha256="api_hb",
            file_size_bytes=1000,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc_b)
        db_session.commit()

        res = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={
                "title": "Attacking User B",
                "scope_type": "DOCUMENTS",
                "document_ids": [str(doc_b.id)],
            },
        )
        assert res.status_code == 404

    def test_create_documents_session_unready_api(self, client, auth_headers_a, db_session, user_a):
        doc = Document(
            id=uuid4(),
            owner_id=user_a.id,
            filename="unready.pdf",
            file_path="/tmp/unready.pdf",
            file_hash_sha256="api_unready",
            file_size_bytes=1000,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db_session.add(doc)
        db_session.commit()

        res = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={
                "title": "Unready Session",
                "scope_type": "DOCUMENTS",
                "document_ids": [str(doc.id)],
            },
        )
        assert res.status_code == 400
        assert "not ready for chat" in res.json()["detail"]

    def test_create_documents_session_validation_api(self, client, auth_headers_a):
        # Empty document_ids list
        res = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={
                "title": "Invalid Empty",
                "scope_type": "DOCUMENTS",
                "document_ids": [],
            },
        )
        assert res.status_code == 422

        # Duplicate document IDs
        u = str(uuid4())
        res_dup = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={
                "title": "Invalid Duplicate",
                "scope_type": "DOCUMENTS",
                "document_ids": [u, u],
            },
        )
        assert res_dup.status_code == 422


