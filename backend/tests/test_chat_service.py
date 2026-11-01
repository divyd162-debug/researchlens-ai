"""Unit tests for ChatService business logic, scope validation, tenant isolation, and turn persistence."""
import threading
from typing import Generator
from uuid import UUID, uuid4
import numpy as np
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.chat import ChatScopeType, ChatSession, Message, MessageCitation, SenderType
from app.models.chunk import Chunk
from app.models.collection import Collection, CollectionDocument
from app.models.document import Document, ProcessingStatus
from app.models.user import User
from app.schemas.chat import (
    ChatMessageCreateRequest,
    ChatSessionCreateRequest,
    ChatSessionUpdateRequest,
)
from app.schemas.rag import CitationSnapshot, RAGResponse
from app.services.chat_concurrency import ChatConcurrencyManager, ChatSessionBusyError
from app.services.chat_service import (
    ChatService,
    ChatSessionArchivedError,
    ChatSessionNotFoundError,
    ChatSessionValidationError,
    TargetEntityNotFoundError,
    TargetEntityNotReadyError,
)
from app.services.llm.exceptions import LLMProviderError, LLMQuotaExceededError
from app.services.llm.mock_provider import MockLLMProvider
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStore


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
def chat_service() -> ChatService:
    return ChatService()


@pytest.fixture
def user_a(db_session: Session) -> User:
    user = User(
        id=uuid4(),
        email="user_a@researchlens.ai",
        hashed_password="pw_hash_user_a",
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
        email="user_b@researchlens.ai",
        hashed_password="pw_hash_user_b",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def doc_a(db_session: Session, user_a: User) -> Document:
    doc = Document(
        id=uuid4(),
        owner_id=user_a.id,
        filename="user_a_doc.pdf",
        file_path="/tmp/doc_a.pdf",
        file_hash_sha256="hash_doc_a",
        file_size_bytes=1024,
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


class TestChatSessionCreation:
    """Verifies session creation across scopes, target entity validation, and tenant isolation."""

    def test_create_global_session_default_title(self, chat_service: ChatService, db_session: Session, user_a: User):
        req = ChatSessionCreateRequest(scope_type=ChatScopeType.GLOBAL)
        res = chat_service.create_session(db_session, user_a.id, req)
        assert res.id is not None
        assert res.title == "New Conversation"
        assert res.scope_type == ChatScopeType.GLOBAL
        assert res.target_document_id is None
        assert res.target_collection_id is None
        assert res.is_archived is False
        assert res.message_count == 0

    def test_create_global_session_custom_title(self, chat_service: ChatService, db_session: Session, user_a: User):
        req = ChatSessionCreateRequest(scope_type=ChatScopeType.GLOBAL, title=" Quantum Physics Study ")
        res = chat_service.create_session(db_session, user_a.id, req)
        assert res.title == "Quantum Physics Study"

    def test_create_document_session_success(
        self, chat_service: ChatService, db_session: Session, user_a: User, doc_a: Document
    ):
        req = ChatSessionCreateRequest(
            scope_type=ChatScopeType.DOCUMENT,
            target_document_id=doc_a.id,
            title="Doc Discussion",
        )
        res = chat_service.create_session(db_session, user_a.id, req)
        assert res.scope_type == ChatScopeType.DOCUMENT
        assert res.target_document_id == doc_a.id
        assert res.target_collection_id is None

    def test_create_document_session_unready_document_rejected(
        self, chat_service: ChatService, db_session: Session, user_a: User
    ):
        doc = Document(
            id=uuid4(),
            owner_id=user_a.id,
            filename="unready.pdf",
            file_path="/tmp/unready.pdf",
            file_hash_sha256="hash_unready",
            file_size_bytes=1024,
            processing_status=ProcessingStatus.PROCESSING_STATUS_CHOICES[0]
            if hasattr(ProcessingStatus, "PROCESSING_STATUS_CHOICES")
            else ProcessingStatus.PENDING.value,
        )
        db_session.add(doc)
        db_session.commit()

        req = ChatSessionCreateRequest(
            scope_type=ChatScopeType.DOCUMENT,
            target_document_id=doc.id,
        )
        with pytest.raises(TargetEntityNotReadyError) as exc_info:
            chat_service.create_session(db_session, user_a.id, req)
        assert "not ready for chat" in str(exc_info.value)

    def test_create_document_session_foreign_document_rejected(
        self, chat_service: ChatService, db_session: Session, user_b: User, doc_a: Document
    ):
        # User B attempts to scope conversation to User A's document
        req = ChatSessionCreateRequest(
            scope_type=ChatScopeType.DOCUMENT,
            target_document_id=doc_a.id,
        )
        with pytest.raises(TargetEntityNotFoundError):
            chat_service.create_session(db_session, user_b.id, req)

    def test_create_collection_session_success(
        self, chat_service: ChatService, db_session: Session, user_a: User, col_a: Collection
    ):
        req = ChatSessionCreateRequest(
            scope_type=ChatScopeType.COLLECTION,
            target_collection_id=col_a.id,
            title="Collection Research",
        )
        res = chat_service.create_session(db_session, user_a.id, req)
        assert res.scope_type == ChatScopeType.COLLECTION
        assert res.target_collection_id == col_a.id
        assert res.target_document_id is None

    def test_create_collection_session_foreign_collection_rejected(
        self, chat_service: ChatService, db_session: Session, user_b: User, col_a: Collection
    ):
        req = ChatSessionCreateRequest(
            scope_type=ChatScopeType.COLLECTION,
            target_collection_id=col_a.id,
        )
        with pytest.raises(TargetEntityNotFoundError):
            chat_service.create_session(db_session, user_b.id, req)


class TestChatSessionListingAndRetrieval:
    """Verifies listing, pagination, message count aggregation, and single session retrieval."""

    def test_list_sessions_tenant_isolation(
        self, chat_service: ChatService, db_session: Session, user_a: User, user_b: User
    ):
        chat_service.create_session(db_session, user_a.id, ChatSessionCreateRequest(title="A1"))
        chat_service.create_session(db_session, user_a.id, ChatSessionCreateRequest(title="A2"))
        chat_service.create_session(db_session, user_b.id, ChatSessionCreateRequest(title="B1"))

        list_a = chat_service.list_sessions(db_session, user_a.id)
        assert list_a.total == 2
        assert {item.title for item in list_a.items} == {"A1", "A2"}

        list_b = chat_service.list_sessions(db_session, user_b.id)
        assert list_b.total == 1
        assert list_b.items[0].title == "B1"

    def test_list_sessions_pagination_and_archive_filter(
        self, chat_service: ChatService, db_session: Session, user_a: User
    ):
        s1 = chat_service.create_session(db_session, user_a.id, ChatSessionCreateRequest(title="S1"))
        s2 = chat_service.create_session(db_session, user_a.id, ChatSessionCreateRequest(title="S2"))
        chat_service.update_session(db_session, s1.id, user_a.id, ChatSessionUpdateRequest(is_archived=True))

        all_active = chat_service.list_sessions(db_session, user_a.id, is_archived=False)
        assert all_active.total == 1
        assert all_active.items[0].id == s2.id

        all_archived = chat_service.list_sessions(db_session, user_a.id, is_archived=True)
        assert all_archived.total == 1
        assert all_archived.items[0].id == s1.id

    def test_get_session_details_and_message_count(
        self, chat_service: ChatService, db_session: Session, user_a: User, user_b: User
    ):
        session_resp = chat_service.create_session(db_session, user_a.id, ChatSessionCreateRequest(title="Detailed Session"))

        # Add two messages
        m1 = Message(chat_session_id=session_resp.id, sender_type=SenderType.USER.value, content="Hello")
        m2 = Message(chat_session_id=session_resp.id, sender_type=SenderType.ASSISTANT.value, content="World")
        db_session.add_all([m1, m2])
        db_session.commit()

        details = chat_service.get_session(db_session, session_resp.id, user_a.id)
        assert details.id == session_resp.id
        assert details.message_count == 2

        # User B cannot read User A's session
        with pytest.raises(ChatSessionNotFoundError):
            chat_service.get_session(db_session, session_resp.id, user_b.id)


class TestChatSessionUpdate:
    """Verifies title updates, archiving, and safe unarchiving."""

    def test_update_session_title(self, chat_service: ChatService, db_session: Session, user_a: User):
        s = chat_service.create_session(db_session, user_a.id, ChatSessionCreateRequest(title="Initial"))
        updated = chat_service.update_session(db_session, s.id, user_a.id, ChatSessionUpdateRequest(title="New Title"))
        assert updated.title == "New Title"

    def test_unarchive_scoped_session_with_missing_target_rejected(
        self, chat_service: ChatService, db_session: Session, user_a: User, doc_a: Document
    ):
        s = chat_service.create_session(
            db_session, user_a.id, ChatSessionCreateRequest(scope_type=ChatScopeType.DOCUMENT, target_document_id=doc_a.id)
        )
        # Simulate target document deletion having archived the session and cleared target_document_id
        db_session.query(ChatSession).filter(ChatSession.id == s.id).update(
            {"is_archived": True, "target_document_id": None}
        )
        db_session.commit()

        # Attempt to unarchive
        with pytest.raises(ChatSessionValidationError) as exc_info:
            chat_service.update_session(db_session, s.id, user_a.id, ChatSessionUpdateRequest(is_archived=False))
        assert "whose target document was deleted" in str(exc_info.value)


class TestMessageHistoryAndTurnExecution:
    """Verifies chronological message history, grounded turns, citation snapshots, and rollback."""

    def test_get_session_messages_chronological(self, chat_service: ChatService, db_session: Session, user_a: User):
        from datetime import datetime, timedelta, timezone

        s = chat_service.create_session(db_session, user_a.id, ChatSessionCreateRequest())
        now = datetime.now(timezone.utc)
        m1 = Message(chat_session_id=s.id, sender_type=SenderType.USER.value, content="First", created_at=now)
        m2 = Message(
            chat_session_id=s.id,
            sender_type=SenderType.ASSISTANT.value,
            content="Second",
            created_at=now + timedelta(seconds=1),
        )
        db_session.add_all([m1, m2])
        db_session.commit()

        history = chat_service.get_session_messages(db_session, s.id, user_a.id)
        assert history.total == 2
        assert history.items[0].content == "First"
        assert history.items[0].sender_type == SenderType.USER
        assert history.items[1].content == "Second"
        assert history.items[1].sender_type == SenderType.ASSISTANT

    def test_post_message_and_respond_atomic_persistence(
        self, chat_service: ChatService, db_session: Session, user_a: User, doc_a: Document
    ):
        s = chat_service.create_session(
            db_session, user_a.id, ChatSessionCreateRequest(scope_type=ChatScopeType.DOCUMENT, target_document_id=doc_a.id)
        )

        chunk = Chunk(
            document_id=doc_a.id,
            chunk_index=0,
            page_number=1,
            content="Passage text about research.",
            char_start=0,
            char_end=28,
            token_count=7,
        )
        db_session.add(chunk)
        db_session.commit()

        # Mock RAGService
        mock_rag = RAGService(
            embedding_service=DummyEmbeddingService(dimension=4),
            vector_store=VectorStore(dimension=4, model_name="dummy-embedder"),
            llm_provider=MockLLMProvider(model_name="mock-model"),
        )
        fake_citations = [
            CitationSnapshot(
                chunk_id=chunk.id,
                document_id=doc_a.id,
                quoted_text="Passage text about research.",
                page_number=1,
                document_title="user_a_doc.pdf",
                similarity_score=0.92,
                citation_frequency=1,
            )
        ]
        fake_rag_response = RAGResponse(
            query="Tell me about research.",
            answer="Research is systematic investigation. [1]",
            citations=fake_citations,
            status="SUCCESS",
            model_name="mock-model",
            provider_name="MockProvider",
            retrieved_chunk_count=1,
            completion_tokens=15,
            prompt_tokens=30,
            total_tokens=45,
        )
        mock_rag.answer_query = lambda **kwargs: fake_rag_response

        req = ChatMessageCreateRequest(content="Tell me about research.")
        turn = chat_service.post_message_and_respond(db_session, s.id, user_a.id, req, mock_rag)

        assert turn.session_id == s.id
        assert turn.user_message.content == "Tell me about research."
        assert turn.assistant_message.content == "Research is systematic investigation. [1]"
        assert len(turn.assistant_message.citations) == 1
        cit = turn.assistant_message.citations[0]
        assert cit.chunk_id == chunk.id
        assert cit.document_title == "user_a_doc.pdf"
        assert cit.page_number == 1
        assert cit.similarity_score == 0.92

        # Verify persisted in database
        persisted_messages = db_session.query(Message).filter(Message.chat_session_id == s.id).all()
        assert len(persisted_messages) == 2
        persisted_cits = db_session.query(MessageCitation).all()
        assert len(persisted_cits) == 1

    def test_post_message_to_archived_session_rejected(
        self, chat_service: ChatService, db_session: Session, user_a: User
    ):
        s = chat_service.create_session(db_session, user_a.id, ChatSessionCreateRequest())
        chat_service.update_session(db_session, s.id, user_a.id, ChatSessionUpdateRequest(is_archived=True))

        mock_rag = RAGService(
            embedding_service=DummyEmbeddingService(dimension=4),
            vector_store=VectorStore(dimension=4, model_name="dummy-embedder"),
            llm_provider=MockLLMProvider(model_name="mock-model"),
        )
        req = ChatMessageCreateRequest(content="Should fail on archived session")
        with pytest.raises(ChatSessionArchivedError):
            chat_service.post_message_and_respond(db_session, s.id, user_a.id, req, mock_rag)

    def test_post_message_provider_error_rolls_back_cleanly(
        self, chat_service: ChatService, db_session: Session, user_a: User
    ):
        s = chat_service.create_session(db_session, user_a.id, ChatSessionCreateRequest())

        mock_rag = RAGService(
            embedding_service=DummyEmbeddingService(dimension=4),
            vector_store=VectorStore(dimension=4, model_name="dummy-embedder"),
            llm_provider=MockLLMProvider(model_name="mock-model"),
        )

        def raise_llm_error(**kwargs):
            raise LLMProviderError("Upstream timeout")

        mock_rag.answer_query = raise_llm_error

        req = ChatMessageCreateRequest(content="Should not be persisted on LLM failure")
        with pytest.raises(LLMProviderError):
            chat_service.post_message_and_respond(db_session, s.id, user_a.id, req, mock_rag)

        # Verify NO user or assistant messages were persisted
        messages = db_session.query(Message).filter(Message.chat_session_id == s.id).all()
        assert len(messages) == 0

    def test_post_message_session_archived_during_rag_processing_rejected(
        self, chat_service: ChatService, db_session: Session, user_a: User
    ):
        """Deterministic race condition test:
        Verifies that if an active session is archived while RAG/provider processing is in flight,
        the post-processing recheck catches it, raises ChatSessionArchivedError, and persists NO messages or citations.
        """
        import threading

        s = chat_service.create_session(db_session, user_a.id, ChatSessionCreateRequest())

        rag_started_event = threading.Event()
        archive_completed_event = threading.Event()

        fake_rag_response = RAGResponse(
            query="Tell me about concurrent turns.",
            answer="This answer should never be saved.",
            citations=[],
            status="SUCCESS",
            model_name="mock-model",
            provider_name="mock-provider",
            retrieved_chunk_count=0,
            completion_tokens=10,
            prompt_tokens=20,
            total_tokens=30,
        )

        mock_rag = RAGService(
            embedding_service=DummyEmbeddingService(dimension=4),
            vector_store=VectorStore(dimension=4, model_name="dummy-embedder"),
            llm_provider=MockLLMProvider(model_name="mock-model"),
        )

        def hooked_answer_query(**kwargs):
            rag_started_event.set()
            archive_completed_event.wait(timeout=5.0)
            return fake_rag_response

        mock_rag.answer_query = hooked_answer_query

        thread_errors = []

        def worker_post():
            session_factory = sessionmaker(
                bind=db_session.get_bind(),
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
            )
            thread_db = session_factory()
            try:
                req = ChatMessageCreateRequest(content="Tell me about concurrent turns.")
                chat_service.post_message_and_respond(thread_db, s.id, user_a.id, req, mock_rag)
            except Exception as e:
                thread_errors.append(e)
            finally:
                thread_db.close()

        t = threading.Thread(target=worker_post)
        t.start()

        # Wait deterministically for worker to enter RAG generation
        assert rag_started_event.wait(timeout=5.0), "Worker did not begin RAG processing in time"

        # Concurrently archive the session in the database
        chat_service.update_session(
            db_session,
            s.id,
            user_a.id,
            ChatSessionUpdateRequest(is_archived=True),
        )

        # Allow worker thread's RAG call to return
        archive_completed_event.set()
        t.join(timeout=5.0)

        # 1. Verify worker caught the archive race and raised ChatSessionArchivedError
        assert len(thread_errors) == 1
        assert isinstance(thread_errors[0], ChatSessionArchivedError)
        assert "archived chat session" in str(thread_errors[0]).lower()

        # 2. Verify NO messages were persisted
        messages = db_session.query(Message).filter(Message.chat_session_id == s.id).all()
        assert len(messages) == 0

        # 3. Verify NO citations were persisted
        citations = db_session.query(MessageCitation).all()
        assert len(citations) == 0


class TestChatSessionDeletion:
    """Tests verifying owner-scoped chat session deletion, cascades, and tenant isolation."""

    def test_delete_session_success_by_owner_and_cascades(self, db_session, user_a, user_b, chat_service):
        """Verifies owner can delete chat session, cascading to messages and citations, while preserving documents, chunks, collections, and other sessions."""
        # Setup document, chunk, and collection
        doc = Document(
            id=uuid4(),
            owner_id=user_a.id,
            filename="analysis.pdf",
            file_path="/tmp/analysis.pdf",
            file_hash_sha256="deadbeef" * 8,
            file_size_bytes=1024,
            processing_status=ProcessingStatus.READY.value,
        )
        col = Collection(
            id=uuid4(),
            owner_id=user_a.id,
            name="Security Papers",
        )
        db_session.add_all([doc, col])
        db_session.commit()

        chunk = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content="Evidence passage text",
            char_start=0,
            char_end=21,
            token_count=5,
        )
        db_session.add(chunk)
        db_session.commit()

        # Session 1 to be deleted
        s1 = chat_service.create_session(
            db_session,
            user_a.id,
            ChatSessionCreateRequest(title="Chat to Delete", scope_type=ChatScopeType.GLOBAL),
        )
        msg_user = Message(
            id=uuid4(),
            chat_session_id=s1.id,
            sender_type=SenderType.USER.value,
            content="Hello",
        )
        msg_asst = Message(
            id=uuid4(),
            chat_session_id=s1.id,
            sender_type=SenderType.ASSISTANT.value,
            content="Here is evidence [CHUNK:1].",
        )
        db_session.add_all([msg_user, msg_asst])
        db_session.commit()

        cit = MessageCitation(
            id=uuid4(),
            message_id=msg_asst.id,
            chunk_id=chunk.id,
            quoted_text="Evidence passage text",
            page_number=1,
            document_title="analysis.pdf",
            similarity_score=0.95,
        )
        db_session.add(cit)
        db_session.commit()

        # Session 2 (unrelated session owned by user_a)
        s2 = chat_service.create_session(
            db_session,
            user_a.id,
            ChatSessionCreateRequest(title="Preserved Chat User A", scope_type=ChatScopeType.GLOBAL),
        )

        # Session 3 (unrelated session owned by user_b)
        s3 = chat_service.create_session(
            db_session,
            user_b.id,
            ChatSessionCreateRequest(title="Preserved Chat User B", scope_type=ChatScopeType.GLOBAL),
        )

        # Delete session 1
        chat_service.delete_session(db_session, s1.id, user_a.id)

        # 1. Verify session 1 is gone
        assert db_session.query(ChatSession).filter(ChatSession.id == s1.id).first() is None

        # 2. Verify messages and citations belonging to session 1 are deleted
        assert db_session.query(Message).filter(Message.chat_session_id == s1.id).count() == 0
        assert db_session.query(MessageCitation).filter(MessageCitation.message_id == msg_asst.id).count() == 0

        # 3. Verify Document, Chunk, and Collection remain completely intact
        assert db_session.query(Document).filter(Document.id == doc.id).first() is not None
        assert db_session.query(Chunk).filter(Chunk.id == chunk.id).first() is not None
        assert db_session.query(Collection).filter(Collection.id == col.id).first() is not None

        # 4. Verify unrelated sessions for user_a and user_b remain completely intact
        assert db_session.query(ChatSession).filter(ChatSession.id == s2.id).first() is not None
        assert db_session.query(ChatSession).filter(ChatSession.id == s3.id).first() is not None

    def test_delete_session_cross_owner_isolation(self, db_session, user_a, user_b, chat_service):
        """Verifies that non-owner cannot delete another user's session and receives 404 without leakage."""
        s = chat_service.create_session(
            db_session,
            user_a.id,
            ChatSessionCreateRequest(title="User A Session", scope_type=ChatScopeType.GLOBAL),
        )

        with pytest.raises(ChatSessionNotFoundError, match="Chat session not found."):
            chat_service.delete_session(db_session, s.id, user_b.id)

        # Ensure session was not deleted
        assert db_session.query(ChatSession).filter(ChatSession.id == s.id).first() is not None

    def test_delete_nonexistent_session_raises_not_found(self, db_session, user_a, chat_service):
        """Verifies deleting a random non-existent UUID raises ChatSessionNotFoundError."""
        with pytest.raises(ChatSessionNotFoundError, match="Chat session not found."):
            chat_service.delete_session(db_session, uuid4(), user_a.id)

    def test_delete_session_rolls_back_on_error(self, db_session, user_a, chat_service, monkeypatch):
        """Verifies transaction rollback if db.delete fails."""
        s = chat_service.create_session(
            db_session,
            user_a.id,
            ChatSessionCreateRequest(title="Session to Fail", scope_type=ChatScopeType.GLOBAL),
        )

        def raise_db_error(*args, **kwargs):
            raise sa.exc.OperationalError("Simulated database failure", {}, None)

        monkeypatch.setattr(db_session, "delete", raise_db_error)

        with pytest.raises(sa.exc.OperationalError):
            chat_service.delete_session(db_session, s.id, user_a.id)

        # Rollback was executed; session still exists
        assert db_session.query(ChatSession).filter(ChatSession.id == s.id).first() is not None


class TestChatConcurrency:
    """Verifies safe concurrent chat operation handling, busy locking, tenant isolation, and failure release."""

    def test_post_message_and_delete_session_conflict_raises_busy_error(
        self, db_session, user_a, chat_service
    ):
        """Deleting a session while generation is in flight must raise ChatSessionBusyError."""
        s = chat_service.create_session(
            db_session,
            user_a.id,
            ChatSessionCreateRequest(title="Active Session", scope_type=ChatScopeType.GLOBAL),
        )

        rag_started_event = threading.Event()
        rag_proceed_event = threading.Event()
        generation_result = {}

        mock_rag = RAGService(
            embedding_service=DummyEmbeddingService(dimension=4),
            vector_store=VectorStore(dimension=4, model_name="dummy-embedder"),
            llm_provider=MockLLMProvider(model_name="mock-model"),
        )

        def slow_answer_query(**kwargs):
            rag_started_event.set()
            assert rag_proceed_event.wait(timeout=5.0), "Timed out waiting to proceed"
            return RAGResponse(
                query=kwargs.get("query", "Hello"),
                answer="Answer after delay.",
                citations=[],
                status="SUCCESS",
                model_name="mock-model",
                provider_name="MockProvider",
                retrieved_chunk_count=0,
                completion_tokens=5,
                prompt_tokens=10,
                total_tokens=15,
            )

        mock_rag.answer_query = slow_answer_query

        # Run post_message_and_respond in a background thread
        def run_generation():
            try:
                turn = chat_service.post_message_and_respond(
                    db_session,
                    s.id,
                    user_a.id,
                    ChatMessageCreateRequest(content="Hello"),
                    mock_rag,
                )
                generation_result["turn"] = turn
            except Exception as e:
                generation_result["error"] = e

        t = threading.Thread(target=run_generation)
        t.start()

        assert rag_started_event.wait(timeout=5.0), "Generation did not start in time"

        # While generation is running, attempt delete_session
        with pytest.raises(ChatSessionBusyError) as exc_info:
            chat_service.delete_session(db_session, s.id, user_a.id)

        assert exc_info.value.code == "SESSION_BUSY"
        assert "generation is in progress" in str(exc_info.value)

        # Allow generation to finish
        rag_proceed_event.set()
        t.join(timeout=5.0)

        assert "turn" in generation_result
        assert generation_result["turn"].assistant_message.content == "Answer after delay."

        # Verify session and messages still exist
        assert db_session.query(ChatSession).filter(ChatSession.id == s.id).first() is not None
        assert db_session.query(Message).filter(Message.chat_session_id == s.id).count() == 2

    def test_duplicate_post_message_raises_busy_error(
        self, db_session, user_a, chat_service
    ):
        """Submitting a message while another message is in flight on the same session raises ChatSessionBusyError."""
        s = chat_service.create_session(
            db_session,
            user_a.id,
            ChatSessionCreateRequest(title="Duplicate Post Session", scope_type=ChatScopeType.GLOBAL),
        )

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
                answer="Slow answer.",
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

        t = threading.Thread(
            target=lambda: chat_service.post_message_and_respond(
                db_session, s.id, user_a.id, ChatMessageCreateRequest(content="First"), mock_rag
            )
        )
        t.start()

        assert rag_started_event.wait(timeout=5.0)

        # Attempt second post_message_and_respond on same session
        with pytest.raises(ChatSessionBusyError) as exc_info:
            chat_service.post_message_and_respond(
                db_session, s.id, user_a.id, ChatMessageCreateRequest(content="Second"), mock_rag
            )

        assert exc_info.value.code == "SESSION_BUSY"

        rag_proceed_event.set()
        t.join(timeout=5.0)

    def test_cross_owner_delete_while_generating_returns_not_found_not_busy(
        self, db_session, user_a, user_b, chat_service
    ):
        """A non-owner attempting delete on a busy session must receive ChatSessionNotFoundError, never ChatSessionBusyError."""
        s = chat_service.create_session(
            db_session,
            user_a.id,
            ChatSessionCreateRequest(title="User A Busy Session", scope_type=ChatScopeType.GLOBAL),
        )

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
                answer="Answer",
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

        t = threading.Thread(
            target=lambda: chat_service.post_message_and_respond(
                db_session, s.id, user_a.id, ChatMessageCreateRequest(content="Q"), mock_rag
            )
        )
        t.start()

        assert rag_started_event.wait(timeout=5.0)

        # User B tries to delete User A's active session
        with pytest.raises(ChatSessionNotFoundError, match="Chat session not found."):
            chat_service.delete_session(db_session, s.id, user_b.id)

        rag_proceed_event.set()
        t.join(timeout=5.0)

    def test_independent_sessions_do_not_block_each_other(
        self, db_session, user_a, chat_service
    ):
        """Operations on session 2 proceed smoothly even if session 1 has generation in flight."""
        s1 = chat_service.create_session(
            db_session,
            user_a.id,
            ChatSessionCreateRequest(title="Session 1", scope_type=ChatScopeType.GLOBAL),
        )
        s2 = chat_service.create_session(
            db_session,
            user_a.id,
            ChatSessionCreateRequest(title="Session 2", scope_type=ChatScopeType.GLOBAL),
        )

        rag_started_event = threading.Event()
        rag_proceed_event = threading.Event()

        mock_rag_1 = RAGService(
            embedding_service=DummyEmbeddingService(dimension=4),
            vector_store=VectorStore(dimension=4, model_name="dummy-embedder"),
            llm_provider=MockLLMProvider(model_name="mock-model"),
        )

        def slow_answer(**kwargs):
            rag_started_event.set()
            assert rag_proceed_event.wait(timeout=5.0)
            return RAGResponse(
                query=kwargs.get("query", "Q1"),
                answer="Answer 1",
                citations=[],
                status="SUCCESS",
                model_name="mock-model",
                provider_name="MockProvider",
                retrieved_chunk_count=0,
                completion_tokens=5,
                prompt_tokens=10,
                total_tokens=15,
            )

        mock_rag_1.answer_query = slow_answer

        t = threading.Thread(
            target=lambda: chat_service.post_message_and_respond(
                db_session, s1.id, user_a.id, ChatMessageCreateRequest(content="Q1"), mock_rag_1
            )
        )
        t.start()

        assert rag_started_event.wait(timeout=5.0)

        # Deleting session 2 succeeds without error
        chat_service.delete_session(db_session, s2.id, user_a.id)
        assert db_session.query(ChatSession).filter(ChatSession.id == s2.id).first() is None

        rag_proceed_event.set()
        t.join(timeout=5.0)

    def test_busy_state_released_on_generation_failure(
        self, db_session, user_a, chat_service
    ):
        """If generation raises an error, the busy reservation is cleanly released in finally."""
        s = chat_service.create_session(
            db_session,
            user_a.id,
            ChatSessionCreateRequest(title="Failing Session", scope_type=ChatScopeType.GLOBAL),
        )

        mock_rag = RAGService(
            embedding_service=DummyEmbeddingService(dimension=4),
            vector_store=VectorStore(dimension=4, model_name="dummy-embedder"),
            llm_provider=MockLLMProvider(model_name="mock-model"),
        )

        def failing_answer(**kwargs):
            raise LLMProviderError("Upstream failure")

        mock_rag.answer_query = failing_answer

        with pytest.raises(LLMProviderError):
            chat_service.post_message_and_respond(
                db_session, s.id, user_a.id, ChatMessageCreateRequest(content="Failing Q"), mock_rag
            )

        # Busy state should be released immediately; delete_session must succeed
        chat_service.delete_session(db_session, s.id, user_a.id)
        assert db_session.query(ChatSession).filter(ChatSession.id == s.id).first() is None


class TestMultiDocumentScope:
    """Tests for multi-document ('DOCUMENTS') chat scope creation, authorization, turn execution, and deletion."""

    def test_create_documents_scope_session_success(self, db_session, user_a, chat_service):
        doc1 = Document(
            id=uuid4(),
            owner_id=user_a.id,
            filename="paper1.pdf",
            title="Paper 1 Title",
            file_path="/tmp/paper1.pdf",
            file_hash_sha256="hash1",
            file_size_bytes=1000,
            page_count=5,
            processing_status=ProcessingStatus.READY.value,
        )
        doc2 = Document(
            id=uuid4(),
            owner_id=user_a.id,
            filename="paper2.pdf",
            title="Paper 2 Title",
            file_path="/tmp/paper2.pdf",
            file_hash_sha256="hash2",
            file_size_bytes=2000,
            page_count=8,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add_all([doc1, doc2])
        db_session.commit()

        req = ChatSessionCreateRequest(
            title="Multi-Doc Chat",
            scope_type=ChatScopeType.DOCUMENTS,
            document_ids=[doc1.id, doc2.id],
        )
        session_resp = chat_service.create_session(db_session, user_a.id, req)

        assert session_resp.title == "Multi-Doc Chat"
        assert session_resp.scope_type == ChatScopeType.DOCUMENTS
        assert session_resp.target_document_id is None
        assert session_resp.target_collection_id is None
        assert len(session_resp.selected_documents) == 2
        selected_ids = {d.id for d in session_resp.selected_documents}
        assert selected_ids == {doc1.id, doc2.id}

        # Verify get_session loads selected_documents
        fetched = chat_service.get_session(db_session, session_resp.id, user_a.id)
        assert len(fetched.selected_documents) == 2

        # Verify list_sessions loads selected_documents without error
        listed = chat_service.list_sessions(db_session, user_a.id)
        assert listed.total == 1
        assert len(listed.items[0].selected_documents) == 2

    def test_create_documents_scope_cross_tenant_rejected(self, db_session, user_a, user_b, chat_service):
        doc_b = Document(
            id=uuid4(),
            owner_id=user_b.id,
            filename="paper_b.pdf",
            file_path="/tmp/b.pdf",
            file_hash_sha256="hashb",
            file_size_bytes=1000,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc_b)
        db_session.commit()

        req = ChatSessionCreateRequest(
            title="Cross-tenant Attack",
            scope_type=ChatScopeType.DOCUMENTS,
            document_ids=[doc_b.id],
        )
        with pytest.raises(TargetEntityNotFoundError):
            chat_service.create_session(db_session, user_a.id, req)

    def test_create_documents_scope_unready_document_rejected(self, db_session, user_a, chat_service):
        doc_pending = Document(
            id=uuid4(),
            owner_id=user_a.id,
            filename="pending.pdf",
            file_path="/tmp/pending.pdf",
            file_hash_sha256="hash_pending",
            file_size_bytes=1000,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db_session.add(doc_pending)
        db_session.commit()

        req = ChatSessionCreateRequest(
            title="Pending Chat",
            scope_type=ChatScopeType.DOCUMENTS,
            document_ids=[doc_pending.id],
        )
        with pytest.raises(TargetEntityNotReadyError):
            chat_service.create_session(db_session, user_a.id, req)

    def test_create_documents_scope_empty_or_duplicate_rejected(self, user_a):
        doc_id = uuid4()
        with pytest.raises(ValueError):
            ChatSessionCreateRequest(
                title="Empty Docs",
                scope_type=ChatScopeType.DOCUMENTS,
                document_ids=[],
            )

        with pytest.raises(ValueError):
            ChatSessionCreateRequest(
                title="Duplicate Docs",
                scope_type=ChatScopeType.DOCUMENTS,
                document_ids=[doc_id, doc_id],
            )

    def test_multi_document_scoped_retrieval(self, db_session, user_a, chat_service):
        # Doc 1 & Doc 2 are scoped to session; Doc 3 is owned by user_a but NOT in this session
        doc1 = Document(
            id=uuid4(),
            owner_id=user_a.id,
            filename="doc1.pdf",
            title="Doc 1",
            file_path="/tmp/doc1.pdf",
            file_hash_sha256="h1",
            file_size_bytes=100,
            page_count=1,
            processing_status=ProcessingStatus.READY.value,
        )
        doc2 = Document(
            id=uuid4(),
            owner_id=user_a.id,
            filename="doc2.pdf",
            title="Doc 2",
            file_path="/tmp/doc2.pdf",
            file_hash_sha256="h2",
            file_size_bytes=100,
            page_count=1,
            processing_status=ProcessingStatus.READY.value,
        )
        doc3 = Document(
            id=uuid4(),
            owner_id=user_a.id,
            filename="doc3.pdf",
            title="Doc 3",
            file_path="/tmp/doc3.pdf",
            file_hash_sha256="h3",
            file_size_bytes=100,
            page_count=1,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add_all([doc1, doc2, doc3])
        db_session.flush()

        chunk1 = Chunk(id=101, document_id=doc1.id, chunk_index=0, content="Content from Doc 1 [CHUNK:101]", page_number=1, char_start=0, char_end=30, token_count=10)
        chunk2 = Chunk(id=102, document_id=doc2.id, chunk_index=0, content="Content from Doc 2 [CHUNK:102]", page_number=1, char_start=0, char_end=30, token_count=10)
        chunk3 = Chunk(id=103, document_id=doc3.id, chunk_index=0, content="Unselected doc content with high similarity", page_number=1, char_start=0, char_end=44, token_count=10)
        db_session.add_all([chunk1, chunk2, chunk3])
        db_session.commit()

        # Vector store indexing: chunk 3 has vector closest to query, chunks 1 & 2 are further
        dim = 4
        store = VectorStore(dimension=dim, model_name="dummy-embedder")
        # chunk 1 & 2: [0.5, 0.5, 0.5, 0.5]
        v1 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        v2 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        # chunk 3: [1.0, 0, 0, 0] (highest similarity to [1, 0, 0, 0])
        v3 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        store.add_vectors([101, 102, 103], [v1, v2, v3])

        class QuerySpecificEmbeddingService(DummyEmbeddingService):
            def embed_text(self, text: str) -> np.ndarray:
                return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        rag_service = RAGService(
            embedding_service=QuerySpecificEmbeddingService(dimension=dim),
            vector_store=store,
            llm_provider=MockLLMProvider(model_name="mock-model"),
        )

        session = chat_service.create_session(
            db_session,
            user_a.id,
            ChatSessionCreateRequest(
                title="Multi Scope Chat",
                scope_type=ChatScopeType.DOCUMENTS,
                document_ids=[doc1.id, doc2.id],
            ),
        )

        turn = chat_service.post_message_and_respond(
            db=db_session,
            session_id=session.id,
            user_id=user_a.id,
            request=ChatMessageCreateRequest(content="Compare doc1 and doc2"),
            rag_service=rag_service,
        )

        # Chunks from doc1 or doc2 are retrieved, but chunk 103 from doc3 must NEVER be included
        retrieved_chunk_ids = [c.chunk_id for c in turn.assistant_message.citations]
        assert 103 not in retrieved_chunk_ids

    def test_document_deletion_in_multi_document_session(self, db_session, user_a, chat_service):
        from app.services.document_service import document_service

        doc1 = Document(
            id=uuid4(),
            owner_id=user_a.id,
            filename="paper_to_delete.pdf",
            title="Paper A",
            file_path="/tmp/a.pdf",
            file_hash_sha256="h_del_1",
            file_size_bytes=100,
            processing_status=ProcessingStatus.READY.value,
        )
        doc2 = Document(
            id=uuid4(),
            owner_id=user_a.id,
            filename="paper_to_keep.pdf",
            title="Paper B",
            file_path="/tmp/b.pdf",
            file_hash_sha256="h_del_2",
            file_size_bytes=100,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add_all([doc1, doc2])
        db_session.flush()

        chunk1 = Chunk(id=201, document_id=doc1.id, chunk_index=0, content="Evidence in doc1", page_number=1, char_start=0, char_end=16, token_count=10)
        db_session.add(chunk1)
        db_session.commit()

        session = chat_service.create_session(
            db_session,
            user_a.id,
            ChatSessionCreateRequest(
                title="Session with Two Docs",
                scope_type=ChatScopeType.DOCUMENTS,
                document_ids=[doc1.id, doc2.id],
            ),
        )

        # Create message with citation snapshot referencing chunk 201
        msg = Message(chat_session_id=session.id, sender_type=SenderType.ASSISTANT.value, content="Answer [CHUNK:201]")
        db_session.add(msg)
        db_session.flush()
        cit = MessageCitation(
            message_id=msg.id,
            chunk_id=201,
            quoted_text="Evidence in doc1",
            page_number=1,
            document_title="Paper A",
            similarity_score=0.9,
        )
        db_session.add(cit)
        db_session.commit()

        # Delete doc1: session should NOT be archived because doc2 remains!
        document_service.delete_document(db_session, user_a.id, doc1.id)

        session_after_doc1 = chat_service.get_session(db_session, session.id, user_a.id)
        assert session_after_doc1.is_archived is False
        assert len(session_after_doc1.selected_documents) == 1
        assert session_after_doc1.selected_documents[0].id == doc2.id

        # Citation snapshot must be preserved with chunk_id=None
        db_session.refresh(cit)
        assert cit.chunk_id is None
        assert cit.quoted_text == "Evidence in doc1"
        assert cit.document_title == "Paper A"

        # Now delete doc2: last remaining document -> session MUST be archived!
        document_service.delete_document(db_session, user_a.id, doc2.id)

        db_session.expire_all()
        session_after_doc2 = chat_service.get_session(db_session, session.id, user_a.id)
        assert session_after_doc2.is_archived is True
        assert len(session_after_doc2.selected_documents) == 0

        # Cannot post to archived session
        mock_rag = RAGService(
            embedding_service=DummyEmbeddingService(dimension=4),
            vector_store=VectorStore(dimension=4, model_name="dummy-embedder"),
            llm_provider=MockLLMProvider(model_name="mock-model"),
        )
        with pytest.raises(ChatSessionArchivedError):
            chat_service.post_message_and_respond(
                db=db_session,
                session_id=session.id,
                user_id=user_a.id,
                request=ChatMessageCreateRequest(content="Try to post"),
                rag_service=mock_rag,
            )

    def test_stale_reservation_not_evicted_while_thread_actively_running(self):
        """Proves that ChatConcurrencyManager does not release an in-flight reservation while its thread is actively running."""
        import time
        from uuid import uuid4
        from app.services.chat_concurrency import ChatConcurrencyManager, ChatSessionBusyError

        # Short stale timeout: 0.05s
        mgr = ChatConcurrencyManager(stale_timeout_seconds=0.05)
        sid = uuid4()

        entered_event = threading.Event()
        release_event = threading.Event()

        def active_worker():
            with mgr.session_reservation(sid, "GENERATING"):
                entered_event.set()
                release_event.wait(timeout=5.0)

        t = threading.Thread(target=active_worker)
        t.start()
        assert entered_event.wait(timeout=2.0)

        # Sleep longer than the stale timeout (0.1s > 0.05s)
        time.sleep(0.1)

        # Because thread is alive and is_active is True, manager must NOT evict the reservation!
        assert mgr.is_busy(sid) is True
        assert mgr.get_active_operation(sid) == "GENERATING"

        # Attempting concurrent reservation must still fail closed with ChatSessionBusyError
        with pytest.raises(ChatSessionBusyError):
            mgr.reserve(sid, "DELETING")

        # Now release the active worker and join
        release_event.set()
        t.join(timeout=2.0)

        # Now that worker has finished, reservation is cleanly released
        assert mgr.is_busy(sid) is False
        assert mgr.get_active_operation(sid) is None

    def test_orphaned_reservation_from_dead_thread_is_evicted_after_timeout(self):
        """Proves that an unreleased reservation from a dead/abandoned thread is safely evicted after timeout."""
        import time
        from uuid import uuid4
        from app.services.chat_concurrency import ChatConcurrencyManager, _ReservationEntry

        mgr = ChatConcurrencyManager(stale_timeout_seconds=0.05)
        sid = uuid4()

        def dead_worker():
            # Acquire direct reservation and terminate thread without releasing
            mgr.reserve(sid, "GENERATING")

        t = threading.Thread(target=dead_worker)
        t.start()
        t.join(timeout=2.0)
        assert not t.is_alive()

        # Thread is dead; wait for timeout to expire
        time.sleep(0.1)

        # Stale cleanup should detect dead thread and evict
        assert mgr.is_busy(sid) is False
        assert mgr.get_active_operation(sid) is None

    def test_superseded_operation_cannot_release_newer_reservation(self):
        """Proves that a stale release attempt from a superseded operation cannot invalidate a newer reservation."""
        from uuid import uuid4
        from app.services.chat_concurrency import ChatConcurrencyManager

        mgr = ChatConcurrencyManager()
        sid = uuid4()

        # Operation 1 acquires reservation
        entry1 = mgr.reserve(sid, "GENERATING")
        assert mgr.is_busy(sid) is True

        # Simulate administrative release or edge-case eviction
        mgr.release(sid, entry=entry1)
        assert mgr.is_busy(sid) is False

        # Operation 2 acquires newer reservation on same session
        entry2 = mgr.reserve(sid, "DELETING")
        assert mgr.is_busy(sid) is True
        assert mgr.get_active_operation(sid) == "DELETING"

        # Operation 1 delayed finally block attempts to release
        mgr.release(sid, entry=entry1)

        # Operation 2 reservation MUST NOT be released
        assert mgr.is_busy(sid) is True
        assert mgr.get_active_operation(sid) == "DELETING"

        # Operation 2 legitimate release
        mgr.release(sid, entry=entry2)
        assert mgr.is_busy(sid) is False

    def test_abandoned_reservation_on_live_pooled_thread_is_evicted_after_timeout(self):
        """Proves that an unmanaged/abandoned reservation on a reusable, idle thread-pool worker is evicted after timeout."""
        import concurrent.futures
        import time
        from uuid import uuid4
        from app.services.chat_concurrency import ChatConcurrencyManager

        # 1. Create a ChatConcurrencyManager with a short stale timeout
        mgr = ChatConcurrencyManager(stale_timeout_seconds=0.05)
        sid = uuid4()

        # 10. Ensure the test shuts down the worker pool cleanly via context manager
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            # 2. Submit a task to the worker pool that directly acquires a reservation outside session_reservation()
            # 3. Allow that task to finish and return to the pool without releasing its reservation
            future1 = pool.submit(lambda: (mgr.reserve(sid, "GENERATING"), threading.get_ident()))
            entry1, worker_thread_id = future1.result(timeout=2.0)

            # 4. Confirm that the task has completed
            assert future1.done() is True
            assert mgr.is_busy(sid) is True

            # 5. Confirm that the worker thread is still alive and can execute another task
            future2 = pool.submit(lambda: threading.get_ident())
            reused_thread_id = future2.result(timeout=2.0)
            assert future2.done() is True
            assert reused_thread_id == worker_thread_id

            alive_thread_ids = {t.ident for t in threading.enumerate() if t.is_alive()}
            assert worker_thread_id in alive_thread_ids

            # 6. Wait until the reservation exceeds its stale timeout (0.05s) using bounded elapsed polling
            deadline = time.monotonic() + 1.0
            while time.monotonic() - entry1.started_at <= 0.05 and time.monotonic() < deadline:
                time.sleep(0.01)

            # 7 & 8. Trigger the manager's normal cleanup path and verify eviction
            assert mgr.is_busy(sid) is False
            assert mgr.get_active_operation(sid) is None

            # 9. Verify that a new operation can now reserve the session
            new_entry = mgr.reserve(sid, "DELETING")
            assert mgr.is_busy(sid) is True
            assert mgr.get_active_operation(sid) == "DELETING"
            mgr.release(sid, entry=new_entry)
            assert mgr.is_busy(sid) is False

    def test_completed_operation_releases_reservation_before_thread_reused(self):
        """Proves that when an operation finishes in session_reservation, it is immediately released."""
        from uuid import uuid4
        from app.services.chat_concurrency import ChatConcurrencyManager

        mgr = ChatConcurrencyManager()
        sid = uuid4()

        # Run operation within session_reservation
        with mgr.session_reservation(sid, "GENERATING") as entry:
            assert mgr.is_busy(sid) is True
            assert entry.in_context is True
            assert mgr.get_active_operation(sid) == "GENERATING"

        # Context exited cleanly: session is immediately unreserved
        assert mgr.is_busy(sid) is False
        assert mgr.get_active_operation(sid) is None
        assert entry.in_context is False
        assert entry.is_active is False




