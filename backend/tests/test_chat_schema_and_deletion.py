"""Tests for Chat schemas and target-deletion safety in Document and Collection services.

Verifies:
1. Schema Validation:
   - ChatSessionCreateRequest: scope pairing, forbidden user_id, title trimming, extra field rejection.
   - ChatSessionUpdateRequest: title, is_archived, empty update rejection, extra field rejection.
   - ChatMessageCreateRequest: content validation, bounds for top_k/similarity_threshold, extra field rejection.
   - Response Schemas: ChatSessionResponse, ChatMessageResponse, MessageCitationResponse, ChatTurnResponse.
2. Target Deletion Safety:
   - Deleting a document safely archives active scoped chat sessions without violating ck_chat_session_scope_targets.
   - Deleting a collection safely archives active collection-scoped sessions.
   - Unrelated sessions (global, other documents, other collections, already archived) remain untouched.
   - MessageCitation historical evidence snapshots survive document and chunk deletion.
   - Database rollback cleanly restores both target deletion and session archiving on failure.
"""
from datetime import datetime, timezone
from typing import Generator
import uuid
from uuid import uuid4

import pytest
from pydantic import ValidationError
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
    ChatMessageListResponse,
    ChatMessageResponse,
    ChatSessionCreateRequest,
    ChatSessionListResponse,
    ChatSessionResponse,
    ChatSessionUpdateRequest,
    ChatTurnResponse,
    MessageCitationResponse,
)
from app.services.collection_service import collection_service
from app.services.document_service import DocumentNotFoundError, document_service


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
def test_user(db_session: Session) -> User:
    user = User(
        id=uuid4(),
        email="researcher@example.com",
        hashed_password="hashed_test_password",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def second_user(db_session: Session) -> User:
    user = User(
        id=uuid4(),
        email="other@example.com",
        hashed_password="hashed_other_password",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


# ===========================================================================
# 1. Schema Validation Tests
# ===========================================================================

class TestChatSessionCreateSchema:
    """Tests validation for ChatSessionCreateRequest."""

    def test_valid_global_session(self):
        req = ChatSessionCreateRequest(scope_type=ChatScopeType.GLOBAL)
        assert req.scope_type == ChatScopeType.GLOBAL
        assert req.target_document_id is None
        assert req.target_collection_id is None
        assert req.title is None

    def test_valid_document_session(self):
        doc_id = uuid4()
        req = ChatSessionCreateRequest(
            title="Paper Discussion",
            scope_type=ChatScopeType.DOCUMENT,
            target_document_id=doc_id,
        )
        assert req.scope_type == ChatScopeType.DOCUMENT
        assert req.target_document_id == doc_id
        assert req.target_collection_id is None
        assert req.title == "Paper Discussion"

    def test_valid_collection_session(self):
        col_id = uuid4()
        req = ChatSessionCreateRequest(
            title="Collection Thread",
            scope_type=ChatScopeType.COLLECTION,
            target_collection_id=col_id,
        )
        assert req.scope_type == ChatScopeType.COLLECTION
        assert req.target_collection_id == col_id
        assert req.target_document_id is None

    def test_title_whitespace_trimmed(self):
        req = ChatSessionCreateRequest(title="  Trimmed Title  ")
        assert req.title == "Trimmed Title"

    def test_empty_or_whitespace_title_rejected(self):
        with pytest.raises(ValidationError, match="Title must contain non-whitespace characters"):
            ChatSessionCreateRequest(title="   ")

    def test_title_exceeding_max_length_rejected(self):
        with pytest.raises(ValidationError, match="Title cannot exceed 255 characters"):
            ChatSessionCreateRequest(title="x" * 256)

    def test_global_scope_with_targets_rejected(self):
        with pytest.raises(ValidationError, match="GLOBAL conversations must not specify"):
            ChatSessionCreateRequest(
                scope_type=ChatScopeType.GLOBAL,
                target_document_id=uuid4(),
            )

        with pytest.raises(ValidationError, match="GLOBAL conversations must not specify"):
            ChatSessionCreateRequest(
                scope_type=ChatScopeType.GLOBAL,
                target_collection_id=uuid4(),
            )

    def test_document_scope_missing_document_id_rejected(self):
        with pytest.raises(ValidationError, match="DOCUMENT conversations require target_document_id"):
            ChatSessionCreateRequest(scope_type=ChatScopeType.DOCUMENT)

    def test_document_scope_with_collection_id_rejected(self):
        with pytest.raises(ValidationError, match="DOCUMENT conversations cannot specify target_collection_id"):
            ChatSessionCreateRequest(
                scope_type=ChatScopeType.DOCUMENT,
                target_document_id=uuid4(),
                target_collection_id=uuid4(),
            )

    def test_collection_scope_missing_collection_id_rejected(self):
        with pytest.raises(ValidationError, match="COLLECTION conversations require target_collection_id"):
            ChatSessionCreateRequest(scope_type=ChatScopeType.COLLECTION)

    def test_collection_scope_with_document_id_rejected(self):
        with pytest.raises(ValidationError, match="COLLECTION conversations cannot specify target_document_id"):
            ChatSessionCreateRequest(
                scope_type=ChatScopeType.COLLECTION,
                target_collection_id=uuid4(),
                target_document_id=uuid4(),
            )

    def test_extra_fields_and_user_id_rejected(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            ChatSessionCreateRequest.model_validate({"user_id": str(uuid4())})

        with pytest.raises(ValidationError, match="extra_forbidden"):
            ChatSessionCreateRequest.model_validate({"owner_id": str(uuid4())})


class TestChatSessionUpdateSchema:
    """Tests validation for ChatSessionUpdateRequest."""

    def test_valid_title_update(self):
        req = ChatSessionUpdateRequest(title="New Title")
        assert req.title == "New Title"
        assert req.is_archived is None

    def test_valid_archive_update(self):
        req = ChatSessionUpdateRequest(is_archived=True)
        assert req.title is None
        assert req.is_archived is True

    def test_empty_update_rejected(self):
        with pytest.raises(ValidationError, match="At least one field"):
            ChatSessionUpdateRequest()

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            ChatSessionUpdateRequest.model_validate({"title": "Valid", "user_id": str(uuid4())})


class TestChatMessageCreateSchema:
    """Tests validation for ChatMessageCreateRequest."""

    def test_valid_message_create(self):
        req = ChatMessageCreateRequest(content="What are the key findings?")
        assert req.content == "What are the key findings?"
        assert req.top_k == 5
        assert req.similarity_threshold == 0.0

    def test_content_whitespace_trimmed(self):
        req = ChatMessageCreateRequest(content="  trimmed message  ")
        assert req.content == "trimmed message"

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError, match="Message content must contain non-whitespace characters"):
            ChatMessageCreateRequest(content="   \n\t  ")

    def test_content_length_exceeding_10000_rejected(self):
        with pytest.raises(ValidationError, match="Message content cannot exceed 10,000 characters"):
            ChatMessageCreateRequest(content="a" * 10001)

    def test_top_k_bounds(self):
        with pytest.raises(ValidationError):
            ChatMessageCreateRequest(content="valid", top_k=0)
        with pytest.raises(ValidationError):
            ChatMessageCreateRequest(content="valid", top_k=51)

    def test_similarity_threshold_bounds(self):
        with pytest.raises(ValidationError):
            ChatMessageCreateRequest(content="valid", similarity_threshold=-1.1)
        with pytest.raises(ValidationError):
            ChatMessageCreateRequest(content="valid", similarity_threshold=1.1)

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError, match="extra_forbidden"):
            ChatMessageCreateRequest.model_validate({"content": "valid", "user_id": str(uuid4())})


class TestResponseSchemaSerialization:
    """Tests serialization of response DTOs."""

    def test_message_citation_response_serialization(self):
        cit = MessageCitationResponse(
            id=uuid4(),
            message_id=uuid4(),
            chunk_id=101,
            quoted_text="Attention is all you need.",
            page_number=1,
            document_title="Attention Paper",
            similarity_score=0.91,
            created_at=datetime.now(timezone.utc),
        )
        data = cit.model_dump()
        assert data["chunk_id"] == 101
        assert data["quoted_text"] == "Attention is all you need."

    def test_chat_turn_response_serialization(self):
        session_id = uuid4()
        user_msg = ChatMessageResponse(
            id=uuid4(),
            chat_session_id=session_id,
            sender_type=SenderType.USER,
            content="Explain self-attention.",
            token_count=5,
            created_at=datetime.now(timezone.utc),
            citations=[],
        )
        asst_msg = ChatMessageResponse(
            id=uuid4(),
            chat_session_id=session_id,
            sender_type=SenderType.ASSISTANT,
            content="Self-attention computes representations by comparing positions [CHUNK:101].",
            token_count=15,
            created_at=datetime.now(timezone.utc),
            citations=[
                MessageCitationResponse(
                    id=uuid4(),
                    message_id=user_msg.id,
                    chunk_id=101,
                    quoted_text="Self-attention relates different positions.",
                    page_number=2,
                    document_title="Attention Paper",
                    similarity_score=0.95,
                    created_at=datetime.now(timezone.utc),
                )
            ],
        )
        turn = ChatTurnResponse(
            session_id=session_id,
            user_message=user_msg,
            assistant_message=asst_msg,
            retrieved_chunk_count=1,
            model_name="mock-model-v1",
            provider_name="mock",
            prompt_tokens=5,
            completion_tokens=15,
            total_tokens=20,
        )
        assert turn.session_id == session_id
        assert len(turn.assistant_message.citations) == 1
        assert turn.total_tokens == 20


# ===========================================================================
# 2. Target Deletion Safety Tests
# ===========================================================================

class TestDocumentDeletionSafety:
    """Tests verifying document deletion safely archives referenced chat sessions."""

    def test_delete_document_archives_active_scoped_chat_sessions(
        self,
        db_session: Session,
        test_user: User,
    ):
        """Verifies deleting a document archives active sessions without violating ck_chat_session_scope_targets."""
        doc = Document(
            owner_id=test_user.id,
            filename="paper.pdf",
            file_path="/tmp/test_paper.pdf",
            file_hash_sha256="sha256_doc_del_1",
            file_size_bytes=2048,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.commit()

        # Create active chat session scoped to this document
        session_1 = ChatSession(
            owner_id=test_user.id,
            title="Paper Chat 1",
            scope_type=ChatScopeType.DOCUMENT.value,
            target_document_id=doc.id,
            is_archived=False,
        )
        session_2 = ChatSession(
            owner_id=test_user.id,
            title="Paper Chat 2",
            scope_type=ChatScopeType.DOCUMENT.value,
            target_document_id=doc.id,
            is_archived=False,
        )
        db_session.add_all([session_1, session_2])
        db_session.commit()

        # Delete document via DocumentService
        document_service.delete_document(db=db_session, owner_id=test_user.id, document_id=doc.id)

        # Verify document is deleted
        assert db_session.get(Document, doc.id) is None

        # Verify both chat sessions were safely archived
        refreshed_s1 = db_session.get(ChatSession, session_1.id)
        refreshed_s2 = db_session.get(ChatSession, session_2.id)

        assert refreshed_s1 is not None
        assert refreshed_s1.is_archived is True
        assert refreshed_s1.target_document_id is None
        assert refreshed_s1.scope_type == ChatScopeType.DOCUMENT.value  # NEVER silently converted to GLOBAL

        assert refreshed_s2 is not None
        assert refreshed_s2.is_archived is True
        assert refreshed_s2.target_document_id is None
        assert refreshed_s2.scope_type == ChatScopeType.DOCUMENT.value

    def test_delete_document_preserves_unrelated_sessions(
        self,
        db_session: Session,
        test_user: User,
    ):
        """Verifies deleting Document A leaves Document B, Collection, and Global sessions untouched."""
        doc_a = Document(
            owner_id=test_user.id,
            filename="doc_a.pdf",
            file_path="/tmp/doc_a.pdf",
            file_hash_sha256="sha256_doc_a",
            file_size_bytes=1024,
            processing_status=ProcessingStatus.READY.value,
        )
        doc_b = Document(
            owner_id=test_user.id,
            filename="doc_b.pdf",
            file_path="/tmp/doc_b.pdf",
            file_hash_sha256="sha256_doc_b",
            file_size_bytes=1024,
            processing_status=ProcessingStatus.READY.value,
        )
        col = Collection(owner_id=test_user.id, name="Active Col")
        db_session.add_all([doc_a, doc_b, col])
        db_session.commit()

        session_a = ChatSession(
            owner_id=test_user.id,
            title="Chat A",
            scope_type=ChatScopeType.DOCUMENT.value,
            target_document_id=doc_a.id,
            is_archived=False,
        )
        session_b = ChatSession(
            owner_id=test_user.id,
            title="Chat B",
            scope_type=ChatScopeType.DOCUMENT.value,
            target_document_id=doc_b.id,
            is_archived=False,
        )
        session_col = ChatSession(
            owner_id=test_user.id,
            title="Chat Col",
            scope_type=ChatScopeType.COLLECTION.value,
            target_collection_id=col.id,
            is_archived=False,
        )
        session_global = ChatSession(
            owner_id=test_user.id,
            title="Chat Global",
            scope_type=ChatScopeType.GLOBAL.value,
            is_archived=False,
        )
        db_session.add_all([session_a, session_b, session_col, session_global])
        db_session.commit()

        # Delete Document A
        document_service.delete_document(db=db_session, owner_id=test_user.id, document_id=doc_a.id)

        # Session A must be archived
        refreshed_a = db_session.get(ChatSession, session_a.id)
        assert refreshed_a.is_archived is True
        assert refreshed_a.target_document_id is None

        # Session B must remain ACTIVE
        refreshed_b = db_session.get(ChatSession, session_b.id)
        assert refreshed_b.is_archived is False
        assert refreshed_b.target_document_id == doc_b.id

        # Collection session must remain ACTIVE
        refreshed_col = db_session.get(ChatSession, session_col.id)
        assert refreshed_col.is_archived is False
        assert refreshed_col.target_collection_id == col.id

        # Global session must remain ACTIVE
        refreshed_global = db_session.get(ChatSession, session_global.id)
        assert refreshed_global.is_archived is False

    def test_delete_document_preserves_citation_snapshots_and_messages(
        self,
        db_session: Session,
        test_user: User,
    ):
        """Verifies deleting the document archives the session while preserving message history and citation snapshots."""
        doc = Document(
            owner_id=test_user.id,
            filename="citation_doc.pdf",
            file_path="/tmp/citation_doc.pdf",
            file_hash_sha256="sha256_cit_doc",
            file_size_bytes=1024,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.commit()

        chunk = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=3,
            content="Evidence passage text for citation verification.",
            char_start=0,
            char_end=48,
            token_count=10,
        )
        session = ChatSession(
            owner_id=test_user.id,
            title="Citation Research",
            scope_type=ChatScopeType.DOCUMENT.value,
            target_document_id=doc.id,
            is_archived=False,
        )
        db_session.add_all([chunk, session])
        db_session.commit()

        msg = Message(
            chat_session_id=session.id,
            sender_type=SenderType.ASSISTANT.value,
            content="Based on the paper: Evidence passage text [CHUNK:1].",
            token_count=12,
        )
        db_session.add(msg)
        db_session.commit()

        citation = MessageCitation(
            message_id=msg.id,
            chunk_id=chunk.id,
            quoted_text="Evidence passage text for citation verification.",
            page_number=3,
            document_title="Citation Doc Title",
            similarity_score=0.94,
        )
        db_session.add(citation)
        db_session.commit()
        cit_id = citation.id
        msg_id = msg.id

        # Delete document
        document_service.delete_document(db=db_session, owner_id=test_user.id, document_id=doc.id)

        # Message survives
        saved_msg = db_session.get(Message, msg_id)
        assert saved_msg is not None
        assert saved_msg.content == "Based on the paper: Evidence passage text [CHUNK:1]."

        # Citation snapshot survives with chunk_id set to NULL
        saved_cit = db_session.get(MessageCitation, cit_id)
        assert saved_cit is not None
        assert saved_cit.chunk_id is None
        assert saved_cit.quoted_text == "Evidence passage text for citation verification."
        assert saved_cit.page_number == 3
        assert saved_cit.document_title == "Citation Doc Title"
        assert saved_cit.similarity_score == pytest.approx(0.94)

    def test_delete_document_transaction_rollback_on_failure(
        self,
        db_session: Session,
        test_user: User,
        monkeypatch,
    ):
        """Verifies session archiving rolls back cleanly if commit fails."""
        doc = Document(
            owner_id=test_user.id,
            filename="fail_doc.pdf",
            file_path="/tmp/fail_doc.pdf",
            file_hash_sha256="sha256_fail_doc",
            file_size_bytes=1024,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.commit()

        session = ChatSession(
            owner_id=test_user.id,
            title="Fail Doc Chat",
            scope_type=ChatScopeType.DOCUMENT.value,
            target_document_id=doc.id,
            is_archived=False,
        )
        db_session.add(session)
        db_session.commit()

        def _raise_commit_error():
            raise RuntimeError("Simulated database commit failure.")

        monkeypatch.setattr(db_session, "commit", _raise_commit_error)

        with pytest.raises(RuntimeError, match="Simulated database commit failure"):
            document_service.delete_document(db=db_session, owner_id=test_user.id, document_id=doc.id)

        # Verify rollback: document still exists and chat session remains active
        db_session.rollback()
        assert db_session.get(Document, doc.id) is not None
        refreshed_session = db_session.get(ChatSession, session.id)
        assert refreshed_session.is_archived is False
        assert refreshed_session.target_document_id == doc.id


# ===========================================================================
# 3. Collection Deletion Safety Tests
# ===========================================================================

class TestCollectionDeletionSafety:
    """Tests verifying collection deletion safely archives referenced chat sessions."""

    def test_delete_collection_archives_active_scoped_chat_sessions(
        self,
        db_session: Session,
        test_user: User,
    ):
        """Verifies deleting a collection archives active sessions without violating ck_chat_session_scope_targets."""
        col = Collection(owner_id=test_user.id, name="Vision Research")
        db_session.add(col)
        db_session.commit()

        session = ChatSession(
            owner_id=test_user.id,
            title="Vision Chat",
            scope_type=ChatScopeType.COLLECTION.value,
            target_collection_id=col.id,
            is_archived=False,
        )
        db_session.add(session)
        db_session.commit()

        # Delete collection via CollectionService
        collection_service.delete_collection(db=db_session, owner_id=test_user.id, collection_id=col.id)

        # Collection is deleted
        assert db_session.get(Collection, col.id) is None

        # Chat session is archived
        refreshed_session = db_session.get(ChatSession, session.id)
        assert refreshed_session is not None
        assert refreshed_session.is_archived is True
        assert refreshed_session.target_collection_id is None
        assert refreshed_session.scope_type == ChatScopeType.COLLECTION.value

    def test_delete_collection_preserves_documents_and_document_sessions(
        self,
        db_session: Session,
        test_user: User,
    ):
        """Verifies deleting a collection leaves documents intact and document-scoped sessions active."""
        col = Collection(owner_id=test_user.id, name="NLP Collection")
        doc = Document(
            owner_id=test_user.id,
            filename="nlp_doc.pdf",
            file_path="/tmp/nlp_doc.pdf",
            file_hash_sha256="sha256_nlp",
            file_size_bytes=1024,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add_all([col, doc])
        db_session.commit()

        # Link document to collection
        col_doc = CollectionDocument(collection_id=col.id, document_id=doc.id)
        db_session.add(col_doc)

        # Session targeting collection
        col_session = ChatSession(
            owner_id=test_user.id,
            title="Col Chat",
            scope_type=ChatScopeType.COLLECTION.value,
            target_collection_id=col.id,
            is_archived=False,
        )
        # Session targeting document
        doc_session = ChatSession(
            owner_id=test_user.id,
            title="Doc Chat",
            scope_type=ChatScopeType.DOCUMENT.value,
            target_document_id=doc.id,
            is_archived=False,
        )
        db_session.add_all([col_session, doc_session])
        db_session.commit()

        # Delete collection
        collection_service.delete_collection(db=db_session, owner_id=test_user.id, collection_id=col.id)

        # Collection is deleted
        assert db_session.get(Collection, col.id) is None

        # Document is NOT deleted
        assert db_session.get(Document, doc.id) is not None

        # Collection session is archived
        refreshed_col_session = db_session.get(ChatSession, col_session.id)
        assert refreshed_col_session.is_archived is True
        assert refreshed_col_session.target_collection_id is None

        # Document session remains ACTIVE
        refreshed_doc_session = db_session.get(ChatSession, doc_session.id)
        assert refreshed_doc_session.is_archived is False
        assert refreshed_doc_session.target_document_id == doc.id

    def test_delete_collection_transaction_rollback_on_failure(
        self,
        db_session: Session,
        test_user: User,
        monkeypatch,
    ):
        """Verifies collection deletion and session archiving roll back on commit failure."""
        col = Collection(owner_id=test_user.id, name="Rollback Col")
        db_session.add(col)
        db_session.commit()

        session = ChatSession(
            owner_id=test_user.id,
            title="Rollback Col Chat",
            scope_type=ChatScopeType.COLLECTION.value,
            target_collection_id=col.id,
            is_archived=False,
        )
        db_session.add(session)
        db_session.commit()

        def _raise_commit_error():
            raise RuntimeError("Simulated collection commit failure.")

        monkeypatch.setattr(db_session, "commit", _raise_commit_error)

        with pytest.raises(RuntimeError, match="Simulated collection commit failure"):
            collection_service.delete_collection(db=db_session, owner_id=test_user.id, collection_id=col.id)

        db_session.rollback()
        assert db_session.get(Collection, col.id) is not None
        refreshed_session = db_session.get(ChatSession, session.id)
        assert refreshed_session.is_archived is False
        assert refreshed_session.target_collection_id == col.id
