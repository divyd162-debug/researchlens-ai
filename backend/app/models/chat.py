"""ChatSession, Message, and MessageCitation ORM models for ResearchLens AI."""
import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import UUID

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.chunk import Chunk
    from app.models.collection import Collection
    from app.models.document import Document
    from app.models.user import User


class ChatScopeType(str, enum.Enum):
    """Scope defining the document corpus searchable within a conversation."""
    GLOBAL = "GLOBAL"
    COLLECTION = "COLLECTION"
    DOCUMENT = "DOCUMENT"
    DOCUMENTS = "DOCUMENTS"


class SenderType(str, enum.Enum):
    """Author of an individual message in a chat session."""
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"


class ChatSession(Base):
    """Interactive conversation thread bounded by a query scope.

    Security & Scope Architecture:
    - owner_id isolates conversations per user.
    - Scope constraint rules:
      * GLOBAL: target_document_id and target_collection_id must be NULL.
      * DOCUMENT: target_collection_id must be NULL. An active session requires
        target_document_id IS NOT NULL. When target document is deleted, the session
        is archived (is_archived=True) and target_document_id may be cleared.
      * COLLECTION: target_document_id must be NULL. An active session requires
        target_collection_id IS NOT NULL. When target collection is deleted, the session
        is archived (is_archived=True) and target_collection_id may be cleared.
      * An archived scoped session NEVER silently becomes a GLOBAL chat because scope_type
        remains unchanged.
    """
    __tablename__ = "chat_sessions"
    __table_args__ = (
        CheckConstraint(
            "scope_type IN ('GLOBAL', 'COLLECTION', 'DOCUMENT', 'DOCUMENTS')",
            name="ck_chat_session_scope_type",
        ),
        CheckConstraint(
            "(scope_type = 'GLOBAL' AND target_document_id IS NULL AND target_collection_id IS NULL) OR "
            "(scope_type = 'DOCUMENT' AND target_collection_id IS NULL AND (target_document_id IS NOT NULL OR is_archived)) OR "
            "(scope_type = 'COLLECTION' AND target_document_id IS NULL AND (target_collection_id IS NOT NULL OR is_archived)) OR "
            "(scope_type = 'DOCUMENTS' AND target_document_id IS NULL AND target_collection_id IS NULL)",
            name="ck_chat_session_scope_targets",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        default="New Conversation",
    )
    scope_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default=ChatScopeType.GLOBAL.value,
    )
    target_document_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    target_collection_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    is_archived: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    owner: Mapped["User"] = relationship(
        "User",
        back_populates="chat_sessions",
    )
    target_document: Mapped[Optional["Document"]] = relationship(
        "Document",
        back_populates="chat_sessions",
    )
    target_collection: Mapped[Optional["Collection"]] = relationship(
        "Collection",
        back_populates="chat_sessions",
    )
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="chat_session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )
    selected_documents: Mapped[List["Document"]] = relationship(
        "Document",
        secondary="chat_session_documents",
        back_populates="multi_scoped_chat_sessions",
        overlaps="chat_session_documents,document_chat_sessions",
    )
    chat_session_documents: Mapped[List["ChatSessionDocument"]] = relationship(
        "ChatSessionDocument",
        back_populates="chat_session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        overlaps="selected_documents,multi_scoped_chat_sessions",
    )

    def archive(self) -> None:
        """Archives the chat session safely, clearing any targets."""
        self.is_archived = True
        self.target_document_id = None
        self.target_collection_id = None

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id} title={self.title!r} scope={self.scope_type} archived={self.is_archived}>"


class Message(Base):
    """Single message dialogue turn in a ChatSession."""
    __tablename__ = "messages"
    __table_args__ = (
        CheckConstraint(
            "sender_type IN ('USER', 'ASSISTANT', 'SYSTEM')",
            name="ck_message_sender_type",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    chat_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sender_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    token_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    chat_session: Mapped["ChatSession"] = relationship(
        "ChatSession",
        back_populates="messages",
    )
    citations: Mapped[List["MessageCitation"]] = relationship(
        "MessageCitation",
        back_populates="message",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Message id={self.id} session={self.chat_session_id} sender={self.sender_type}>"


class MessageCitation(Base):
    """Citation snapshot linking assistant response passages to evidence.

    Citation Snapshot Pattern:
    - chunk_id: Nullable foreign key to Chunk (ondelete='SET NULL').
    - quoted_text, page_number, and document_title store an immutable historical snapshot.
    - If the chunk or document is deleted or re-chunked, chunk_id transitions to NULL,
      while the snapshot fields remain intact for permanent auditability.
    """
    __tablename__ = "message_citations"
    __table_args__ = (
        CheckConstraint("page_number >= 1", name="ck_citation_page_number_positive"),
        CheckConstraint(
            "similarity_score IS NULL OR (similarity_score >= -1.0 AND similarity_score <= 1.0)",
            name="ck_citation_similarity_score_range",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_id: Mapped[Optional[int]] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey("chunks.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    quoted_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    document_title: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )
    similarity_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    message: Mapped["Message"] = relationship(
        "Message",
        back_populates="citations",
    )
    chunk: Mapped[Optional["Chunk"]] = relationship(
        "Chunk",
        back_populates="citations",
    )

    def __repr__(self) -> str:
        return f"<MessageCitation id={self.id} msg={self.message_id} chunk={self.chunk_id} page={self.page_number}>"


class ChatSessionDocument(Base):
    """Association table linking chat sessions with multiple scoped documents."""
    __tablename__ = "chat_session_documents"

    chat_session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    chat_session: Mapped["ChatSession"] = relationship(
        "ChatSession",
        back_populates="chat_session_documents",
        overlaps="selected_documents,multi_scoped_chat_sessions",
    )
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="document_chat_sessions",
        overlaps="selected_documents,multi_scoped_chat_sessions",
    )

    def __repr__(self) -> str:
        return f"<ChatSessionDocument session={self.chat_session_id} doc={self.document_id}>"
