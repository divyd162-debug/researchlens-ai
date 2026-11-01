"""Document ORM model and ingestion status for ResearchLens AI."""
import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import UUID

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.chat import ChatSession, ChatSessionDocument
    from app.models.chunk import Chunk
    from app.models.collection import Collection, CollectionDocument
    from app.models.user import User


class ProcessingStatus(str, enum.Enum):
    """Lifecycle states of document processing pipeline."""
    PENDING = "PENDING"
    EXTRACTING = "EXTRACTING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    READY = "READY"
    FAILED = "FAILED"


class Document(Base):
    """Represents an uploaded research PDF and its ingestion status.

    Security & Architecture Note:
    - owner_id isolates documents per user.
    - (owner_id, file_hash_sha256) is constrained unique via uq_document_owner_hash to prevent
      concurrent duplicate uploads for the same user.
    - error_message contains only safe, sanitized summaries (no stack traces or secrets).
    """
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_owner_status", "owner_id", "processing_status"),
        Index("ix_documents_owner_hash", "owner_id", "file_hash_sha256"),
        UniqueConstraint("owner_id", "file_hash_sha256", name="uq_document_owner_hash"),
        CheckConstraint(
            "processing_status IN ('PENDING', 'EXTRACTING', 'CHUNKING', 'EMBEDDING', 'READY', 'FAILED')",
            name="ck_document_processing_status",
        ),
        CheckConstraint("file_size_bytes >= 0", name="ck_document_file_size_non_negative"),
        CheckConstraint("page_count IS NULL OR page_count >= 0", name="ck_document_page_count_non_negative"),
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
    filename: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )
    title: Mapped[Optional[str]] = mapped_column(
        String(512),
        nullable=True,
    )
    file_path: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
    )
    file_hash_sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )
    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    page_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    processing_status: Mapped[str] = mapped_column(
        String(32),
        default=ProcessingStatus.PENDING.value,
        nullable=False,
        index=True,
    )
    error_message: Mapped[Optional[str]] = mapped_column(
        String(1024),
        nullable=True,
    )
    lease_token: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
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
        back_populates="documents",
    )
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Chunk.chunk_index",
    )
    collections: Mapped[List["Collection"]] = relationship(
        "Collection",
        secondary="collection_documents",
        back_populates="documents",
        overlaps="collection_documents,document_collections",
    )
    document_collections: Mapped[List["CollectionDocument"]] = relationship(
        "CollectionDocument",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        overlaps="collections,documents",
    )
    chat_sessions: Mapped[List["ChatSession"]] = relationship(
        "ChatSession",
        back_populates="target_document",
    )
    multi_scoped_chat_sessions: Mapped[List["ChatSession"]] = relationship(
        "ChatSession",
        secondary="chat_session_documents",
        back_populates="selected_documents",
        overlaps="chat_session_documents,document_chat_sessions",
    )
    document_chat_sessions: Mapped[List["ChatSessionDocument"]] = relationship(
        "ChatSessionDocument",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        overlaps="selected_documents,multi_scoped_chat_sessions",
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename!r} status={self.processing_status}>"
