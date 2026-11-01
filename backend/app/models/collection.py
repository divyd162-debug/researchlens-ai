"""Collection and CollectionDocument association models for ResearchLens AI."""
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional
from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import UUID

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.chat import ChatSession
    from app.models.document import Document
    from app.models.user import User


class Collection(Base):
    """User-curated collection of research documents for multi-document synthesis.

    Security & Architecture Note:
    - owner_id isolates collections per user.
    - UniqueConstraint("owner_id", "name") prevents duplicate collection names per user.
    - Application services MUST verify that all documents added to this collection
      have matching owner_id.
    """
    __tablename__ = "collections"
    __table_args__ = (
        UniqueConstraint("owner_id", "name", name="uq_collection_owner_name"),
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
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
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
        back_populates="collections",
    )
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        secondary="collection_documents",
        back_populates="collections",
        overlaps="collection_documents,document_collections",
    )
    collection_documents: Mapped[List["CollectionDocument"]] = relationship(
        "CollectionDocument",
        back_populates="collection",
        cascade="all, delete-orphan",
        passive_deletes=True,
        overlaps="documents,collections",
    )
    chat_sessions: Mapped[List["ChatSession"]] = relationship(
        "ChatSession",
        back_populates="target_collection",
    )

    def __repr__(self) -> str:
        return f"<Collection id={self.id} name={self.name!r} owner_id={self.owner_id}>"


class CollectionDocument(Base):
    """Explicit association model between Collection and Document with audit metadata."""
    __tablename__ = "collection_documents"

    collection_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("collections.id", ondelete="CASCADE"),
        primary_key=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    collection: Mapped["Collection"] = relationship(
        "Collection",
        back_populates="collection_documents",
        overlaps="collections,documents",
    )
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="document_collections",
        overlaps="collections,documents",
    )

    def __repr__(self) -> str:
        return f"<CollectionDocument collection_id={self.collection_id} document_id={self.document_id}>"
