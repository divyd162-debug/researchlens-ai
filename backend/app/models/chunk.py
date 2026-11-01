"""Chunk ORM model representing page-bounded extracted passages for vector search."""
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import UUID

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.chat import MessageCitation
    from app.models.document import Document


class Chunk(Base):
    """Discrete text passage extracted from a PDF page by ChunkingService.

    Architecture & Vector Search Notes:
    - id: Autoincrementing 64-bit integer (BIGINT in PostgreSQL, INTEGER in SQLite).
      This integer ID directly maps 1:1 to FAISS IndexIDMap2 vector IDs.
    - Chunks are page-bounded: citations in academic literature reference specific physical pages.
    - char_start and char_end record the exact string slice offsets within the page text.
    - embedding_model_name identifies the exact model used for vector encoding.
    """
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_document_index"),
        CheckConstraint("char_end >= char_start", name="ck_chunk_char_offsets"),
        CheckConstraint("char_start >= 0", name="ck_chunk_char_start_non_negative"),
        CheckConstraint("page_number >= 1", name="ck_chunk_page_number_positive"),
        CheckConstraint("token_count >= 0", name="ck_chunk_token_count_non_negative"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    page_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    char_start: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    char_end: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    token_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    embedding_model_name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    # Relationships
    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )
    citations: Mapped[List["MessageCitation"]] = relationship(
        "MessageCitation",
        back_populates="chunk",
    )

    @property
    def char_count(self) -> int:
        """Returns character length of chunk content."""
        return len(self.content)

    def __repr__(self) -> str:
        return f"<Chunk id={self.id} doc={self.document_id} page={self.page_number} idx={self.chunk_index}>"
