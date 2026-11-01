"""User ORM model representing authenticated accounts in ResearchLens AI."""
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List
from sqlalchemy import Boolean, DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates
from sqlalchemy.types import UUID

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.chat import ChatSession
    from app.models.collection import Collection
    from app.models.document import Document


class User(Base):
    """Represents a registered user identity and root tenant owner.

    Security & Architecture Note:
    All tenant data (documents, collections, chats) is owned by a User.
    Foreign keys provide relational integrity, NOT application authorization.
    Cross-user operations must be strictly verified in application services.
    """
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    email: Mapped[str] = mapped_column(
        String(320),
        unique=True,
        index=True,
        nullable=False,
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
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
    documents: Mapped[List["Document"]] = relationship(
        "Document",
        back_populates="owner",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    collections: Mapped[List["Collection"]] = relationship(
        "Collection",
        back_populates="owner",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    chat_sessions: Mapped[List["ChatSession"]] = relationship(
        "ChatSession",
        back_populates="owner",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    @validates("email")
    def validate_and_normalize_email(self, key: str, value: str) -> str:
        """Normalizes email to trimmed lowercase and validates basic format."""
        if not value or not isinstance(value, str):
            raise ValueError("Email must be a non-empty string")
        normalized = value.strip().lower()
        if not normalized or "@" not in normalized:
            raise ValueError(f"Invalid email address: {value}")
        return normalized

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} is_active={self.is_active}>"
