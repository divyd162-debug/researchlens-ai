"""Pydantic schemas for Collection management."""
from datetime import datetime
from typing import Any, List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator



class CollectionCreateRequest(BaseModel):
    """Payload for creating a new collection."""
    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Unique collection name for the owner",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="Optional description of the collection's research scope",
    )

    @field_validator("name")
    @classmethod
    def validate_and_normalize_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Collection name cannot be empty or whitespace-only.")
        stripped = v.strip()
        if len(stripped) > 255:
            raise ValueError("Collection name cannot exceed 255 characters.")
        return stripped

    @field_validator("description")
    @classmethod
    def normalize_description(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None


class CollectionUpdateRequest(BaseModel):
    """Payload for updating an existing collection."""
    model_config = ConfigDict(frozen=True)

    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="New collection name",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=5000,
        description="New description",
    )

    @model_validator(mode="before")
    @classmethod
    def check_name_not_null(cls, data: Any) -> Any:
        """Ensures that omitted name is allowed, but an explicit null is rejected."""
        if isinstance(data, dict) and "name" in data and data["name"] is None:
            raise ValueError("Collection name cannot be null.")
        return data


    @field_validator("name")
    @classmethod
    def validate_and_normalize_name(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not v or not v.strip():
            raise ValueError("Collection name cannot be empty or whitespace-only.")
        stripped = v.strip()
        if len(stripped) > 255:
            raise ValueError("Collection name cannot exceed 255 characters.")
        return stripped

    @field_validator("description")
    @classmethod
    def normalize_description(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        stripped = v.strip()
        return stripped if stripped else None


class CollectionResponse(BaseModel):
    """Public representation of a collection, excluding owner_id and internal details."""
    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: UUID = Field(..., description="Unique collection identifier")
    name: str = Field(..., description="Collection name")
    description: Optional[str] = Field(default=None, description="Collection description")
    document_count: int = Field(default=0, ge=0, description="Total documents in this collection")
    created_at: datetime = Field(..., description="Timestamp of collection creation")
    updated_at: datetime = Field(..., description="Timestamp of last update")


class CollectionListResponse(BaseModel):
    """Paginated list of collections for the authenticated caller."""
    model_config = ConfigDict(frozen=True)

    items: List[CollectionResponse] = Field(default_factory=list, description="Collections in the current page")
    total: int = Field(..., ge=0, description="Total count of collections owned by the caller")
    limit: int = Field(..., gt=0, description="Pagination page size limit")
    offset: int = Field(..., ge=0, description="Pagination offset")


class MessageResponse(BaseModel):
    """Standard acknowledgment response for action-oriented endpoints."""
    model_config = ConfigDict(frozen=True)

    message: str = Field(..., description="Status or result message")


class AddDocumentsToCollectionRequest(BaseModel):
    """Payload for associating one or more documents with a collection."""
    model_config = ConfigDict(frozen=True)

    document_ids: List[UUID] = Field(
        default_factory=list,
        description="One or more document IDs to associate with the collection",
    )
    document_id: Optional[UUID] = Field(
        default=None,
        description="Single document ID (convenience alternative to document_ids)",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_document_ids(cls, data: Any) -> Any:
        if isinstance(data, dict):
            doc_id = data.get("document_id")
            doc_ids = data.get("document_ids")
            if doc_id and not doc_ids:
                data = dict(data)
                data["document_ids"] = [doc_id]
        return data

    @field_validator("document_ids")
    @classmethod
    def validate_non_empty_ids(cls, v: List[UUID]) -> List[UUID]:
        if not v:
            raise ValueError("At least one document ID must be provided.")
        return v
