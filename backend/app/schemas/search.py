"""Pydantic validation schemas for Semantic Search endpoints."""
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.chat import ChatScopeType


class SearchPassageResponse(BaseModel):
    """Ranked passage result returned by semantic search."""
    model_config = ConfigDict(frozen=True, from_attributes=True)

    chunk_id: int = Field(..., gt=0, description="Positive 64-bit integer ID of the chunk")
    document_id: UUID = Field(..., description="UUID of the parent document")
    document_title: str = Field(..., min_length=1, description="Title of the source document")
    page_number: int = Field(..., ge=1, description="Physical page number in source PDF")
    content: str = Field(..., min_length=1, description="Passage text from the chunk")
    similarity_score: float = Field(..., description="Cosine similarity score from vector search")
    char_start: int = Field(..., ge=0, description="0-based start character offset within physical PDF page text")
    char_end: int = Field(..., ge=0, description="0-based end character offset within physical PDF page text")

    @field_validator("char_end")
    @classmethod
    def validate_char_offsets(cls, v: int, info) -> int:
        char_start = info.data.get("char_start")
        if char_start is not None and v < char_start:
            raise ValueError(f"char_end ({v}) must be greater than or equal to char_start ({char_start})")
        return v


class SearchRequest(BaseModel):
    """Input payload for a semantic search query."""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query string (maximum 1,000 characters)",
    )
    scope_type: ChatScopeType = Field(
        default=ChatScopeType.GLOBAL,
        description="Search scope (GLOBAL, DOCUMENT, COLLECTION, or DOCUMENTS)",
    )
    scope_id: Optional[UUID] = Field(
        default=None,
        description="Document ID or Collection ID when scoped",
    )
    document_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Target Document IDs when scope_type is DOCUMENTS (1 to 50 unique IDs)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of ranked passages to return (1 to 50)",
    )
    similarity_threshold: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Minimum similarity score threshold (-1.0 to 1.0)",
    )

    @field_validator("query")
    @classmethod
    def validate_query_trimmed_and_length(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query must contain non-whitespace characters.")
        trimmed = v.strip()
        if len(trimmed) > 1000:
            raise ValueError("Query cannot exceed 1000 characters.")
        return trimmed

    @model_validator(mode="after")
    def validate_scope(self) -> "SearchRequest":
        if self.scope_type == ChatScopeType.GLOBAL:
            if self.scope_id is not None:
                raise ValueError("scope_id must be None when scope_type is GLOBAL.")
            if self.document_ids is not None:
                raise ValueError("document_ids must be None when scope_type is GLOBAL.")
        elif self.scope_type in (ChatScopeType.DOCUMENT, ChatScopeType.COLLECTION):
            if self.scope_id is None:
                raise ValueError(f"scope_id is required when scope_type is {self.scope_type.value}.")
            if self.document_ids is not None:
                raise ValueError(f"document_ids cannot be specified when scope_type is {self.scope_type.value}.")
        elif self.scope_type == ChatScopeType.DOCUMENTS:
            if self.scope_id is not None:
                raise ValueError("scope_id must be None when scope_type is DOCUMENTS.")
            if not self.document_ids:
                raise ValueError("document_ids is required and cannot be empty when scope_type is DOCUMENTS.")
            if len(self.document_ids) > 50:
                raise ValueError("document_ids cannot exceed 50 documents.")
            if len(set(self.document_ids)) != len(self.document_ids):
                raise ValueError("document_ids cannot contain duplicate IDs.")
        return self


class SearchResponse(BaseModel):
    """Response payload for a semantic search query."""
    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Executed search query text")
    scope_type: ChatScopeType = Field(..., description="Applied search scope type")
    scope_id: Optional[UUID] = Field(default=None, description="Target scope UUID if scoped")
    document_ids: Optional[List[UUID]] = Field(default=None, description="Target document UUIDs if scoped to DOCUMENTS")
    total_results: int = Field(..., ge=0, description="Total number of ranked passages returned")
    result_count: int = Field(..., ge=0, description="Count of retrieved matching passages")
    results: List[SearchPassageResponse] = Field(..., description="Ranked list of matching passages")
