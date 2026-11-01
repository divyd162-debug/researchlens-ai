"""Pydantic schemas for RAG query inputs, retrieved context chunks, and grounded responses."""
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.chat import ChatScopeType


class RetrievedContextChunk(BaseModel):
    """Passage context retrieved from vector search and verified in PostgreSQL."""
    model_config = ConfigDict(frozen=True)

    chunk_id: int = Field(..., gt=0, description="Positive 64-bit integer ID of the chunk")
    document_id: UUID = Field(..., description="UUID of the parent document")
    content: str = Field(..., min_length=1, description="Passage text from the chunk")
    page_number: int = Field(..., ge=1, description="Physical page number in source PDF")
    document_title: str = Field(..., min_length=1, description="Title of the source document")
    similarity_score: float = Field(..., description="Cosine similarity score from FAISS search")
    char_start: int = Field(..., ge=0, description="0-based start character offset within physical PDF page text")
    char_end: int = Field(..., ge=0, description="0-based end character offset within physical PDF page text")

    @field_validator("char_end")
    @classmethod
    def validate_char_offsets(cls, v: int, info) -> int:
        char_start = info.data.get("char_start")
        if char_start is not None and v < char_start:
            raise ValueError(f"char_end ({v}) must be greater than or equal to char_start ({char_start})")
        return v


class CitationSnapshot(BaseModel):
    """Snapshot of a verified cited evidence passage.

    Integrity & Provenance Note:
    Captures immutable evidence fields (document_id, chunk_id, quoted_text, page_number,
    document_title, similarity_score, citation_frequency) at generation time.
    Citation verification confirms that the model emitted a valid marker corresponding
    to an authorized retrieved context passage; it does NOT prove factual truth or
    semantic entailment of the LLM's claims.
    """
    model_config = ConfigDict(frozen=True)

    chunk_id: int = Field(..., gt=0, description="64-bit integer ID of the cited chunk")
    document_id: UUID = Field(..., description="UUID of the cited document")
    quoted_text: str = Field(..., min_length=1, description="Passage text from the cited chunk")
    page_number: int = Field(..., ge=1, description="Physical page number of source PDF")
    document_title: str = Field(..., min_length=1, description="Title of the source document")
    similarity_score: float = Field(..., description="Vector search similarity score")
    citation_frequency: int = Field(default=1, ge=1, description="Number of times cited in the response")


class QueryRequest(BaseModel):
    """Client request payload for grounded RAG query and answer generation."""
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query or research question (maximum 1,000 characters)",
    )
    scope_type: ChatScopeType = Field(
        default=ChatScopeType.GLOBAL,
        description="Query corpus scope (GLOBAL, DOCUMENT, COLLECTION, or DOCUMENTS)",
    )
    scope_id: Optional[UUID] = Field(
        default=None,
        description="Target Document ID or Collection ID when scoped",
    )
    document_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Target Document IDs when scope_type is DOCUMENTS (1 to 50 unique IDs)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of context passages to retrieve (1 to 50)",
    )
    similarity_threshold: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Minimum cosine similarity score threshold (-1.0 to 1.0)",
    )

    @field_validator("query", mode="before")
    @classmethod
    def validate_query_trimmed_and_length(cls, v: object) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Query must contain non-whitespace characters.")
        trimmed = v.strip()
        if len(trimmed) > 1000:
            raise ValueError("Query cannot exceed 1000 characters.")
        return trimmed

    @model_validator(mode="after")
    def validate_scope(self) -> "QueryRequest":
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


class RAGQueryRequest(BaseModel):
    """Input payload for a RAG search and answer generation query."""
    query: str = Field(..., min_length=1, description="User query or question")
    user_id: UUID = Field(..., description="Authenticated user ID for tenant access control")
    scope_type: ChatScopeType = Field(default=ChatScopeType.GLOBAL, description="Query corpus scope")
    scope_id: Optional[UUID] = Field(default=None, description="Document ID or Collection ID if scoped")
    document_ids: Optional[List[UUID]] = Field(default=None, description="Document IDs if scoped to DOCUMENTS")
    top_k: int = Field(default=5, gt=0, le=50, description="Maximum number of context chunks to retrieve")
    similarity_threshold: float = Field(default=0.0, description="Minimum similarity score required")

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query must contain non-whitespace characters")
        return v.strip()


class RAGResponse(BaseModel):
    """Result of an end-to-end RAG query."""
    query: str = Field(..., description="Original user query text")
    answer: str = Field(..., description="Raw text generated by the LLM provider with citation markers")
    citations: List[CitationSnapshot] = Field(default_factory=list, description="Verified citation snapshots")
    unknown_chunk_ids: List[int] = Field(default_factory=list, description="Chunk IDs cited by LLM but not in retrieved context")
    malformed_markers: List[str] = Field(default_factory=list, description="Malformed marker-like strings found in answer")
    unsupported_numbers: List[str] = Field(default_factory=list, description="Decimal numbers in answer not found in cited evidence")
    status: str = Field(default="SUCCESS", description="Execution status: 'SUCCESS' or 'NO_RELEVANT_CONTEXT'")
    model_name: str = Field(..., description="Model name used for generation")
    provider_name: str = Field(..., description="Provider name used for generation")
    retrieved_chunk_count: int = Field(default=0, ge=0, description="Total authorized chunks retrieved and considered")
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
