"""Pydantic validation schemas for Chat Sessions, Messages, Citations, and Turns."""
from datetime import datetime
from typing import List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models.chat import ChatScopeType, SenderType


class MessageCitationResponse(BaseModel):
    """Immutable evidence snapshot attached to an assistant message."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Unique citation identifier")
    message_id: UUID = Field(..., description="Parent message identifier")
    chunk_id: Optional[int] = Field(
        default=None,
        description="64-bit integer chunk identifier (set to None if source chunk was deleted)",
    )
    quoted_text: str = Field(..., min_length=1, description="Passage text from the cited evidence chunk")
    page_number: int = Field(..., ge=1, description="Physical page number of source PDF")
    document_title: str = Field(..., min_length=1, description="Title of the source document at generation time")
    similarity_score: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Vector search similarity score (-1.0 to 1.0)",
    )
    created_at: datetime = Field(..., description="Timestamp when citation snapshot was recorded")


class ChatMessageResponse(BaseModel):
    """Dialogue turn message within a chat session."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Unique message identifier")
    chat_session_id: UUID = Field(..., description="Parent chat session identifier")
    sender_type: SenderType = Field(..., description="Author role (USER, ASSISTANT, SYSTEM)")
    content: str = Field(..., description="Message text content")
    token_count: Optional[int] = Field(default=None, ge=0, description="Token count if recorded")
    created_at: datetime = Field(..., description="Timestamp when message was created")
    citations: List[MessageCitationResponse] = Field(
        default_factory=list,
        description="Evidence citations attached to this message",
    )


class ChatMessageCreateRequest(BaseModel):
    """Payload to post a new user message within an existing chat session."""
    model_config = ConfigDict(extra="forbid")

    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User message content or research query (max 10,000 characters)",
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of context chunks to retrieve (1 to 50)",
    )
    similarity_threshold: Optional[float] = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Minimum similarity score threshold (-1.0 to 1.0)",
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content_trimmed(cls, v: object) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Message content must contain non-whitespace characters.")
        trimmed = v.strip()
        if len(trimmed) > 10000:
            raise ValueError("Message content cannot exceed 10,000 characters.")
        return trimmed


class DocumentScopeSummary(BaseModel):
    """Brief metadata summary of a document attached to a multi-document chat session."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Document identifier")
    filename: str = Field(..., description="Source filename")
    title: Optional[str] = Field(default=None, description="Document title if set")
    page_count: Optional[int] = Field(default=None, description="Page count of document")
    processing_status: str = Field(..., description="Processing status of document")


class ChatSessionCreateRequest(BaseModel):
    """Request payload to create a new persistent conversation thread."""
    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Optional title for the conversation (default: 'New Conversation')",
    )
    scope_type: ChatScopeType = Field(
        default=ChatScopeType.GLOBAL,
        description="Corpus scope for the conversation (GLOBAL, DOCUMENT, COLLECTION, or DOCUMENTS)",
    )
    target_document_id: Optional[UUID] = Field(
        default=None,
        description="Target Document ID when scope_type is DOCUMENT",
    )
    target_collection_id: Optional[UUID] = Field(
        default=None,
        description="Target Collection ID when scope_type is COLLECTION",
    )
    document_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Target Document IDs when scope_type is DOCUMENTS (1 to 50 unique IDs)",
    )

    @field_validator("title", mode="before")
    @classmethod
    def validate_title(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Title must contain non-whitespace characters if provided.")
        trimmed = v.strip()
        if len(trimmed) > 255:
            raise ValueError("Title cannot exceed 255 characters.")
        return trimmed

    @model_validator(mode="after")
    def validate_scope_targets(self) -> "ChatSessionCreateRequest":
        if self.scope_type == ChatScopeType.GLOBAL:
            if self.target_document_id is not None or self.target_collection_id is not None:
                raise ValueError("GLOBAL conversations must not specify target_document_id or target_collection_id.")
            if self.document_ids is not None:
                raise ValueError("GLOBAL conversations cannot specify document_ids.")
        elif self.scope_type == ChatScopeType.DOCUMENT:
            if self.target_collection_id is not None:
                raise ValueError("DOCUMENT conversations cannot specify target_collection_id.")
            if self.target_document_id is None:
                raise ValueError("DOCUMENT conversations require target_document_id.")
            if self.document_ids is not None:
                raise ValueError("DOCUMENT conversations cannot specify document_ids.")
        elif self.scope_type == ChatScopeType.COLLECTION:
            if self.target_document_id is not None:
                raise ValueError("COLLECTION conversations cannot specify target_document_id.")
            if self.target_collection_id is None:
                raise ValueError("COLLECTION conversations require target_collection_id.")
            if self.document_ids is not None:
                raise ValueError("COLLECTION conversations cannot specify document_ids.")
        elif self.scope_type == ChatScopeType.DOCUMENTS:
            if self.target_document_id is not None or self.target_collection_id is not None:
                raise ValueError("DOCUMENTS conversations must not specify target_document_id or target_collection_id.")
            if not self.document_ids:
                raise ValueError("DOCUMENTS conversations require at least one document_id in document_ids.")
            if len(self.document_ids) > 50:
                raise ValueError("DOCUMENTS conversations cannot exceed 50 selected documents.")
            if len(set(self.document_ids)) != len(self.document_ids):
                raise ValueError("DOCUMENTS conversations cannot contain duplicate document IDs.")
        return self


class ChatSessionUpdateRequest(BaseModel):
    """Request payload to update an existing chat session's title or archive state."""
    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Updated conversation title",
    )
    is_archived: Optional[bool] = Field(
        default=None,
        description="Set to true to archive the conversation thread",
    )

    @field_validator("title", mode="before")
    @classmethod
    def validate_title(cls, v: object) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Title must contain non-whitespace characters if provided.")
        trimmed = v.strip()
        if len(trimmed) > 255:
            raise ValueError("Title cannot exceed 255 characters.")
        return trimmed

    @model_validator(mode="after")
    def validate_at_least_one_field(self) -> "ChatSessionUpdateRequest":
        if self.title is None and self.is_archived is None:
            raise ValueError("At least one field (title or is_archived) must be provided for update.")
        return self


class ChatSessionResponse(BaseModel):
    """Detailed view of a chat session thread."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Unique chat session identifier")
    title: str = Field(..., description="Conversation title")
    scope_type: ChatScopeType = Field(..., description="Conversation corpus scope")
    target_document_id: Optional[UUID] = Field(
        default=None,
        description="Referenced document ID when scoped to DOCUMENT",
    )
    target_collection_id: Optional[UUID] = Field(
        default=None,
        description="Referenced collection ID when scoped to COLLECTION",
    )
    is_archived: bool = Field(..., description="Whether the conversation is archived")
    created_at: datetime = Field(..., description="Timestamp when conversation was created")
    updated_at: datetime = Field(..., description="Timestamp when conversation was last updated")
    message_count: int = Field(default=0, ge=0, description="Total number of messages in this session")
    selected_documents: List[DocumentScopeSummary] = Field(
        default_factory=list,
        description="List of selected documents when scoped to DOCUMENTS",
    )


class ChatSessionListResponse(BaseModel):
    """Paginated list of chat sessions."""
    model_config = ConfigDict(from_attributes=True)

    items: List[ChatSessionResponse] = Field(default_factory=list, description="Page of chat sessions")
    total: int = Field(..., ge=0, description="Total number of matching chat sessions")
    limit: int = Field(..., ge=1, description="Page limit applied")
    offset: int = Field(..., ge=0, description="Page offset applied")


class ChatMessageListResponse(BaseModel):
    """Chronologically ordered list of messages in a conversation thread."""
    model_config = ConfigDict(from_attributes=True)

    session_id: UUID = Field(..., description="Parent session identifier")
    items: List[ChatMessageResponse] = Field(
        default_factory=list,
        description="List of messages in chronological order",
    )
    total: int = Field(..., ge=0, description="Total number of messages in the session")


class ChatTurnResponse(BaseModel):
    """Response representing a completed dialogue turn (persisted user message + grounded assistant response)."""
    model_config = ConfigDict(from_attributes=True)

    session_id: UUID = Field(..., description="Chat session identifier")
    user_message: ChatMessageResponse = Field(..., description="Persisted user message")
    assistant_message: ChatMessageResponse = Field(..., description="Persisted assistant response with citations")
    retrieved_chunk_count: int = Field(
        default=0,
        ge=0,
        description="Number of authorized source passages retrieved for grounding",
    )
    model_name: str = Field(..., description="Model name that generated the completion")
    provider_name: str = Field(..., description="LLM provider name that generated the completion")
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    total_tokens: Optional[int] = Field(default=None, ge=0)
