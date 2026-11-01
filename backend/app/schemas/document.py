"""Pydantic validation schemas for Document management."""
import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DocumentResponse(BaseModel):
    """Safe public representation of a Document.
    
    Excludes server-internal filesystem paths and sensitive storage details.
    """
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original sanitized filename")
    title: Optional[str] = Field(None, description="Optional document title")
    file_size_bytes: int = Field(..., description="Size of file in bytes")
    page_count: Optional[int] = Field(None, description="Total number of pages")
    processing_status: str = Field(..., description="Current processing status")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""
    items: List[DocumentResponse] = Field(..., description="List of documents on current page")
    total: int = Field(..., description="Total number of documents matching query")
    limit: int = Field(..., description="Maximum items per page")
    offset: int = Field(..., description="Number of items skipped")
