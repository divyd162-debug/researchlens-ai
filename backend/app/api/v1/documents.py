"""API router for Document management endpoints."""
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func, or_, update
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.core.database import create_session_factory
from app.models.document import Document, ProcessingStatus
from app.models.user import User
from app.schemas.collection import MessageResponse
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.services.document_service import (
    DocumentNotFoundError,
    DocumentServiceError,
    DocumentSizeLimitExceededError,
    DuplicateDocumentError,
    InvalidDocumentError,
    document_service,
)
from app.services.ingestion_service import ingestion_service

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a new document",
    description="Uploads and validates a research PDF for the authenticated user, triggering async ingestion.",
)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF document file to upload"),
    title: Optional[str] = Form(None, description="Optional custom document title"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """Uploads, validates, and registers a research PDF owned by the current authenticated user."""
    try:
        doc_response = document_service.upload_document(
            db=db,
            owner_id=current_user.id,
            file=file,
            title=title,
        )
    except InvalidDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except DocumentSizeLimitExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except DuplicateDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except DocumentServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store uploaded document file.",
        ) from exc

    # Queue background ingestion task using engine-bound factory rather than live request session
    engine = db.get_bind()
    session_factory = create_session_factory(engine)
    background_tasks.add_task(
        ingestion_service.process_document,
        document_id=doc_response.id,
        session_factory=session_factory,
    )

    return doc_response


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List documents",
    description="Returns a paginated list of documents owned by the authenticated caller.",
)
def list_documents(
    limit: int = Query(default=50, ge=1, le=100, description="Page limit"),
    offset: int = Query(default=0, ge=0, description="Page offset"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """Lists documents owned exclusively by the authenticated caller."""
    return document_service.list_documents(
        db=db,
        owner_id=current_user.id,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{id}",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Get document by ID",
    description="Retrieves metadata for a document owned by the caller.",
)
def get_document(
    id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """Retrieves a single document by ID, strictly checking ownership."""
    try:
        return document_service.get_document(
            db=db,
            owner_id=current_user.id,
            document_id=id,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.delete(
    "/{id}",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete document",
    description="Deletes a document owned by the caller, including its stored file.",
)
def delete_document(
    id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Deletes a document owned by the authenticated caller."""
    try:
        document_service.delete_document(
            db=db,
            owner_id=current_user.id,
            document_id=id,
        )
        return MessageResponse(message="Document deleted successfully")
    except DocumentNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.post(
    "/{id}/retry",
    response_model=DocumentResponse,
    status_code=status.HTTP_200_OK,
    summary="Retry document ingestion",
    description="Retries the ingestion pipeline for a document owned by the caller.",
)
def retry_document_ingestion(
    id: UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """Initiates an idempotent retry of the ingestion pipeline for a document owned by the caller."""
    doc = (
        db.query(Document)
        .filter(Document.owner_id == current_user.id, Document.id == id)
        .first()
    )
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    # If document is actively in-flight, avoid interrupting it or queuing duplicate tasks
    in_flight_statuses = {
        ProcessingStatus.EXTRACTING.value,
        ProcessingStatus.CHUNKING.value,
        ProcessingStatus.EMBEDDING.value,
    }

    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(seconds=settings.STALE_INGESTION_TIMEOUT_SECONDS)

    # Determine if the document is actively being processed by a running worker.
    # A document in an in-flight status is active ONLY IF its updated_at timestamp
    # is fresher than the configured STALE_INGESTION_TIMEOUT_SECONDS.
    is_actively_in_flight = False
    if doc.processing_status in in_flight_statuses:
        doc_updated = doc.updated_at
        if doc_updated is not None:
            if doc_updated.tzinfo is None:
                doc_updated = doc_updated.replace(tzinfo=timezone.utc)
            if (now - doc_updated).total_seconds() < settings.STALE_INGESTION_TIMEOUT_SECONDS:
                is_actively_in_flight = True

    if not is_actively_in_flight:
        db_cutoff = stale_cutoff
        if doc.updated_at is not None and doc.updated_at.tzinfo is None:
            db_cutoff = db_cutoff.replace(tzinfo=None)

        # Atomically reset status to PENDING only if the document is still eligible
        # (not actively claimed/in-flight by another worker).
        reset_stmt = (
            update(Document)
            .where(
                Document.id == doc.id,
                Document.owner_id == current_user.id,
                or_(
                    Document.processing_status.notin_(in_flight_statuses),
                    Document.updated_at <= db_cutoff,
                    Document.updated_at.is_(None),
                ),
            )
            .values(
                processing_status=ProcessingStatus.PENDING.value,
                lease_token=None,
                error_message=None,
                updated_at=func.now(),
            )
            .execution_options(synchronize_session=False)
        )
        result = db.execute(reset_stmt)
        if result.rowcount > 0:
            db.commit()
            db.refresh(doc)

            engine = db.get_bind()
            session_factory = create_session_factory(engine)
            background_tasks.add_task(
                ingestion_service.process_document,
                document_id=doc.id,
                session_factory=session_factory,
            )
        else:
            db.rollback()
            db.refresh(doc)

    return DocumentResponse.model_validate(doc)

