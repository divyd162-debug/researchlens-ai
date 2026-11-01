"""API router for Semantic Search endpoints."""
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user, get_db
from app.models.user import User
from app.schemas.search import (
    SearchPassageResponse,
    SearchRequest,
    SearchResponse,
)
from app.services.retrieval_service import (
    DocumentNotReadyError,
    RetrievalError,
    RetrievalService,
)
from app.services.vector_store import VectorStoreNotReadyError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["Search"])


def get_retrieval_service() -> RetrievalService:
    """Dependency provider returning an instance of RetrievalService."""
    return RetrievalService()


@router.post(
    "",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic vector search across research documents",
    description=(
        "Executes dense vector search across research papers with multi-tenant PostgreSQL "
        "access verification and corpus scoping (GLOBAL, DOCUMENT, or COLLECTION)."
    ),
)
def search_passages(
    request: SearchRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> SearchResponse:
    """Performs semantic search over documents owned by the active authenticated user."""
    try:
        retrieved_chunks = retrieval_service.retrieve_relevant_chunks(
            db=db,
            query=request.query,
            user_id=current_user.id,
            scope_type=request.scope_type,
            scope_id=request.scope_id,
            document_ids=request.document_ids,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
        )
    except VectorStoreNotReadyError as exc:
        logger.warning("Vector store is not ready during search query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector search service is currently unavailable. Please try again later.",
        ) from exc
    except DocumentNotReadyError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        err_msg = str(exc)
        if "not found or not owned" in err_msg:
            # Multi-tenant security isolation: Return 404 to avoid leaking existence of documents/collections
            if "Document" in err_msg:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Target document not found.",
                ) from exc
            elif "Collection" in err_msg:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Target collection not found.",
                ) from exc
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Target scope entity not found.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
        ) from exc
    except RetrievalError as exc:
        logger.error("Unexpected retrieval failure during search: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve search results.",
        ) from exc

    passages = [
        SearchPassageResponse(
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_title=c.document_title,
            page_number=c.page_number,
            content=c.content,
            similarity_score=c.similarity_score,
            char_start=c.char_start,
            char_end=c.char_end,
        )
        for c in retrieved_chunks
    ]

    return SearchResponse(
        query=request.query,
        scope_type=request.scope_type,
        scope_id=request.scope_id,
        document_ids=request.document_ids,
        total_results=len(passages),
        result_count=len(passages),
        results=passages,
    )
