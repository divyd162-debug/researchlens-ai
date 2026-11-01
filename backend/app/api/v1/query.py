"""API router for Grounded RAG Query endpoints."""
import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user, get_db
from app.models.user import User
from app.schemas.rag import QueryRequest, RAGResponse
from app.services.embedding_service import get_embedding_service
from app.services.llm.exceptions import (
    LLMContextLimitExceededError,
    LLMError,
    LLMProviderError,
    LLMQuotaExceededError,
)
from app.services.llm.factory import get_llm_provider
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStoreNotReadyError, get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


def get_rag_service() -> RAGService:
    """Dependency provider returning an instance of RAGService reusing singletons."""
    return RAGService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        llm_provider=get_llm_provider(),
    )


@router.post(
    "",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Grounded RAG query and answer generation",
    description=(
        "Executes dense vector search across research documents, authoritative PostgreSQL access "
        "verification, corpus scoping, and grounded generation with citation attribution."
    ),
)
def query_rag(
    request: QueryRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGResponse:
    """Performs grounded RAG question answering over documents owned by the active user."""
    try:
        return rag_service.answer_query(
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
        logger.warning("Vector store is not ready during RAG query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector search service is currently unavailable. Please try again later.",
        ) from exc
    except LLMQuotaExceededError as exc:
        logger.warning("LLM provider quota exceeded during RAG query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="LLM provider rate limit or quota exceeded. Please try again later.",
        ) from exc
    except LLMContextLimitExceededError as exc:
        logger.warning("LLM context limit exceeded during RAG query: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The query or retrieved context exceeds the model maximum context length.",
        ) from exc
    except LLMProviderError as exc:
        logger.error("Upstream LLM provider failure during RAG query: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream LLM provider failed to generate a response. Please try again later.",
        ) from exc
    except LLMError as exc:
        logger.error("LLM configuration or generic failure during RAG query: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM service encountered an unexpected error.",
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

        # Sanitize internal or sensitive details from arbitrary ValueError messages
        lower_msg = err_msg.lower()
        if (
            len(err_msg) > 200
            or "\n" in err_msg
            or any(term in lower_msg for term in ("user_id", "secret", "token", "password", "key", "internal", "database", "traceback"))
        ):
            logger.warning("Sanitized internal ValueError detail during RAG query: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request parameters.",
            ) from exc

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during RAG query: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred while processing the query.",
        ) from exc
