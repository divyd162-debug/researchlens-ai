"""API router for Persistent Chat Sessions and Conversations."""
import logging
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_active_user, get_db
from app.api.v1.query import get_rag_service
from app.models.user import User
from app.schemas.chat import (
    ChatMessageCreateRequest,
    ChatMessageListResponse,
    ChatSessionCreateRequest,
    ChatSessionListResponse,
    ChatSessionResponse,
    ChatSessionUpdateRequest,
    ChatTurnResponse,
)
from app.schemas.collection import MessageResponse
from app.services.chat_service import (
    ChatService,
    ChatSessionArchivedError,
    ChatSessionNotFoundError,
    ChatSessionValidationError,
    TargetEntityNotFoundError,
    TargetEntityNotReadyError,
    chat_service,
)
from app.services.chat_concurrency import ChatSessionBusyError
from app.services.llm.exceptions import (
    LLMContextLimitExceededError,
    LLMError,
    LLMProviderError,
    LLMQuotaExceededError,
)
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStoreNotReadyError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


def get_chat_service() -> ChatService:
    """Dependency provider returning singleton ChatService instance."""
    return chat_service


@router.post(
    "/sessions",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a persistent chat session",
    description="Creates a new conversation thread scoped to GLOBAL, DOCUMENT, or COLLECTION.",
)
def create_session(
    request: ChatSessionCreateRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
) -> ChatSessionResponse:
    """Creates a chat session for the authenticated user."""
    try:
        return service.create_session(
            db=db,
            user_id=current_user.id,
            request=request,
        )
    except TargetEntityNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (TargetEntityNotReadyError, ChatSessionValidationError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error creating chat session: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while creating the chat session.",
        ) from exc


@router.get(
    "/sessions",
    response_model=ChatSessionListResponse,
    status_code=status.HTTP_200_OK,
    summary="List chat sessions",
    description="Returns a paginated list of chat sessions owned by the authenticated user.",
)
def list_sessions(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of items to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    is_archived: Optional[bool] = Query(None, description="Optional archive state filter"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
) -> ChatSessionListResponse:
    """Lists conversations for the active user."""
    try:
        return service.list_sessions(
            db=db,
            user_id=current_user.id,
            limit=limit,
            offset=offset,
            is_archived=is_archived,
        )
    except Exception as exc:
        logger.error("Unexpected error listing chat sessions: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while listing chat sessions.",
        ) from exc


@router.get(
    "/sessions/{session_id}",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get chat session details",
    description="Retrieves a specific chat session by ID, enforcing tenant isolation.",
)
def get_session(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
) -> ChatSessionResponse:
    """Retrieves session metadata if owned by the current user."""
    try:
        return service.get_session(
            db=db,
            session_id=session_id,
            user_id=current_user.id,
        )
    except ChatSessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error retrieving chat session %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving the chat session.",
        ) from exc


@router.patch(
    "/sessions/{session_id}",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_200_OK,
    summary="Update chat session",
    description="Updates a conversation title or archive state.",
)
def update_session(
    session_id: UUID,
    request: ChatSessionUpdateRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
) -> ChatSessionResponse:
    """Updates title or archives/unarchives an owned session."""
    try:
        return service.update_session(
            db=db,
            session_id=session_id,
            user_id=current_user.id,
            request=request,
        )
    except ChatSessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found.",
        ) from exc
    except (ChatSessionValidationError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error updating chat session %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while updating the chat session.",
        ) from exc


@router.delete(
    "/sessions/{session_id}",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete chat session",
    description="Permanently deletes a chat session and its message history. Owner only.",
)
def delete_session(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
) -> MessageResponse:
    """Deletes a chat session owned by the authenticated caller."""
    try:
        service.delete_session(
            db=db,
            session_id=session_id,
            user_id=current_user.id,
        )
        return MessageResponse(message="Chat session deleted successfully.")
    except ChatSessionBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"SESSION_BUSY: {exc}",
        ) from exc
    except ChatSessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error deleting chat session %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while deleting the chat session.",
        ) from exc


@router.get(
    "/sessions/{session_id}/messages",
    response_model=ChatMessageListResponse,
    status_code=status.HTTP_200_OK,
    summary="Get conversation message history",
    description="Returns chronological message history with citations for an owned session.",
)
def get_session_messages(
    session_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
) -> ChatMessageListResponse:
    """Retrieves message history for the specified session."""
    try:
        return service.get_session_messages(
            db=db,
            session_id=session_id,
            user_id=current_user.id,
        )
    except ChatSessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error retrieving messages for session %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while retrieving message history.",
        ) from exc


@router.post(
    "/sessions/{session_id}/messages",
    response_model=ChatTurnResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit message and generate grounded turn",
    description="Submits a user query, generates a grounded assistant reply with citations, and persists the turn.",
)
def post_message(
    session_id: UUID,
    request: ChatMessageCreateRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
    service: ChatService = Depends(get_chat_service),
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatTurnResponse:
    """Generates and persists a grounded conversation turn."""
    try:
        return service.post_message_and_respond(
            db=db,
            session_id=session_id,
            user_id=current_user.id,
            request=request,
            rag_service=rag_service,
        )
    except ChatSessionBusyError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"SESSION_BUSY: {exc}",
        ) from exc
    except ChatSessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found.",
        ) from exc
    except TargetEntityNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (ChatSessionArchivedError, TargetEntityNotReadyError, ChatSessionValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except VectorStoreNotReadyError as exc:
        logger.warning("Vector store is not ready during chat turn: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector search service is currently unavailable. Please try again later.",
        ) from exc
    except LLMQuotaExceededError as exc:
        logger.warning("LLM provider quota exceeded during chat turn: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="LLM provider rate limit or quota exceeded. Please try again later.",
        ) from exc
    except LLMContextLimitExceededError as exc:
        logger.warning("LLM context limit exceeded during chat turn: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The query or retrieved context exceeds the model maximum context length.",
        ) from exc
    except LLMProviderError as exc:
        logger.error("Upstream LLM provider failure during chat turn: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream LLM provider failed to generate a response. Please try again later.",
        ) from exc
    except LLMError as exc:
        logger.error("LLM configuration or generic failure during chat turn: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM service encountered an unexpected error.",
        ) from exc
    except ValueError as exc:
        err_msg = str(exc)
        if "not found or not owned" in err_msg:
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
            logger.warning("Sanitized internal ValueError detail during chat turn: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid request parameters.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg,
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during chat turn in session %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred while processing the conversation turn.",
        ) from exc
