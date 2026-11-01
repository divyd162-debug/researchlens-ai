from datetime import datetime, timedelta, timezone
import logging
import time
from typing import List, Optional
from uuid import UUID
from sqlalchemy import func, or_
from sqlalchemy.orm import Session, joinedload, selectinload


from app.models.chat import (
    ChatScopeType,
    ChatSession,
    ChatSessionDocument,
    Message,
    MessageCitation,
    SenderType,
)
from app.models.collection import Collection
from app.models.document import Document, ProcessingStatus
from app.schemas.chat import (
    ChatMessageCreateRequest,
    ChatMessageListResponse,
    ChatMessageResponse,
    ChatSessionCreateRequest,
    ChatSessionListResponse,
    ChatSessionResponse,
    ChatSessionUpdateRequest,
    ChatTurnResponse,
    DocumentScopeSummary,
    MessageCitationResponse,
)
from app.services.chat_concurrency import (
    ChatConcurrencyManager,
    ChatSessionBusyError,
    chat_concurrency_manager,
)
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)


class ChatServiceError(Exception):
    """Base exception for chat service errors."""
    pass


class ChatSessionNotFoundError(ChatServiceError):
    """Raised when a chat session is not found or not owned by the user."""
    pass


class ChatSessionArchivedError(ChatServiceError):
    """Raised when an operation is attempted on an archived chat session."""
    pass


class ChatSessionValidationError(ChatServiceError):
    """Raised when chat session parameters or state transitions are invalid."""
    pass


class TargetEntityNotFoundError(ChatServiceError):
    """Raised when a scoped document or collection is not found or not owned by user."""
    pass


class TargetEntityNotReadyError(ChatServiceError):
    """Raised when a scoped document is not yet in READY status for querying."""
    pass


class ChatService:
    """Service encapsulating business logic and persistence for multi-tenant chat conversations."""

    def __init__(self, concurrency_manager: Optional[ChatConcurrencyManager] = None):
        """Initializes ChatService with an optional injected concurrency manager."""
        self._concurrency_manager = concurrency_manager or chat_concurrency_manager

    def create_session(
        self,
        db: Session,
        user_id: UUID,
        request: ChatSessionCreateRequest,
    ) -> ChatSessionResponse:
        """Creates a new persistent conversation thread scoped to GLOBAL, DOCUMENT, or COLLECTION.

        Security & Integrity:
        - owner_id is strictly derived from the authenticated user.
        - DOCUMENT scope requires an existing, READY document owned by user_id.
        - COLLECTION scope requires an existing collection owned by user_id.
        - Scope targets must satisfy ck_chat_session_scope_targets check constraint.
        """
        title = request.title.strip() if request.title else "New Conversation"

        if request.scope_type == ChatScopeType.GLOBAL:
            if request.target_document_id is not None or request.target_collection_id is not None:
                raise ChatSessionValidationError("GLOBAL conversations must not specify target entities.")

        elif request.scope_type == ChatScopeType.DOCUMENT:
            if request.target_document_id is None:
                raise ChatSessionValidationError("DOCUMENT conversations require a target_document_id.")
            if request.target_collection_id is not None:
                raise ChatSessionValidationError("DOCUMENT conversations cannot specify target_collection_id.")

            # Validate target document exists, belongs to user, and is READY
            doc = (
                db.query(Document)
                .filter(Document.id == request.target_document_id, Document.owner_id == user_id)
                .first()
            )
            if doc is None:
                raise TargetEntityNotFoundError("Target document not found.")
            if doc.processing_status != ProcessingStatus.READY.value:
                raise TargetEntityNotReadyError(
                    f"Target document '{doc.filename}' is currently in status '{doc.processing_status}' and not ready for chat."
                )

        elif request.scope_type == ChatScopeType.COLLECTION:
            if request.target_collection_id is None:
                raise ChatSessionValidationError("COLLECTION conversations require a target_collection_id.")
            if request.target_document_id is not None:
                raise ChatSessionValidationError("COLLECTION conversations cannot specify target_document_id.")

            # Validate target collection exists and belongs to user
            col = (
                db.query(Collection)
                .filter(Collection.id == request.target_collection_id, Collection.owner_id == user_id)
                .first()
            )
            if col is None:
                raise TargetEntityNotFoundError("Target collection not found.")

        elif request.scope_type == ChatScopeType.DOCUMENTS:
            if request.target_document_id is not None or request.target_collection_id is not None:
                raise ChatSessionValidationError("DOCUMENTS conversations must not specify target_document_id or target_collection_id.")
            if not request.document_ids:
                raise ChatSessionValidationError("DOCUMENTS conversations require at least one document_id in document_ids.")

            # Validate target documents exist, belong to user, and are READY
            docs = (
                db.query(Document)
                .filter(Document.id.in_(request.document_ids), Document.owner_id == user_id)
                .all()
            )
            if len(docs) != len(set(request.document_ids)):
                raise TargetEntityNotFoundError("One or more selected documents not found or not owned by user.")
            for doc in docs:
                if doc.processing_status != ProcessingStatus.READY.value:
                    raise TargetEntityNotReadyError(
                        f"Target document '{doc.filename}' is currently in status '{doc.processing_status}' and not ready for chat."
                    )

        session = ChatSession(
            owner_id=user_id,
            title=title,
            scope_type=request.scope_type.value,
            target_document_id=request.target_document_id,
            target_collection_id=request.target_collection_id,
            is_archived=False,
        )
        db.add(session)
        db.flush()

        if request.scope_type == ChatScopeType.DOCUMENTS:
            for doc in docs:
                db.add(ChatSessionDocument(chat_session_id=session.id, document_id=doc.id))

        try:
            db.commit()
            db.refresh(session)
        except Exception:
            db.rollback()
            raise

        response = ChatSessionResponse.model_validate(session)
        response.message_count = 0
        if request.scope_type == ChatScopeType.DOCUMENTS:
            response.selected_documents = [
                DocumentScopeSummary.model_validate(doc)
                for doc in docs
            ]
        return response

    def list_sessions(
        self,
        db: Session,
        user_id: UUID,
        limit: int = 50,
        offset: int = 0,
        is_archived: Optional[bool] = None,
    ) -> ChatSessionListResponse:
        """Returns paginated list of chat sessions owned by user with aggregated message counts.

        Avoids N+1 query pattern by joining an aggregated message count subquery.
        """
        # Clamp bounds
        safe_limit = max(1, min(limit, 100))
        safe_offset = max(0, offset)

        # Base filter condition
        filter_cond = [ChatSession.owner_id == user_id]
        if is_archived is not None:
            filter_cond.append(ChatSession.is_archived == is_archived)

        total = db.query(func.count(ChatSession.id)).filter(*filter_cond).scalar() or 0

        # Subquery to count messages per session
        message_count_subq = (
            db.query(
                Message.chat_session_id,
                func.count(Message.id).label("msg_count"),
            )
            .group_by(Message.chat_session_id)
            .subquery()
        )

        rows = (
            db.query(
                ChatSession,
                func.coalesce(message_count_subq.c.msg_count, 0).label("message_count"),
            )
            .options(selectinload(ChatSession.selected_documents))
            .outerjoin(
                message_count_subq,
                ChatSession.id == message_count_subq.c.chat_session_id,
            )
            .filter(*filter_cond)
            .order_by(ChatSession.updated_at.desc(), ChatSession.created_at.desc())
            .offset(safe_offset)
            .limit(safe_limit)
            .all()
        )

        items = []
        for session, msg_count in rows:
            resp = ChatSessionResponse.model_validate(session)
            resp.message_count = int(msg_count)
            items.append(resp)

        return ChatSessionListResponse(
            items=items,
            total=total,
            limit=safe_limit,
            offset=safe_offset,
        )

    def get_session(
        self,
        db: Session,
        session_id: UUID,
        user_id: UUID,
    ) -> ChatSessionResponse:
        """Retrieves a specific chat session with its message count, enforcing owner isolation."""
        message_count_subq = (
            db.query(
                Message.chat_session_id,
                func.count(Message.id).label("msg_count"),
            )
            .filter(Message.chat_session_id == session_id)
            .group_by(Message.chat_session_id)
            .subquery()
        )

        row = (
            db.query(
                ChatSession,
                func.coalesce(message_count_subq.c.msg_count, 0).label("message_count"),
            )
            .options(selectinload(ChatSession.selected_documents))
            .outerjoin(
                message_count_subq,
                ChatSession.id == message_count_subq.c.chat_session_id,
            )
            .filter(ChatSession.id == session_id, ChatSession.owner_id == user_id)
            .first()
        )

        if row is None:
            raise ChatSessionNotFoundError("Chat session not found.")

        session, msg_count = row
        resp = ChatSessionResponse.model_validate(session)
        resp.message_count = int(msg_count)
        return resp

    def update_session(
        self,
        db: Session,
        session_id: UUID,
        user_id: UUID,
        request: ChatSessionUpdateRequest,
    ) -> ChatSessionResponse:
        """Updates a chat session's title or archive state while enforcing constraints.

        Security & Validation:
        - Enforces session ownership (returns 404 if not found/unowned).
        - If unarchiving a scoped session whose target was deleted, rejects unarchiving
          to satisfy ck_chat_session_scope_targets check constraint.
        """
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.owner_id == user_id)
            .first()
        )
        if session is None:
            raise ChatSessionNotFoundError("Chat session not found.")

        if request.title is not None:
            session.title = request.title.strip()

        if request.is_archived is not None:
            if request.is_archived is False and session.is_archived is True:
                # Validate unarchiving: active scoped sessions must retain their targets
                if session.scope_type == ChatScopeType.DOCUMENT.value:
                    if session.target_document_id is None:
                        raise ChatSessionValidationError(
                            "Cannot unarchive document-scoped session whose target document was deleted."
                        )
                    # Verify target document still exists and belongs to user
                    doc = (
                        db.query(Document)
                        .filter(Document.id == session.target_document_id, Document.owner_id == user_id)
                        .first()
                    )
                    if doc is None:
                        raise ChatSessionValidationError(
                            "Cannot unarchive session because target document no longer exists."
                        )

                elif session.scope_type == ChatScopeType.COLLECTION.value:
                    if session.target_collection_id is None:
                        raise ChatSessionValidationError(
                            "Cannot unarchive collection-scoped session whose target collection was deleted."
                        )
                    # Verify target collection still exists and belongs to user
                    col = (
                        db.query(Collection)
                        .filter(Collection.id == session.target_collection_id, Collection.owner_id == user_id)
                        .first()
                    )
                    if col is None:
                        raise ChatSessionValidationError(
                            "Cannot unarchive session because target collection no longer exists."
                        )

                elif session.scope_type == ChatScopeType.DOCUMENTS.value:
                    remaining_docs = (
                        db.query(ChatSessionDocument)
                        .filter(ChatSessionDocument.chat_session_id == session.id)
                        .count()
                    )
                    if remaining_docs == 0:
                        raise ChatSessionValidationError(
                            "Cannot unarchive multi-document session because all target documents were deleted."
                        )

                session.is_archived = False
            elif request.is_archived is True:
                session.is_archived = True

        try:
            db.commit()
            db.refresh(session)
        except Exception:
            db.rollback()
            raise

        # Fetch message count for response
        msg_count = (
            db.query(func.count(Message.id))
            .filter(Message.chat_session_id == session.id)
            .scalar()
            or 0
        )
        resp = ChatSessionResponse.model_validate(session)
        resp.message_count = int(msg_count)
        return resp

    def delete_session(
        self,
        db: Session,
        session_id: UUID,
        user_id: UUID,
    ) -> None:
        """Deletes a chat session and all its associated messages and citations.

        Security & Integrity:
        - Owner-scoped: Returns ChatSessionNotFoundError if session does not exist
          or is not owned by user_id, preventing cross-tenant information leakage.
        - Cascades deletion to child messages and message citations safely within
          a database transaction.
        - Preserves all documents, chunks, vectors, collections, and other sessions.
        """
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.owner_id == user_id)
            .first()
        )
        if session is None:
            raise ChatSessionNotFoundError("Chat session not found.")

        # Concurrency safety: Atomically reserve session for DELETING; raises ChatSessionBusyError if busy
        with self._concurrency_manager.session_reservation(session_id, "DELETING"):
            try:
                db.delete(session)
                db.commit()
                logger.info("Deleted chat session %s for user %s", session_id, user_id)
            except Exception:
                db.rollback()
                raise

    def get_session_messages(
        self,
        db: Session,
        session_id: UUID,
        user_id: UUID,
    ) -> ChatMessageListResponse:
        """Retrieves chronological message history with preloaded citation snapshots.

        Avoids N+1 query patterns by using joinedload(Message.citations).
        """
        # Ownership check
        session_exists = (
            db.query(ChatSession.id)
            .filter(ChatSession.id == session_id, ChatSession.owner_id == user_id)
            .scalar()
        )
        if not session_exists:
            raise ChatSessionNotFoundError("Chat session not found.")

        messages = (
            db.query(Message)
            .options(joinedload(Message.citations))
            .filter(Message.chat_session_id == session_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
            .all()
        )

        items = [ChatMessageResponse.model_validate(m) for m in messages]
        return ChatMessageListResponse(
            session_id=session_id,
            items=items,
            total=len(items),
        )

    def post_message_and_respond(
        self,
        db: Session,
        session_id: UUID,
        user_id: UUID,
        request: ChatMessageCreateRequest,
        rag_service: RAGService,
    ) -> ChatTurnResponse:
        """Submits a user message and generates a grounded response, persisting the turn atomically.

        Transaction & Concurrency Safety:
        1. Pre-flight check: Verifies session ownership and ensures session is not archived.
        2. Scope resolution: Derives retrieval scope directly from the database record;
           never accepts client overrides.
        3. LLM Generation: RAGService.answer_query() is executed before opening the write
           transaction, preventing long connection locks during model generation.
        4. Recheck & Row Locking: Immediately before persisting the turn, re-verifies session
           ownership and archive state with row locking (.with_for_update()) and fresh state
           (.populate_existing()), preventing in-flight archiving races.
        5. Failure atomicity: If the LLM provider fails, no partial records are written.
        6. Atomic turn commit: User message, assistant message, validated citations, and session
           updated_at timestamp are persisted together in a single transaction.
        """
        # 1. Pre-flight check: Session exists, belongs to user, and is active
        t_chat_start = time.monotonic()
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.owner_id == user_id)
            .first()
        )
        if session is None:
            raise ChatSessionNotFoundError("Chat session not found.")

        if session.is_archived:
            raise ChatSessionArchivedError("Cannot post messages to an archived chat session.")

        # 2. Concurrency reservation: Atomically reserve session for GENERATING
        # Rejects overlapping generation or concurrent deletion attempts
        with self._concurrency_manager.session_reservation(session_id, "GENERATING"):
            # 3. Scope resolution
            scope_type = ChatScopeType(session.scope_type)
            scope_id: Optional[UUID] = None
            document_ids: Optional[List[UUID]] = None

            if scope_type == ChatScopeType.GLOBAL:
                scope_id = None
            elif scope_type == ChatScopeType.DOCUMENT:
                if not session.target_document_id:
                    raise ChatSessionValidationError("Target document is missing for this document-scoped session.")
                doc = (
                    db.query(Document)
                    .filter(Document.id == session.target_document_id, Document.owner_id == user_id)
                    .first()
                )
                if doc is None:
                    raise TargetEntityNotFoundError("Target document not found.")
                if doc.processing_status != ProcessingStatus.READY.value:
                    raise TargetEntityNotReadyError(
                        f"Target document '{doc.filename}' is currently in status '{doc.processing_status}' and not ready for queries."
                    )
                scope_id = session.target_document_id
            elif scope_type == ChatScopeType.COLLECTION:
                if not session.target_collection_id:
                    raise ChatSessionValidationError("Target collection is missing for this collection-scoped session.")
                col = (
                    db.query(Collection)
                    .filter(Collection.id == session.target_collection_id, Collection.owner_id == user_id)
                    .first()
                )
                if col is None:
                    raise TargetEntityNotFoundError("Target collection not found.")
                scope_id = session.target_collection_id
            elif scope_type == ChatScopeType.DOCUMENTS:
                session_docs = (
                    db.query(Document)
                    .join(ChatSessionDocument, ChatSessionDocument.document_id == Document.id)
                    .filter(
                        ChatSessionDocument.chat_session_id == session.id,
                        Document.owner_id == user_id,
                    )
                    .all()
                )
                if not session_docs:
                    raise ChatSessionValidationError("No documents remain associated with this multi-document session.")
                for d in session_docs:
                    if d.processing_status != ProcessingStatus.READY.value:
                        raise TargetEntityNotReadyError(
                            f"Target document '{d.filename}' is currently in status '{d.processing_status}' and not ready for queries."
                        )
                document_ids = [d.id for d in session_docs]

            # 4. Grounded RAG Query execution (network / LLM call happens outside the write transaction)
            t_rag_start = time.monotonic()
            rag_response = rag_service.answer_query(
                db=db,
                query=request.content,
                user_id=user_id,
                scope_type=scope_type,
                scope_id=scope_id,
                document_ids=document_ids,
                top_k=request.top_k,
                similarity_threshold=request.similarity_threshold,
            )
            rag_duration_ms = (time.monotonic() - t_rag_start) * 1000

            # 5. Atomic turn persistence
            t_persist_start = time.monotonic()
            now = datetime.now(timezone.utc)
            try:
                # Concurrency & Archive Protection:
                # Recheck session ownership and archive state immediately before persistence.
                # with_for_update() locks the session row in PostgreSQL during this write transaction,
                # preventing concurrent archiving requests from committing between check and commit.
                # populate_existing() ensures we read the latest committed state rather than stale
                # in-memory attributes cached during the earlier pre-flight read.
                current_session = (
                    db.query(ChatSession)
                    .filter(ChatSession.id == session_id, ChatSession.owner_id == user_id)
                    .with_for_update()
                    .populate_existing()
                    .first()
                )
                if current_session is None:
                    raise ChatSessionNotFoundError("Chat session not found.")

                if current_session.is_archived:
                    raise ChatSessionArchivedError("Cannot post messages to an archived chat session.")

                user_msg = Message(
                    chat_session_id=current_session.id,
                    sender_type=SenderType.USER.value,
                    content=request.content,
                    token_count=None,
                    created_at=now,
                )
                db.add(user_msg)
                db.flush()

                assistant_msg = Message(
                    chat_session_id=current_session.id,
                    sender_type=SenderType.ASSISTANT.value,
                    content=rag_response.answer,
                    token_count=rag_response.completion_tokens,
                    created_at=now + timedelta(microseconds=1000),
                )
                db.add(assistant_msg)
                db.flush()

                # Attach validated citation snapshots
                persisted_citations = []
                for cit in rag_response.citations:
                    db_cit = MessageCitation(
                        message_id=assistant_msg.id,
                        chunk_id=cit.chunk_id,
                        quoted_text=cit.quoted_text,
                        page_number=cit.page_number,
                        document_title=cit.document_title,
                        similarity_score=cit.similarity_score,
                    )
                    db.add(db_cit)
                    persisted_citations.append(db_cit)

                current_session.updated_at = func.now()
                db.commit()

                # Refresh persisted messages
                db.refresh(user_msg)
                db.refresh(assistant_msg)
                for c in persisted_citations:
                    db.refresh(c)
            except Exception:
                db.rollback()
                raise

            persist_duration_ms = (time.monotonic() - t_persist_start) * 1000
            total_chat_duration_ms = (time.monotonic() - t_chat_start) * 1000

            scope_str = scope_type.value if hasattr(scope_type, "value") else str(scope_type)
            logger.info(
                "Chat turn completed in %.2fms (RAG: %.2fms, DB persistence: %.2fms) [session_id: %s, user_id: %s, scope: %s, retrieved_chunks: %d, citations: %d, model: %s]",
                total_chat_duration_ms,
                rag_duration_ms,
                persist_duration_ms,
                session_id,
                user_id,
                scope_str,
                rag_response.retrieved_chunk_count,
                len(rag_response.citations),
                rag_response.model_name,
            )

            user_resp = ChatMessageResponse.model_validate(user_msg)
            user_resp.citations = []

            asst_resp = ChatMessageResponse.model_validate(assistant_msg)
            asst_resp.citations = [MessageCitationResponse.model_validate(c) for c in persisted_citations]

            return ChatTurnResponse(
                session_id=current_session.id,
                user_message=user_resp,
                assistant_message=asst_resp,
                retrieved_chunk_count=rag_response.retrieved_chunk_count,
                model_name=rag_response.model_name,
                provider_name=rag_response.provider_name,
                prompt_tokens=rag_response.prompt_tokens,
                completion_tokens=rag_response.completion_tokens,
                total_tokens=rag_response.total_tokens,
            )


# Singleton service instance
chat_service = ChatService()
