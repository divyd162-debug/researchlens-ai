"""RAG Orchestration Service coordinating vector retrieval, database authorization, and grounded LLM generation."""
import logging
import time
from typing import Dict, List, Optional
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chat import ChatScopeType
from app.schemas.llm import LLMChunkContext, LLMRequest
from app.schemas.rag import (
    RAGQueryRequest,
    RAGResponse,
    RetrievedContextChunk,
)
from app.services.embedding_service import EmbeddingService
from app.services.llm.base import BaseLLMProvider
from app.services.llm.citation_parser import CitationParser
from app.services.llm.citation_validator import CitationValidator
from app.services.retrieval_service import DocumentNotReadyError, RetrievalService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGService:
    """Service orchestrating dense retrieval, database access verification, and grounded generation.

    Core Architectural & Security Principles:
    1. PostgreSQL is the Canonical Authority:
       FAISS is a derived, unauthenticated nearest-neighbor index. All candidate chunk IDs
       returned by FAISS MUST be joined against PostgreSQL to verify tenant ownership
       (`document.owner_id == user_id`), document ingestion readiness (`READY`), and conversational
       scoping (`DOCUMENT` or `COLLECTION`). Chunks missing from PostgreSQL or belonging to other
       users are discarded before prompt construction.
    2. Zero Token Consumption on Empty Retrieval:
       If no authorized chunks exist, or all fall below `similarity_threshold`, the service
       immediately returns a standardized fallback response without calling the LLM provider.
    3. Grounding Attribution:
       Output citations are parsed and validated strictly against the authorized retrieved
       context map.
    4. Oversampling Limitation:
       Oversampling FAISS by `max(top_k * 3, 15)` retrieves a candidate pool to account for
       unauthorized or deleted items. If a scoped collection or tenant represents a very small
       fraction of the shared FAISS index, fewer than `top_k` authorized chunks may be found
       within the top 3*K global candidates even if more exist further down the index.
    5. Embedding Sequence-Length Alignment:
       `EmbeddingService` utilizes `all-MiniLM-L6-v2` with a sequence length limit of 256 WordPiece tokens.
       `ChunkingService` defaults to ~200 tokens (~800 characters) to ensure passages comfortably fit
       within this limit without silent truncation, preserving full semantic fidelity across vectors and context.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_provider: BaseLLMProvider,
        citation_parser: Optional[CitationParser] = None,
        citation_validator: Optional[CitationValidator] = None,
        retrieval_service: Optional[RetrievalService] = None,
    ):
        """Initializes RAGService with injected service dependencies.

        Args:
            embedding_service: Service for encoding text into dense vectors.
            vector_store: FAISS vector store for similarity search.
            llm_provider: Configured LLM provider instance.
            citation_parser: Optional custom CitationParser instance.
            citation_validator: Optional custom CitationValidator instance.
            retrieval_service: Optional custom RetrievalService instance.
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.citation_parser = citation_parser or CitationParser()
        self.citation_validator = citation_validator or CitationValidator()
        self.retrieval_service = retrieval_service or RetrievalService(
            embedding_service=embedding_service,
            vector_store=vector_store,
        )

    def answer_query(
        self,
        db: Session,
        query: str,
        user_id: UUID,
        scope_type: ChatScopeType = ChatScopeType.GLOBAL,
        scope_id: Optional[UUID] = None,
        document_ids: Optional[List[UUID]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
    ) -> RAGResponse:
        """Executes an end-to-end grounded RAG search and generation workflow.

        Args:
            db: Active SQLAlchemy database session.
            query: User's question or search query.
            user_id: Authenticated user ID for multi-tenant isolation.
            scope_type: Retrieval scope (GLOBAL, DOCUMENT, COLLECTION, DOCUMENTS).
            scope_id: Target Document ID or Collection ID when scoped.
            document_ids: Target Document IDs when scope_type is DOCUMENTS.
            top_k: Maximum number of passage contexts to provide to the LLM.
            similarity_threshold: Minimum cosine similarity score required for candidate chunks.

        Returns:
            RAGResponse containing raw answer, validated citations, and token metrics.

        Raises:
            ValueError: If query is empty, top_k <= 0, or target scope entity is invalid/unowned.
            LLMError subclasses: If the LLM provider fails during generation.
        """
        cleaned_query = str(query).strip() if query is not None else ""
        scope_str = scope_type.value if hasattr(scope_type, "value") else str(scope_type)
        t_rag_start = time.monotonic()

        # Delegate retrieval and authorization to RetrievalService
        try:
            t_retrieval_start = time.monotonic()
            top_contexts = self.retrieval_service.retrieve_relevant_chunks(
                db=db,
                query=query,
                user_id=user_id,
                scope_type=scope_type,
                scope_id=scope_id,
                document_ids=document_ids,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            retrieval_duration_ms = (time.monotonic() - t_retrieval_start) * 1000
        except DocumentNotReadyError as exc:
            total_duration_ms = (time.monotonic() - t_rag_start) * 1000
            logger.info(
                "RAG query aborted: target document not ready in %.2fms [scope: %s, file: %s, status: %s]",
                total_duration_ms,
                scope_str,
                exc.filename,
                exc.processing_status,
            )
            return RAGResponse(
                query=cleaned_query,
                answer=f"Target document '{exc.filename}' is currently in status '{exc.processing_status}' and not ready for queries.",
                citations=[],
                status="NO_RELEVANT_CONTEXT",
                model_name=self.llm_provider.model_name,
                provider_name=self.llm_provider.provider_name,
                retrieved_chunk_count=0,
            )

        # Early Exit if No Authorized Contexts Remain
        if not top_contexts:
            total_duration_ms = (time.monotonic() - t_rag_start) * 1000
            logger.info(
                "RAG query completed early in %.2fms (retrieval: %.2fms): 0 authorized passages above threshold [scope: %s]",
                total_duration_ms,
                retrieval_duration_ms,
                scope_str,
            )
            return RAGResponse(
                query=cleaned_query,
                answer="No relevant context was found in your library to answer the prompt.",
                citations=[],
                status="NO_RELEVANT_CONTEXT",
                model_name=self.llm_provider.model_name,
                provider_name=self.llm_provider.provider_name,
                retrieved_chunk_count=0,
            )

        # 8. Construct LLM Request
        t_prompt_start = time.monotonic()
        llm_chunk_contexts = [
            LLMChunkContext(
                chunk_id=ctx.chunk_id,
                content=ctx.content,
                page_number=ctx.page_number,
                document_title=ctx.document_title,
            )
            for ctx in top_contexts
        ]
        retrieved_map: Dict[int, RetrievedContextChunk] = {ctx.chunk_id: ctx for ctx in top_contexts}

        llm_request = LLMRequest(
            prompt=cleaned_query,
            context_chunks=llm_chunk_contexts,
        )
        prompt_duration_ms = (time.monotonic() - t_prompt_start) * 1000

        # 9. Provider Generation (Exceptions bubble up cleanly)
        logger.info(
            "Dispatching LLM generation request to provider '%s' [model: %s, context_chunks: %d]",
            self.llm_provider.provider_name,
            self.llm_provider.model_name,
            len(llm_chunk_contexts),
        )
        t_gen_start = time.monotonic()
        llm_response = self.llm_provider.generate(llm_request)
        generation_duration_ms = (time.monotonic() - t_gen_start) * 1000

        # 10. Parse & Validate Citations
        t_val_start = time.monotonic()
        parsed = self.citation_parser.parse(llm_response.content)
        validated = self.citation_validator.validate(
            parsed,
            retrieved_map,
            generated_text=llm_response.content,
        )
        validation_duration_ms = (time.monotonic() - t_val_start) * 1000

        if validated.unsupported_numbers:
            logger.warning(
                "RAG answer contains %d numerical value(s) not found in cited evidence: %s",
                len(validated.unsupported_numbers),
                validated.unsupported_numbers,
            )

        total_rag_duration_ms = (time.monotonic() - t_rag_start) * 1000
        logger.info(
            "RAG query completed in %.2fms (retrieval: %.2fms, prompt_construction: %.2fms, generation: %.2fms, citation_validation: %.2fms) [scope: %s, retrieved_chunks: %d, citations: %d, provider: %s, model: %s]",
            total_rag_duration_ms,
            retrieval_duration_ms,
            prompt_duration_ms,
            generation_duration_ms,
            validation_duration_ms,
            scope_str,
            len(top_contexts),
            len(validated.citations),
            self.llm_provider.provider_name,
            self.llm_provider.model_name,
        )

        return RAGResponse(
            query=cleaned_query,
            answer=llm_response.content,
            citations=validated.citations,
            unknown_chunk_ids=validated.unknown_chunk_ids,
            malformed_markers=validated.malformed_markers,
            unsupported_numbers=validated.unsupported_numbers,
            status="SUCCESS",
            model_name=llm_response.model_name,
            provider_name=llm_response.provider_name,
            retrieved_chunk_count=len(top_contexts),
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
        )
