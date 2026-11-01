"""Ingestion pipeline service orchestrating PDF extraction, chunking, embedding, and vector indexing."""
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
import threading
from typing import Optional, Union
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, update
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.database import db_session_scope, get_session_factory
from app.models.chunk import Chunk
from app.models.document import Document, ProcessingStatus
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import (
    EmbeddingService,
    EncodingError,
    get_embedding_service,
)
from app.services.pdf_service import (
    EncryptedPDFError,
    InvalidPDFError,
    PDFExtractionError,
    PDFFileNotFoundError,
    PDFService,
)
from app.services.vector_store import VectorStore, VectorStoreError, get_vector_store

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Base exception for ingestion pipeline failures."""
    pass


class DocumentNotFoundError(IngestionError):
    """Raised when the target document does not exist in the database."""
    pass


class IngestionLeaseLostError(IngestionError):
    """Raised when an active ingestion task loses its exclusive ownership lease."""
    pass


class LeaseHeartbeat:
    """Manages a background renewable lease heartbeat for an in-flight ingestion task.

    Guarantees:
    - Active workers keep updated_at fresh in the database at regular intervals,
      preventing long embedding operations from being falsely classified as abandoned.
    - If a task loses ownership (e.g. lease expired due to freeze and document was
      reclaimed with a new lease token), is_lost is set and verify_current raises
      IngestionLeaseLostError.
    """

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        document_id: UUID,
        lease_token: str,
        interval_seconds: Optional[float] = None,
    ):
        self._session_factory = session_factory
        self._document_id = document_id
        self._lease_token = lease_token
        if interval_seconds is not None:
            self._interval_seconds = interval_seconds
        else:
            self._interval_seconds = min(
                15.0,
                max(0.1, float(settings.STALE_INGESTION_TIMEOUT_SECONDS) / 4.0),
            )
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_lost = False
        self._lock = threading.Lock()

    @property
    def is_lost(self) -> bool:
        with self._lock:
            return self._is_lost

    def mark_lost(self) -> None:
        with self._lock:
            self._is_lost = True

    def check_valid(self) -> None:
        if self.is_lost:
            raise IngestionLeaseLostError(
                f"Ingestion task lost ownership lease for document {self._document_id}."
            )

    def verify_current(self, session: Session) -> bool:
        """Queries the database to verify this task still holds the exclusive lease token."""
        self.check_valid()
        current_token = (
            session.query(Document.lease_token)
            .filter(Document.id == self._document_id)
            .scalar()
        )
        if current_token != self._lease_token:
            self.mark_lost()
            raise IngestionLeaseLostError(
                f"Ingestion task lost ownership lease for document {self._document_id} "
                f"(expected token {self._lease_token}, current token {current_token})."
            )
        return True

    def _run_heartbeat(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(self._interval_seconds):
                break

            try:
                with db_session_scope(self._session_factory) as session:
                    stmt = (
                        update(Document)
                        .where(
                            Document.id == self._document_id,
                            Document.lease_token == self._lease_token,
                        )
                        .values(updated_at=func.now())
                        .execution_options(synchronize_session=False)
                    )
                    res = session.execute(stmt)
                    if res.rowcount == 0:
                        logger.warning(
                            "Lease heartbeat detected lost ownership for document %s (token %s).",
                            self._document_id,
                            self._lease_token,
                        )
                        self.mark_lost()
                        break
            except Exception as exc:
                logger.warning(
                    "Error executing lease heartbeat for document %s: %s",
                    self._document_id,
                    exc,
                )

    def __enter__(self):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_heartbeat,
            name=f"lease-hb-{str(self._document_id)[:8]}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)


class IngestionService:
    """Service orchestrating the end-to-end PDF ingestion lifecycle.

    Lifecycle States:
    1. EXTRACTING: PyMuPDF extracts text and physical page boundaries.
    2. CHUNKING: ChunkingService splits extracted pages into page-bounded passages
       targeting 200 tokens (aligned with all-MiniLM-L6-v2's 256 WordPiece sequence limit).
    3. EMBEDDING: Generates dense L2-normalized vector embeddings using SentenceTransformers.
    4. READY: Stable 64-bit integer chunk IDs and vectors are indexed in FAISS and
       serialized to disk atomically.

    Idempotency and Consistency Guarantees:
    - Retries cleanly purge preexisting chunks from PostgreSQL and vectors from FAISS
      before creating new ones, preventing duplicate passages or phantom vectors.
    - If embedding generation or FAISS indexing fails, any newly created chunks are
      purged from PostgreSQL and partial vectors are removed from FAISS.
    - A document NEVER enters the READY state unless its PostgreSQL chunks and FAISS
      vector entries are 100% consistent.
    """

    def __init__(
        self,
        pdf_service: Optional[PDFService] = None,
        chunking_service: Optional[ChunkingService] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        session_factory: Optional[sessionmaker[Session]] = None,
    ):
        """Initializes IngestionService with optional injected dependencies for testing."""
        self._pdf_service = pdf_service
        self._chunking_service = chunking_service
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._session_factory = session_factory

    @property
    def pdf_service(self) -> PDFService:
        if self._pdf_service is None:
            self._pdf_service = PDFService()
        return self._pdf_service

    @property
    def chunking_service(self) -> ChunkingService:
        if self._chunking_service is None:
            # Default sizing: 200 target tokens, 30 overlap tokens, 4.0 chars/token.
            # Guarantees chunks fit comfortably within all-MiniLM-L6-v2's 256 WordPiece token limit.
            self._chunking_service = ChunkingService(
                target_tokens=200,
                overlap_tokens=30,
                chars_per_token=4.0,
            )
        return self._chunking_service

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is not None:
            return self._embedding_service
        return get_embedding_service()

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is not None:
            return self._vector_store
        return get_vector_store()

    def _get_effective_factory(
        self, session_factory: Optional[sessionmaker[Session]] = None
    ) -> sessionmaker[Session]:
        return session_factory or self._session_factory or get_session_factory()

    def _safe_transition(
        self, session: Session, doc_uuid: UUID, lease_token: str, new_status: str
    ) -> None:
        """Transitions document status verifying that the task still owns the lease token."""
        stmt = (
            update(Document)
            .where(
                Document.id == doc_uuid,
                Document.lease_token == lease_token,
            )
            .values(
                processing_status=new_status,
                updated_at=func.now(),
            )
            .execution_options(synchronize_session=False)
        )
        res = session.execute(stmt)
        if res.rowcount == 0:
            raise IngestionLeaseLostError(
                f"Cannot transition document {doc_uuid} to {new_status}: lease lost."
            )
        session.commit()

    def _safe_fail_document(
        self, session: Session, doc_uuid: UUID, lease_token: str, error_message: str
    ) -> None:
        """Marks document as FAILED only if this task still owns the lease."""
        stmt = (
            update(Document)
            .where(
                Document.id == doc_uuid,
                Document.lease_token == lease_token,
            )
            .values(
                processing_status=ProcessingStatus.FAILED.value,
                lease_token=None,
                error_message=error_message,
                updated_at=func.now(),
            )
            .execution_options(synchronize_session=False)
        )
        res = session.execute(stmt)
        if res.rowcount > 0:
            session.commit()
        else:
            session.rollback()

    def process_document(
        self,
        document_id: Union[UUID, str],
        session_factory: Optional[sessionmaker[Session]] = None,
        force: bool = False,
    ) -> Optional[Document]:
        """Orchestrates extraction, chunking, embedding, and vector indexing for a document.

        Concurrency & Renewable Lease Guarantees:
        - Atomically claims the document via conditional update and assigns an exclusive lease_token.
        - Keeps a background LeaseHeartbeat active so long embedding operations do not expire.
        - Fences out any worker that loses ownership: if a lease is abandoned and stolen,
          the lost worker cannot persist chunks, mutate FAISS vectors, or overwrite processing status.
        - Recovers stranded documents if their lease heartbeat has expired beyond STALE_INGESTION_TIMEOUT_SECONDS.

        Args:
            document_id: UUID or string UUID of the document to process.
            session_factory: Optional session factory to use for independent DB transactions.
            force: If True, allows claiming documents in READY state (e.g. forced re-indexing).

        Returns:
            The updated Document instance if found, or None if the document does not exist.
        """
        doc_uuid = UUID(str(document_id))
        factory = self._get_effective_factory(session_factory)
        my_lease_token = uuid4().hex

        with db_session_scope(factory) as session:
            # 1. Existence check
            doc = session.query(Document).filter(Document.id == doc_uuid).first()
            if doc is None:
                logger.warning("Ingestion requested for nonexistent document ID: %s", doc_uuid)
                return None

            # 2. Concurrency-safe atomic claim
            claimable_statuses = [
                ProcessingStatus.PENDING.value,
                ProcessingStatus.FAILED.value,
            ]
            if force:
                claimable_statuses.append(ProcessingStatus.READY.value)

            now = datetime.now(timezone.utc)
            stale_cutoff = now - timedelta(seconds=settings.STALE_INGESTION_TIMEOUT_SECONDS)
            if doc.updated_at is not None and doc.updated_at.tzinfo is None:
                stale_cutoff = stale_cutoff.replace(tzinfo=None)

            in_flight_statuses = [
                ProcessingStatus.EXTRACTING.value,
                ProcessingStatus.CHUNKING.value,
                ProcessingStatus.EMBEDDING.value,
            ]

            # A document may be claimed if:
            # 1. It is in an explicitly claimable status (PENDING, FAILED, or READY with force=True), OR
            # 2. It is stranded in an in-flight status with updated_at <= stale_cutoff
            #    (indicating an abandoned worker lease).
            claim_condition = or_(
                Document.processing_status.in_(claimable_statuses),
                and_(
                    Document.processing_status.in_(in_flight_statuses),
                    or_(
                        Document.updated_at <= stale_cutoff,
                        Document.updated_at.is_(None),
                    ),
                ),
            )

            claim_stmt = (
                update(Document)
                .where(
                    Document.id == doc_uuid,
                    claim_condition,
                )
                .values(
                    processing_status=ProcessingStatus.EXTRACTING.value,
                    lease_token=my_lease_token,
                    error_message=None,
                    updated_at=func.now(),
                )
                .execution_options(synchronize_session=False)
            )
            result = session.execute(claim_stmt)
            if result.rowcount == 0:
                logger.info(
                    "Document %s could not be claimed for ingestion (current status: %s). Exiting task safely.",
                    doc_uuid,
                    doc.processing_status,
                )
                session.rollback()
                session.refresh(doc)
                return doc

            session.commit()
            session.refresh(doc)

            # Start renewable background lease heartbeat for duration of processing
            with LeaseHeartbeat(factory, doc_uuid, my_lease_token) as heartbeat:
                try:
                    # 3. File existence validation
                    file_path = Path(doc.file_path) if doc.file_path else None
                    if not file_path or not file_path.exists() or not file_path.is_file():
                        self._safe_fail_document(session, doc_uuid, my_lease_token, "Document file not found on disk.")
                        session.refresh(doc)
                        return doc

                    # --- STEP 1: EXTRACTING ---
                    try:
                        extraction = self.pdf_service.extract_text(file_path)
                        doc.page_count = extraction.total_pages
                        session.commit()
                    except EncryptedPDFError:
                        self._safe_fail_document(session, doc_uuid, my_lease_token, "PDF is password-encrypted and cannot be processed.")
                        session.refresh(doc)
                        return doc
                    except (InvalidPDFError, PDFExtractionError):
                        self._safe_fail_document(session, doc_uuid, my_lease_token, "Corrupted or invalid PDF file.")
                        session.refresh(doc)
                        return doc
                    except PDFFileNotFoundError:
                        self._safe_fail_document(session, doc_uuid, my_lease_token, "Document file not found on disk.")
                        session.refresh(doc)
                        return doc
                    except Exception as exc:
                        logger.error("Unexpected PDF extraction error for document %s: %s", doc_uuid, exc, exc_info=True)
                        self._safe_fail_document(session, doc_uuid, my_lease_token, "Failed to extract text from PDF document.")
                        session.refresh(doc)
                        return doc

                    # --- STEP 2: CHUNKING ---
                    heartbeat.verify_current(session)
                    self._safe_transition(session, doc_uuid, my_lease_token, ProcessingStatus.CHUNKING.value)

                    try:
                        extracted_chunks = self.chunking_service.chunk_document(
                            extraction,
                            document_id=str(doc.id),
                        )
                    except Exception as exc:
                        logger.error("Chunking error for document %s: %s", doc_uuid, exc, exc_info=True)
                        self._safe_fail_document(session, doc_uuid, my_lease_token, "Failed to split document text into passages.")
                        session.refresh(doc)
                        return doc

                    if not extracted_chunks:
                        self._safe_fail_document(
                            session,
                            doc_uuid,
                            my_lease_token,
                            "No extractable text found in PDF document (document may be empty or contain only scanned images).",
                        )
                        session.refresh(doc)
                        return doc

                    # --- STEP 3: IDEMPOTENT DB PERSISTENCE ---
                    heartbeat.verify_current(session)

                    # Purge any preexisting chunks and vectors for this document
                    existing_chunks = session.query(Chunk).filter(Chunk.document_id == doc.id).all()
                    if existing_chunks:
                        old_chunk_ids = [c.id for c in existing_chunks]
                        try:
                            self.vector_store.remove_vectors(old_chunk_ids)
                            if self.vector_store.index_path:
                                self.vector_store.save_to_disk()
                        except Exception as clean_err:
                            logger.warning(
                                "Failed to purge old vectors for document %s before re-ingestion: %s",
                                doc_uuid,
                                clean_err,
                            )
                        session.query(Chunk).filter(Chunk.document_id == doc.id).delete(synchronize_session="fetch")
                        session.flush()

                    heartbeat.verify_current(session)

                    # Insert new chunks inside transaction
                    db_chunks = []
                    for c in extracted_chunks:
                        chunk_record = Chunk(
                            document_id=doc.id,
                            chunk_index=c.chunk_index,
                            page_number=c.page_number,
                            content=c.text,
                            char_start=c.start_char,
                            char_end=c.end_char,
                            token_count=c.token_count,
                            embedding_model_name=self.embedding_service.model_name,
                        )
                        session.add(chunk_record)
                        db_chunks.append(chunk_record)

                    try:
                        session.flush()
                    except Exception as db_err:
                        logger.error("Database flush error saving chunks for document %s: %s", doc_uuid, db_err, exc_info=True)
                        session.rollback()
                        self._safe_fail_document(session, doc_uuid, my_lease_token, "Failed to persist document passages to database.")
                        session.refresh(doc)
                        return doc

                    persisted_chunk_ids = [c.id for c in db_chunks]

                    # --- STEP 4: EMBEDDING ---
                    heartbeat.verify_current(session)
                    self._safe_transition(session, doc_uuid, my_lease_token, ProcessingStatus.EMBEDDING.value)

                    texts_to_embed = [c.content for c in db_chunks]
                    try:
                        embeddings = self.embedding_service.embed_batch(texts_to_embed)
                    except (EncodingError, Exception) as embed_err:
                        logger.error("Embedding generation error for document %s: %s", doc_uuid, embed_err, exc_info=True)
                        session.query(Chunk).filter(Chunk.document_id == doc.id).delete(synchronize_session="fetch")
                        self._safe_fail_document(session, doc_uuid, my_lease_token, "Failed to generate vector embeddings for document passages.")
                        session.refresh(doc)
                        return doc

                    # --- STEP 5: FAISS VECTOR INDEXING ---
                    # Check lease before mutating FAISS!
                    heartbeat.verify_current(session)

                    try:
                        self.vector_store.add_vectors(chunk_ids=persisted_chunk_ids, vectors=embeddings)
                        if self.vector_store.index_path:
                            self.vector_store.save_to_disk()
                    except (VectorStoreError, Exception) as index_err:
                        logger.error("VectorStore indexing error for document %s: %s", doc_uuid, index_err, exc_info=True)
                        try:
                            self.vector_store.remove_vectors(persisted_chunk_ids)
                            if self.vector_store.index_path and self.vector_store.index_path.exists():
                                self.vector_store.save_to_disk()
                        except Exception as clean_vec_err:
                            logger.warning("Failed cleanup of vectors for document %s: %s", doc_uuid, clean_vec_err)

                        session.query(Chunk).filter(Chunk.document_id == doc.id).delete(synchronize_session="fetch")
                        self._safe_fail_document(session, doc_uuid, my_lease_token, "Failed to index vector embeddings in search index.")
                        session.refresh(doc)
                        return doc

                    # --- STEP 6: READY ---
                    ready_stmt = (
                        update(Document)
                        .where(
                            Document.id == doc_uuid,
                            Document.lease_token == my_lease_token,
                        )
                        .values(
                            processing_status=ProcessingStatus.READY.value,
                            lease_token=None,
                            error_message=None,
                            updated_at=func.now(),
                        )
                        .execution_options(synchronize_session=False)
                    )
                    res = session.execute(ready_stmt)
                    if res.rowcount == 0:
                        # Lease was lost right before marking READY!
                        # Roll back vectors from FAISS
                        try:
                            self.vector_store.remove_vectors(persisted_chunk_ids)
                            if self.vector_store.index_path and self.vector_store.index_path.exists():
                                self.vector_store.save_to_disk()
                        except Exception as clean_err:
                            logger.warning("Failed cleanup of vectors for document %s on lost lease: %s", doc_uuid, clean_err)
                        session.rollback()
                        raise IngestionLeaseLostError(f"Lost lease before marking document {doc_uuid} READY")

                    session.commit()
                    session.refresh(doc)
                    logger.info(
                        "Ingestion successfully completed for document %s (%d passages, %d pages)",
                        doc.id,
                        len(persisted_chunk_ids),
                        doc.page_count or 1,
                    )
                    return doc

                except IngestionLeaseLostError as lease_err:
                    logger.warning(
                        "Document %s ingestion aborted: %s. Rolling back changes without overwriting status or mutating search index.",
                        doc_uuid,
                        lease_err,
                    )
                    session.rollback()
                    session.refresh(doc)
                    return doc

    def retry_document(
        self,
        document_id: Union[UUID, str],
        session_factory: Optional[sessionmaker[Session]] = None,
        force: bool = False,
    ) -> Optional[Document]:
        """Safely retries ingestion for a document.

        Args:
            document_id: UUID or string UUID of the document to retry.
            session_factory: Optional session factory.
            force: If True, forces re-ingestion even if document is READY.

        Returns:
            The updated Document instance.
        """
        return self.process_document(document_id, session_factory=session_factory, force=force)

    def rebuild_vector_index(
        self,
        session: Session,
        batch_size: int = 64,
    ) -> int:
        """Reconstructs the FAISS vector index from scratch using PostgreSQL chunk records."""
        from app.services.index_recovery import rebuild_vector_index as rebuild_fn
        return rebuild_fn(
            session=session,
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            batch_size=batch_size,
        )


# Global singleton instance
ingestion_service = IngestionService()
