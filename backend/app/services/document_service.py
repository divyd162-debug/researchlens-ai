"""Service layer for Document management, validation, secure storage, and tenant isolation."""
import hashlib
import logging
from pathlib import Path
import time
from typing import Any, Optional, Union
from uuid import UUID, uuid4

import pymupdf
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.chat import ChatSession, ChatSessionDocument
from app.models.document import Document, ProcessingStatus
from app.schemas.document import DocumentListResponse, DocumentResponse

logger = logging.getLogger(__name__)


class DocumentServiceError(Exception):
    """Base exception for document service failures."""
    pass


class DocumentNotFoundError(DocumentServiceError):
    """Raised when a document does not exist or does not belong to the user."""
    pass


class DuplicateDocumentError(DocumentServiceError):
    """Raised when an identical document has already been uploaded by this user."""
    pass


class InvalidDocumentError(DocumentServiceError):
    """Raised when uploaded file fails validation (type, structure, corruption)."""
    pass


class DocumentSizeLimitExceededError(DocumentServiceError):
    """Raised when uploaded file exceeds allowed size limits."""
    pass


class DocumentService:
    """Provides isolated document management operations for multi-tenant users."""

    PDF_MAGIC_BYTES = b"%PDF-"
    CHUNK_SIZE = 64 * 1024  # 64 KB streaming buffer

    def upload_document(
        self,
        db: Session,
        owner_id: UUID,
        file: Any,
        title: Optional[str] = None,
        upload_dir: Optional[str] = None,
        max_size_bytes: Optional[int] = None,
    ) -> DocumentResponse:
        """Validates, securely stores, and creates a Document record.

        Args:
            db: Active database session.
            owner_id: Authenticated user ID (tenant owner).
            file: FastAPI UploadFile or file-like object with .filename and .read.
            title: Optional user-supplied document title.
            upload_dir: Target directory path for file persistence.
            max_size_bytes: Maximum allowed file size in bytes.

        Returns:
            DocumentResponse with safe metadata.

        Raises:
            InvalidDocumentError: If file is missing, empty, not a PDF, encrypted, or corrupted.
            DocumentSizeLimitExceededError: If file exceeds size limit.
            DuplicateDocumentError: If an identical document is already owned by this user.
            DocumentServiceError: If filesystem promotion fails.
        """
        raw_filename = getattr(file, "filename", None) or ""
        # Strip null bytes and sanitize filename
        sanitized_filename = raw_filename.replace("\x00", "").strip()
        basename = Path(sanitized_filename).name

        if not basename:
            raise InvalidDocumentError("A valid filename is required.")

        if not basename.lower().endswith(".pdf"):
            raise InvalidDocumentError("Only PDF documents (.pdf) are supported.")

        if len(basename) > 512:
            basename = basename[:508] + ".pdf"

        # Sanitize title if provided
        clean_title: Optional[str] = None
        if title is not None:
            clean_title = title.strip()
            if not clean_title:
                clean_title = None
            elif len(clean_title) > 512:
                clean_title = clean_title[:512]

        dest_dir = Path(upload_dir or settings.UPLOAD_DIR).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        doc_id = uuid4()
        temp_path = dest_dir / f"{doc_id}.tmp"
        final_path = dest_dir / f"{doc_id}.pdf"

        # Traversal safety check
        if not final_path.resolve().is_relative_to(dest_dir):
            raise InvalidDocumentError("Invalid document storage path.")

        max_limit = max_size_bytes or settings.MAX_UPLOAD_SIZE_BYTES
        hasher = hashlib.sha256()
        total_bytes = 0

        # Helper to read chunk from UploadFile or raw file
        def _read_chunk(n: int) -> bytes:
            if hasattr(file, "file") and hasattr(file.file, "read"):
                return file.file.read(n)
            elif hasattr(file, "read"):
                res = file.read(n)
                if hasattr(res, "__await__"):
                    raise TypeError("Synchronous file read expected.")
                return res
            raise InvalidDocumentError("Uploaded file object is not readable.")

        try:
            with open(temp_path, "wb") as f_out:
                first_chunk = _read_chunk(self.CHUNK_SIZE)
                if not first_chunk:
                    raise InvalidDocumentError("Uploaded file is empty (0 bytes).")

                if self.PDF_MAGIC_BYTES not in first_chunk[:1024]:
                    raise InvalidDocumentError("File does not contain valid PDF magic bytes ('%PDF-').")

                total_bytes += len(first_chunk)
                if total_bytes > max_limit:
                    raise DocumentSizeLimitExceededError(
                        f"File size exceeds maximum allowed limit of {max_limit} bytes."
                    )

                hasher.update(first_chunk)
                f_out.write(first_chunk)

                while True:
                    chunk = _read_chunk(self.CHUNK_SIZE)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > max_limit:
                        raise DocumentSizeLimitExceededError(
                            f"File size exceeds maximum allowed limit of {max_limit} bytes."
                        )
                    hasher.update(chunk)
                    f_out.write(chunk)

        except Exception:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception as cleanup_err:
                logger.warning("Failed to clean up temporary file '%s': %s", temp_path, cleanup_err)
            raise

        # Structural inspection using PyMuPDF (without full text extraction)
        try:
            with pymupdf.open(temp_path) as doc:
                if doc.is_encrypted and doc.needs_pass:
                    raise InvalidDocumentError("PDF is password-encrypted and cannot be processed.")
                if doc.page_count == 0:
                    raise InvalidDocumentError("PDF contains 0 pages.")
                if doc.page_count > settings.MAX_DOCUMENT_PAGES:
                    raise InvalidDocumentError(
                        f"Document exceeds maximum allowed limit of {settings.MAX_DOCUMENT_PAGES} pages (got {doc.page_count})."
                    )
                page_count = doc.page_count
        except (InvalidDocumentError, DocumentSizeLimitExceededError):
            try:
                temp_path.unlink(missing_ok=True)
            except Exception as cleanup_err:
                logger.warning("Failed to clean up temporary file '%s': %s", temp_path, cleanup_err)
            raise
        except pymupdf.FileDataError as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception as cleanup_err:
                logger.warning("Failed to clean up temporary file '%s': %s", temp_path, cleanup_err)
            # Never interpolate raw PyMuPDF exception strings into user-facing message to prevent path leakage
            raise InvalidDocumentError("Corrupted or invalid PDF file.") from exc
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception as cleanup_err:
                logger.warning("Failed to clean up temporary file '%s': %s", temp_path, cleanup_err)
            raise InvalidDocumentError("Unable to parse PDF.") from exc

        file_hash = hasher.hexdigest()

        # Duplicate check per owner (fast check before file rename)
        existing = (
            db.query(Document)
            .filter(
                Document.owner_id == owner_id,
                Document.file_hash_sha256 == file_hash,
            )
            .first()
        )
        if existing is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception as cleanup_err:
                logger.warning("Failed to clean up temporary file '%s': %s", temp_path, cleanup_err)
            raise DuplicateDocumentError(
                f"A document with identical content has already been uploaded: '{existing.filename}'."
            )

        # Atomic promote from tmp to final .pdf with robust error handling
        try:
            temp_path.rename(final_path)
        except OSError as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception as cleanup_err:
                logger.warning("Failed to clean up temporary file '%s': %s", temp_path, cleanup_err)
            try:
                final_path.unlink(missing_ok=True)
            except Exception as cleanup_err:
                logger.warning("Failed to clean up destination file '%s': %s", final_path, cleanup_err)
            raise DocumentServiceError(f"Failed to store uploaded document file: {exc}") from exc

        doc_record = Document(
            id=doc_id,
            owner_id=owner_id,
            filename=basename,
            title=clean_title,
            file_path=str(final_path),
            file_hash_sha256=file_hash,
            file_size_bytes=total_bytes,
            page_count=page_count,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db.add(doc_record)

        try:
            db.commit()
            db.refresh(doc_record)
        except IntegrityError as exc:
            db.rollback()
            try:
                final_path.unlink(missing_ok=True)
            except Exception as cleanup_err:
                logger.warning("Failed to clean up stored file '%s' after DB rollback: %s", final_path, cleanup_err)

            err_str = (str(exc.orig) if hasattr(exc, "orig") and exc.orig else str(exc)).lower()
            if (
                "uq_document_owner_hash" in err_str
                or ("owner_id" in err_str and "file_hash" in err_str)
                or "unique constraint failed: documents.owner_id, documents.file_hash_sha256" in err_str
            ):
                raise DuplicateDocumentError(
                    "A document with identical content has already been uploaded."
                ) from exc
            # Re-raise unrelated integrity errors (e.g. foreign key or check constraint violations)
            raise
        except Exception:
            db.rollback()
            try:
                final_path.unlink(missing_ok=True)
            except Exception as cleanup_err:
                logger.warning("Failed to clean up stored file '%s' after DB rollback: %s", final_path, cleanup_err)
            raise

        return DocumentResponse.model_validate(doc_record)

    def list_documents(
        self,
        db: Session,
        owner_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> DocumentListResponse:
        """Returns paginated documents owned by owner_id in deterministic order."""
        base_query = db.query(Document).filter(Document.owner_id == owner_id)
        total = base_query.count()

        documents = (
            base_query.order_by(Document.created_at.desc(), Document.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return DocumentListResponse(
            items=[DocumentResponse.model_validate(d) for d in documents],
            total=total,
            limit=limit,
            offset=offset,
        )

    def get_document(
        self,
        db: Session,
        owner_id: UUID,
        document_id: UUID,
    ) -> DocumentResponse:
        """Retrieves a single document scoped strictly to owner_id."""
        doc = (
            db.query(Document)
            .filter(Document.owner_id == owner_id, Document.id == document_id)
            .first()
        )
        if doc is None:
            raise DocumentNotFoundError("Document not found.")

        return DocumentResponse.model_validate(doc)

    def delete_document(
        self,
        db: Session,
        owner_id: UUID,
        document_id: UUID,
    ) -> None:
        """Deletes a document record, purges its vectors from FAISS, and unlinks its stored file upon commit.
        
        Architecture Note on Recoverable Deletion & Concurrency:
        1. Tenant Isolation & Row Locking:
           Queries the document strictly scoped to owner_id with .with_for_update() and .populate_existing(),
           preventing concurrent ingestion or update races while ensuring foreign-owned documents return 404.
        2. Ingestion Fencing:
           Deleting the database record immediately fences out any background worker running an active lease.
           When the worker attempts to renew its lease heartbeat, transition status, or mark the document READY,
           the document will be missing from the database, raising IngestionLeaseLostError and triggering the
           worker's rollback to discard its changes without persisting new chunks or vectors.
        3. Scope Constraint & Historical Integrity Safety:
           Before deleting the document record, archives all active chat sessions referencing this document.
           This satisfies the check constraint ck_chat_session_scope_targets when foreign key target_document_id
           transitions to NULL, preserving complete chat history, messages, and citation snapshots.
        4. Explicit Post-Commit Orphan Cleanup Strategy:
           This operation uses a database-first deletion strategy: the database record and cascaded chunks /
           collection associations are deleted and committed first.
           - Vectors: Chunks are purged from the FAISS vector store using the chunk IDs collected prior to deletion.
             If vector purging fails, an explicit ERROR is logged with the target chunk IDs and document metadata
             so reconciliation can recover the vector store without leaving zombie database rows.
           - Files: The stored PDF is unlinked on disk. If filesystem removal fails, an explicit ERROR is logged
             with the target file path and metadata for disk reconciliation.
        """
        doc = (
            db.query(Document)
            .filter(Document.owner_id == owner_id, Document.id == document_id)
            .with_for_update()
            .populate_existing()
            .first()
        )
        if doc is None:
            raise DocumentNotFoundError("Document not found.")

        file_path_to_delete = Path(doc.file_path) if doc.file_path else None
        chunk_ids_to_remove = [c.id for c in doc.chunks]

        # Safely archive any active single-document chat sessions referencing this document before deletion
        # to satisfy ck_chat_session_scope_targets check constraint.
        active_sessions = (
            db.query(ChatSession)
            .filter(
                ChatSession.target_document_id == doc.id,
                ChatSession.is_archived.is_(False),
            )
            .all()
        )
        for session in active_sessions:
            session.archive()

        # For multi-document scoped sessions:
        # If this is the last remaining document in the session, archive the session.
        # Otherwise, the cascading delete of chat_session_documents safely leaves remaining sources.
        multi_doc_sessions = (
            db.query(ChatSession)
            .join(ChatSessionDocument, ChatSessionDocument.chat_session_id == ChatSession.id)
            .filter(
                ChatSessionDocument.document_id == doc.id,
                ChatSession.is_archived.is_(False),
            )
            .all()
        )
        for session in multi_doc_sessions:
            doc_count = (
                db.query(ChatSessionDocument)
                .filter(ChatSessionDocument.chat_session_id == session.id)
                .count()
            )
            if doc_count <= 1:
                session.archive()

        db.delete(doc)
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        # Purge associated vectors from FAISS index if indexed
        if chunk_ids_to_remove:
            try:
                from app.services.vector_store import get_vector_store

                store = get_vector_store()
                store.remove_vectors(chunk_ids_to_remove)
                if store.index_path:
                    store.save_to_disk()
            except Exception as vec_err:
                logger.error(
                    "Orphan vector cleanup required: failed to purge vectors %s for deleted document ID %s (owner: %s): %s",
                    chunk_ids_to_remove,
                    document_id,
                    owner_id,
                    vec_err,
                )

        # Only purge file from disk after successful database commit
        if file_path_to_delete and file_path_to_delete.exists():
            try:
                file_path_to_delete.unlink(missing_ok=True)
            except Exception as exc:
                logger.error(
                    "Orphan file recovery required: failed to unlink document file '%s' for deleted document ID %s (owner: %s): %s",
                    file_path_to_delete,
                    document_id,
                    owner_id,
                    exc,
                )


# Singleton instance
document_service = DocumentService()


def cleanup_stale_temp_files(
    upload_dir: Optional[Union[str, Path]] = None,
    max_age_seconds: int = 3600,
) -> int:
    """Scans the upload directory and safely cleans up stale temporary upload files.

    Rules:
    - Only targets regular files ending with the '.tmp' suffix.
    - Never follows symlinks.
    - Only removes files whose last modification time (mtime) is older than max_age_seconds (default: 1 hour).
    - Preserves all .pdf files, .gitkeep, subdirectories, and any unrelated non-.tmp files.
    - Safely catches any filesystem or permission errors per file, logging them without halting or crashing.
    - Returns the number of successfully unlinked stale temporary files.
    """
    target_dir = Path(upload_dir) if upload_dir is not None else Path(settings.UPLOAD_DIR)
    if not target_dir.exists() or not target_dir.is_dir():
        logger.debug("Upload directory '%s' does not exist or is not a directory. Skipping stale cleanup.", target_dir)
        return 0

    now = time.time()
    purged_count = 0

    try:
        entries = list(target_dir.iterdir())
    except (OSError, PermissionError) as exc:
        logger.warning("Could not read directory '%s' during stale temp cleanup: %s", target_dir, exc)
        return 0

    for entry in entries:
        try:
            # Strictly ignore symlinks without following them
            if entry.is_symlink():
                continue

            # Only inspect regular files ending with .tmp
            if not entry.is_file() or not entry.name.endswith(".tmp"):
                continue

            st = entry.stat()
            age_seconds = now - st.st_mtime
            if age_seconds > max_age_seconds:
                entry.unlink()
                purged_count += 1
                logger.info("Purged stale temporary upload file '%s' (age: %.1f hours).", entry.name, age_seconds / 3600)
        except (OSError, PermissionError) as err:
            logger.warning("Failed to inspect or remove temporary file '%s': %s", entry.name, err)

    if purged_count > 0:
        logger.info("Stale temporary file cleanup completed: purged %d file(s) from '%s'.", purged_count, target_dir)

    return purged_count
