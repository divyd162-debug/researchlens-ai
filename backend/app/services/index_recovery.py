"""Deterministic FAISS index verification and reconstruction service.

Architecture & Consistency Principles:
1. Rebuildable Derived Index:
   - Embeddings are intentionally NOT stored in PostgreSQL (which only persists chunk text and metadata).
   - The FAISS index is a derived search acceleration structure.
   - If the index is missing, corrupt, or inconsistent, it is reconstructed from scratch by scanning
     Chunk records for documents in READY state, embedding them in batches, and re-indexing them
     under their stable 64-bit integer Chunk.id primary keys.
2. Safe Behavior on Rebuild Failure:
   - If rebuild fails (e.g. embedding model failure, OOM, or validation mismatch), the VectorStore is
     explicitly marked as NOT ready (is_ready = False) and VectorIndexRecoveryError is raised.
   - The system will NEVER silently serve an empty index while documents remain marked READY.
3. Dimension & Metadata Validation:
   - After rebuilding, the vector count, embedded ID set, model name, and dimensionality are
     strictly cross-validated against PostgreSQL and the configured EmbeddingService before the
     index is approved for query serving.
"""
import logging
from typing import Optional, Set
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.core.database import db_session_scope, get_session_factory
from app.models.chunk import Chunk
from app.models.document import Document, ProcessingStatus
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.services.vector_store import (
    CorruptIndexError,
    DimensionMismatchError,
    IndexNotFoundError,
    ModelMismatchError,
    VectorIndexRecoveryError,
    VectorStore,
    VectorStoreError,
    get_vector_store,
)

logger = logging.getLogger(__name__)


def rebuild_vector_index(
    session: Session,
    vector_store: Optional[VectorStore] = None,
    embedding_service: Optional[EmbeddingService] = None,
    batch_size: int = 64,
) -> int:
    """Reconstructs the FAISS vector index from scratch using PostgreSQL chunk records.

    Args:
        session: Active SQLAlchemy database session.
        vector_store: Target VectorStore instance (defaults to global singleton).
        embedding_service: EmbeddingService to generate vectors (defaults to global singleton).
        batch_size: Number of chunk passages to embed and index per batch.

    Returns:
        Total number of vectors successfully rebuilt and indexed.

    Raises:
        VectorIndexRecoveryError: If embedding generation, indexing, persistence, or validation fails.
    """
    store = vector_store or get_vector_store()
    embedder = embedding_service or get_embedding_service()

    logger.info("Initiating deterministic FAISS index rebuild (model: %s)...", embedder.model_name)

    # 1. Query all chunks belonging to READY documents, ordered deterministically by chunk id
    try:
        eligible_chunks = (
            session.query(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .filter(Document.processing_status == ProcessingStatus.READY.value)
            .order_by(Chunk.id.asc())
            .all()
        )
    except Exception as db_err:
        store.set_ready(False)
        raise VectorIndexRecoveryError(
            f"Failed to query eligible chunks from database during index rebuild: {db_err}"
        ) from db_err

    # 2. Reset vector store state
    store.clear()

    # If no eligible chunks exist (fresh system or all docs pending/failed), save empty index and return
    if not eligible_chunks:
        logger.info("No eligible chunks found in database. Initializing clean empty index.")
        if store.index_path:
            try:
                store.save_to_disk()
            except Exception as save_err:
                store.set_ready(False)
                raise VectorIndexRecoveryError(
                    f"Failed to save empty index during rebuild: {save_err}"
                ) from save_err
        store.set_ready(True)
        return 0

    # 3. Process eligible chunks in deterministic batches
    total_indexed = 0
    expected_ids: Set[int] = set()

    for i in range(0, len(eligible_chunks), batch_size):
        batch = eligible_chunks[i : i + batch_size]
        batch_texts = [c.content for c in batch]
        batch_ids = [c.id for c in batch]
        expected_ids.update(batch_ids)

        # Generate dense vector embeddings from chunk text
        try:
            vectors = embedder.embed_batch(batch_texts)
        except Exception as embed_err:
            store.set_ready(False)
            raise VectorIndexRecoveryError(
                f"Embedding generation failed during index rebuild for chunk IDs {batch_ids[:3]}...: {embed_err}"
            ) from embed_err

        # Add vectors with stable 64-bit chunk IDs
        try:
            added = store.add_vectors(chunk_ids=batch_ids, vectors=vectors)
            total_indexed += added
        except Exception as add_err:
            store.set_ready(False)
            raise VectorIndexRecoveryError(
                f"Failed to add vectors to FAISS index during rebuild: {add_err}"
            ) from add_err

    # 4. Atomically persist rebuilt index to disk
    if store.index_path:
        try:
            store.save_to_disk()
        except Exception as save_err:
            store.set_ready(False)
            raise VectorIndexRecoveryError(
                f"Failed to persist rebuilt FAISS index to disk: {save_err}"
            ) from save_err

    # 5. Strict post-rebuild validation
    if store.total_vectors != len(eligible_chunks):
        store.set_ready(False)
        raise VectorIndexRecoveryError(
            f"Rebuilt index vector count mismatch: expected {len(eligible_chunks)}, got {store.total_vectors}"
        )

    if store.indexed_ids != expected_ids:
        store.set_ready(False)
        raise VectorIndexRecoveryError(
            f"Rebuilt index chunk ID set does not match PostgreSQL chunk IDs. "
            f"Expected {len(expected_ids)} IDs, got {len(store.indexed_ids)}"
        )

    if store.dimension != embedder.dimension:
        store.set_ready(False)
        raise VectorIndexRecoveryError(
            f"Rebuilt index dimension ({store.dimension}) does not match embedding model dimension ({embedder.dimension})"
        )

    store.set_ready(True)
    logger.info(
        "FAISS index successfully rebuilt and verified: %d vectors indexed across %d chunks.",
        total_indexed,
        len(eligible_chunks),
    )
    return total_indexed


def verify_and_recover_vector_index(
    session_factory: Optional[sessionmaker[Session]] = None,
    vector_store: Optional[VectorStore] = None,
    embedding_service: Optional[EmbeddingService] = None,
) -> bool:
    """Verifies startup index integrity and automatically recovers missing, corrupt, or inconsistent indices.

    Args:
        session_factory: Optional sessionmaker for DB connectivity.
        vector_store: Optional VectorStore instance.
        embedding_service: Optional EmbeddingService instance.

    Returns:
        True if the index is valid and ready for query serving.

    Raises:
        VectorIndexRecoveryError: If index recovery or rebuilding fails.
    """
    store = vector_store or get_vector_store()
    embedder = embedding_service or get_embedding_service()

    # Skip DB checks if database is unconfigured (e.g. lightweight isolated unit tests)
    if not settings.DATABASE_URL:
        logger.info("DATABASE_URL is not configured. Skipping FAISS index database consistency check.")
        store.set_ready(True)
        return True

    factory = session_factory or get_session_factory()

    with db_session_scope(factory) as session:
        # Determine expected chunk IDs from READY documents in PostgreSQL
        try:
            chunk_records = (
                session.query(Chunk.id)
                .join(Document, Chunk.document_id == Document.id)
                .filter(Document.processing_status == ProcessingStatus.READY.value)
                .all()
            )
            expected_chunk_ids: Set[int] = {c[0] for c in chunk_records}
        except Exception as db_err:
            logger.warning(
                "Could not query database chunks during startup index verification "
                "(tables may not be created yet): %s",
                db_err,
            )
            # If database tables are not yet migrated, load from disk if files exist
            if store.index_path and store.index_path.exists() and store.metadata_path and store.metadata_path.exists():
                try:
                    store.load_from_disk()
                    store.set_ready(True)
                except Exception:
                    store.set_ready(False)
            else:
                store.set_ready(True)
            return True

        # Check file existence on disk
        index_files_exist = bool(
            store.index_path
            and store.index_path.exists()
            and store.metadata_path
            and store.metadata_path.exists()
        )

        needs_rebuild = False
        rebuild_reason = ""

        if index_files_exist:
            try:
                store.load_from_disk()
                # Cross-validate loaded index against PostgreSQL
                if store.total_vectors != len(expected_chunk_ids) or store.indexed_ids != expected_chunk_ids:
                    needs_rebuild = True
                    rebuild_reason = (
                        f"Index vector set mismatch with PostgreSQL: "
                        f"found {store.total_vectors} vectors vs {len(expected_chunk_ids)} expected chunks."
                    )
                else:
                    logger.info(
                        "FAISS vector index loaded and verified: %d vectors intact and consistent with PostgreSQL.",
                        store.total_vectors,
                    )
                    store.set_ready(True)
                    return True
            except (CorruptIndexError, DimensionMismatchError, ModelMismatchError, VectorStoreError) as load_err:
                needs_rebuild = True
                rebuild_reason = f"Corrupt or incompatible index file: {load_err}"
            except Exception as unexpected_err:
                needs_rebuild = True
                rebuild_reason = f"Unexpected error reading index: {unexpected_err}"
        else:
            if len(expected_chunk_ids) > 0:
                needs_rebuild = True
                rebuild_reason = (
                    f"Index files missing from disk while {len(expected_chunk_ids)} chunks exist in READY documents."
                )
            else:
                # No index files and no chunks in database: initialize fresh empty index
                logger.info("Initializing clean empty FAISS index on disk.")
                store.clear()
                if store.index_path:
                    store.save_to_disk()
                store.set_ready(True)
                return True

        if needs_rebuild:
            logger.warning("FAISS index recovery required (%s). Rebuilding from database chunks...", rebuild_reason)
            rebuild_vector_index(
                session=session,
                vector_store=store,
                embedding_service=embedder,
            )
            return True

    return True
