"""FAISS-backed vector store for ResearchLens AI using IndexIDMap2 around IndexFlatIP."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Collection, List, Optional, Set, Tuple, Union
import faiss
import numpy as np
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Represents a ranked vector similarity search result."""
    chunk_id: int = Field(..., description="Stable 64-bit integer chunk ID")
    score: float = Field(..., description="Similarity score (inner product / cosine similarity, typically [-1.0, 1.0])")


# --- Custom Domain Exceptions ---

class VectorStoreError(Exception):
    """Base exception for all vector store failures."""
    pass


class DuplicateIDError(VectorStoreError):
    """Raised when attempting to insert an ID that already exists or is duplicated in the batch."""
    pass


class DimensionMismatchError(VectorStoreError):
    """Raised when a vector dimension does not match the configured index dimension."""
    pass


class InvalidVectorError(VectorStoreError):
    """Raised when a vector contains NaN, Inf, or has an invalid structure."""
    pass


class InvalidIDError(VectorStoreError):
    """Raised when an ID is not a positive 64-bit integer."""
    pass


class CorruptIndexError(VectorStoreError):
    """Raised when persisted index or metadata files are corrupt or unreadable."""
    pass


class ModelMismatchError(VectorStoreError):
    """Raised when attempting to load an index built with a different embedding model."""
    pass


class IndexNotFoundError(VectorStoreError):
    """Raised when a requested index file does not exist on disk."""
    pass


class VectorStoreNotReadyError(VectorStoreError):
    """Raised when querying or operating on an uninitialized, recovering, or corrupt vector store."""
    pass


class VectorIndexRecoveryError(VectorStoreError):
    """Raised when automatic or manual rebuilding of the FAISS index fails."""
    pass


# --- Vector Store Implementation ---

class VectorStore:
    """FAISS-backed vector store managing dense vector storage and similarity search.

    Architecture Details:
    1. Base Index: Uses faiss.IndexFlatIP for exact inner-product search. When input vectors
       are L2-normalized, inner product is mathematically identical to cosine similarity.
    2. ID Mapping: Wrapped with faiss.IndexIDMap2 to associate each vector with a persistent 64-bit
       integer chunk ID and support direct ID removal.
    3. Thread-Safe Mutations (Process-Local): All additions, removals, searches, and persistence calls
       are protected by an RLock. Note: This lock is process-local and does NOT protect against
       multiple OS processes writing to the same directory concurrently. Deployment requires a single
       API worker when sharing an index directory.
    4. Atomic Persistence & Consistency Checks: Saves index to disk via temporary files and atomic
       rename (os.replace). Validates vector counts and embedded IDs match metadata on load.
    5. Model & Dimension Verification: Stores metadata alongside index to prevent mixing embeddings
       across different models or dimensions.
    """

    DEFAULT_INDEX_FILENAME = "faiss_index.bin"
    DEFAULT_META_FILENAME = "faiss_metadata.json"
    CURRENT_FORMAT_VERSION = 1
    SUPPORTED_FORMAT_VERSIONS = [1]
    INDEX_TYPE = "IndexIDMap2<IndexFlatIP>"
    METRIC = "inner_product"
    REQUIRED_METADATA_KEYS = {
        "format_version",
        "index_type",
        "metric",
        "normalized",
        "dimension",
        "model_name",
        "total_vectors",
        "indexed_ids",
    }

    def __init__(
        self,
        dimension: int = 384,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_dir: Optional[Union[str, Path]] = None,
    ):
        """Initializes an in-memory vector store with optional persistence path.

        Args:
            dimension: Dimensionality of vector embeddings (e.g. 384 for all-MiniLM-L6-v2).
            model_name: Name of the embedding model generating these vectors.
            index_dir: Directory where the index and metadata should be persisted (e.g. data/).
        """
        if dimension <= 0:
            raise ValueError("dimension must be a positive integer.")
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string.")

        self._dimension: int = dimension
        self._model_name: str = model_name
        self._index_dir: Optional[Path] = Path(index_dir).resolve() if index_dir else None

        self._lock = threading.RLock()
        self._indexed_ids: Set[int] = set()
        self._is_ready: bool = True

        # Build initial empty index
        self._index: faiss.IndexIDMap2 = self._create_empty_index(self._dimension)

    @classmethod
    def _create_empty_index(cls, dimension: int) -> faiss.IndexIDMap2:
        """Instantiates a fresh in-memory IndexIDMap2 around IndexFlatIP."""
        flat_index = faiss.IndexFlatIP(dimension)
        return faiss.IndexIDMap2(flat_index)

    @property
    def is_ready(self) -> bool:
        """Returns True if the vector store is healthy and ready for queries."""
        with self._lock:
            return self._is_ready

    def set_ready(self, ready: bool) -> None:
        """Explicitly sets the vector store readiness status."""
        with self._lock:
            self._is_ready = bool(ready)

    @property
    def indexed_ids(self) -> Set[int]:
        """Returns a copy of the currently indexed 64-bit chunk IDs."""
        with self._lock:
            return set(self._indexed_ids)

    def verify_integrity(self, expected_chunk_ids: Optional[Set[int]] = None) -> bool:
        """Validates that internal FAISS structures and metadata match."""
        with self._lock:
            if not self._is_ready:
                return False
            if self._index.ntotal != len(self._indexed_ids):
                return False
            if expected_chunk_ids is not None and self._indexed_ids != expected_chunk_ids:
                return False
            return True

    @property
    def dimension(self) -> int:
        """Returns the configured vector embedding dimension."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Returns the configured embedding model name."""
        return self._model_name

    @property
    def total_vectors(self) -> int:
        """Returns the total number of vectors currently indexed."""
        with self._lock:
            return self._index.ntotal

    @property
    def index_path(self) -> Optional[Path]:
        """Returns the path to the binary index file if index_dir is configured."""
        if self._index_dir:
            return self._index_dir / self.DEFAULT_INDEX_FILENAME
        return None

    @property
    def metadata_path(self) -> Optional[Path]:
        """Returns the path to the metadata JSON file if index_dir is configured."""
        if self._index_dir:
            return self._index_dir / self.DEFAULT_META_FILENAME
        return None

    def _validate_ids(self, chunk_ids: List[int]) -> np.ndarray:
        """Validates that chunk IDs are valid, non-negative 64-bit integers and distinct."""
        if not isinstance(chunk_ids, list):
            raise TypeError(f"chunk_ids must be a list of integers, got {type(chunk_ids).__name__}")

        if len(chunk_ids) == 0:
            return np.empty(0, dtype=np.int64)

        seen_in_batch: Set[int] = set()
        for idx, cid in enumerate(chunk_ids):
            if not isinstance(cid, (int, np.integer)):
                raise InvalidIDError(f"ID at index {idx} must be an integer, got {type(cid).__name__}")
            if cid <= 0:
                raise InvalidIDError(f"ID at index {idx} must be a positive integer, got {cid}")
            if cid > 2**63 - 1:
                raise InvalidIDError(f"ID at index {idx} exceeds 64-bit signed integer maximum: {cid}")

            if cid in seen_in_batch:
                raise DuplicateIDError(f"Duplicate ID {cid} detected within the provided batch at index {idx}")
            seen_in_batch.add(cid)

            if cid in self._indexed_ids:
                raise DuplicateIDError(f"Chunk ID {cid} already exists in the vector store")

        return np.ascontiguousarray(chunk_ids, dtype=np.int64)

    def _validate_and_normalize_vectors(self, vectors: Union[np.ndarray, List[List[float]]]) -> np.ndarray:
        """Validates array dimensions, finiteness, and normalizes vectors to unit L2 norm."""
        try:
            arr = np.asarray(vectors, dtype=np.float32)
        except Exception as e:
            raise InvalidVectorError(f"Could not convert input vectors to float32 numpy array: {e}") from e

        if arr.ndim != 2:
            raise InvalidVectorError(f"Vectors must be a 2D array of shape (N, {self._dimension}), got ndim={arr.ndim}")

        if arr.shape[1] != self._dimension:
            raise DimensionMismatchError(
                f"Vector dimension mismatch: expected {self._dimension}, got {arr.shape[1]}"
            )

        if not np.isfinite(arr).all():
            raise InvalidVectorError("Input vectors contain non-finite values (NaN, +Inf, or -Inf).")

        # L2-normalization for cosine similarity ranking
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise InvalidVectorError("One or more input vectors have zero magnitude and cannot be normalized.")

        # Re-normalize if not already unit length
        if not np.allclose(norms, 1.0, atol=1e-4):
            arr = arr / norms

        return np.ascontiguousarray(arr, dtype=np.float32)

    def add_vectors(
        self,
        chunk_ids: List[int],
        vectors: Union[np.ndarray, List[List[float]]],
    ) -> int:
        """Adds dense vectors with corresponding unique chunk IDs to the index.

        Args:
            chunk_ids: List of positive 64-bit integer IDs.
            vectors: 2D array or list of lists of shape (len(chunk_ids), dimension).

        Returns:
            Number of vectors successfully added.

        Raises:
            TypeError / InvalidIDError: If IDs are invalid types or non-positive.
            DuplicateIDError: If an ID is repeated in the batch or already in the index.
            DimensionMismatchError: If vector dimension does not match index dimension.
            InvalidVectorError: If vectors contain NaN/Inf or row count mismatches ID count.
        """
        with self._lock:
            validated_ids = self._validate_ids(chunk_ids)
            if len(validated_ids) == 0:
                return 0

            validated_vecs = self._validate_and_normalize_vectors(vectors)

            if len(validated_ids) != validated_vecs.shape[0]:
                raise InvalidVectorError(
                    f"Number of IDs ({len(validated_ids)}) does not match number of vectors ({validated_vecs.shape[0]})"
                )

            # Insert into FAISS IndexIDMap2
            self._index.add_with_ids(validated_vecs, validated_ids)
            self._indexed_ids.update(chunk_ids)

            return len(validated_ids)

    def search(
        self,
        query_vector: Union[np.ndarray, List[float]],
        top_k: int = 5,
        allowed_ids: Optional[Collection[int]] = None,
    ) -> List[SearchResult]:
        """Performs nearest-neighbor search for a query vector and returns ranked chunk IDs.

        Args:
            query_vector: 1D array or list of floats of length self.dimension.
            top_k: Maximum number of nearest neighbors to return.
            allowed_ids: Optional collection of integer chunk IDs to restrict the search to
                using faiss.IDSelectorBatch. Resolves candidate dilution completely.

        Returns:
            Ranked list of SearchResult objects ordered by descending similarity score.

        Raises:
            TypeError / InvalidVectorError: If query_vector is malformed.
            DimensionMismatchError: If query_vector dimension does not match index dimension.
            ValueError: If top_k <= 0.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        with self._lock:
            if not self._is_ready:
                raise VectorStoreNotReadyError(
                    "FAISS vector store is not ready. Index recovery or rebuild may be required."
                )

            try:
                q_arr = np.asarray(query_vector, dtype=np.float32)
            except Exception as e:
                raise InvalidVectorError(f"Could not convert query vector to float32: {e}") from e

            if q_arr.ndim == 1:
                q_arr = np.expand_dims(q_arr, axis=0)

            if q_arr.shape != (1, self._dimension):
                raise DimensionMismatchError(
                    f"Query vector shape mismatch: expected (1, {self._dimension}), got {q_arr.shape}"
                )

            if not np.isfinite(q_arr).all():
                raise InvalidVectorError("Query vector contains non-finite values (NaN or Inf).")

            if self._index.ntotal == 0:
                return []

            # If scoped allowed_ids provided, pre-filter to only those existing in index
            search_params = None
            max_available = self._index.ntotal
            if allowed_ids is not None:
                if len(allowed_ids) == 0:
                    return []
                valid_allowed = [int(cid) for cid in allowed_ids if int(cid) in self._indexed_ids]
                if not valid_allowed:
                    return []
                id_arr = np.ascontiguousarray(valid_allowed, dtype=np.int64)
                sel = faiss.IDSelectorBatch(id_arr)
                search_params = faiss.SearchParameters(sel=sel)
                max_available = len(valid_allowed)

            # Normalize query vector for cosine similarity
            q_norm = np.linalg.norm(q_arr)
            if q_norm == 0:
                raise InvalidVectorError("Query vector has zero magnitude and cannot be normalized.")

            if not np.isclose(q_norm, 1.0, atol=1e-4):
                q_arr = q_arr / q_norm

            q_arr = np.ascontiguousarray(q_arr, dtype=np.float32)

            k = min(top_k, max_available)
            if search_params is not None:
                scores, ids = self._index.search(q_arr, k, params=search_params)
            else:
                scores, ids = self._index.search(q_arr, k)

            results: List[SearchResult] = []
            for score, cid in zip(scores[0], ids[0]):
                if cid != -1:  # FAISS returns -1 for unfilled neighbor slots
                    results.append(SearchResult(chunk_id=int(cid), score=float(score)))

            return results

    def remove_ids(self, chunk_ids: List[int]) -> int:
        """Removes the specified chunk IDs and their vectors from the index.

        Args:
            chunk_ids: List of integer IDs to remove.

        Returns:
            Number of vectors successfully removed.
        """
        with self._lock:
            if not chunk_ids:
                return 0

            # Filter for IDs that actually exist in the index
            ids_to_remove = [cid for cid in chunk_ids if cid in self._indexed_ids]
            if not ids_to_remove:
                return 0

            id_arr = np.ascontiguousarray(ids_to_remove, dtype=np.int64)
            selector = faiss.IDSelectorArray(len(id_arr), faiss.swig_ptr(id_arr))
            removed_count = self._index.remove_ids(selector)

            for cid in ids_to_remove:
                self._indexed_ids.discard(cid)

            return removed_count

    # Semantic alias
    remove_vectors = remove_ids

    def contains_id(self, chunk_id: int) -> bool:
        """Returns True if the specified chunk ID is indexed."""
        with self._lock:
            return chunk_id in self._indexed_ids

    def clear(self) -> None:
        """Resets the vector store to an empty state."""
        with self._lock:
            self._index = self._create_empty_index(self._dimension)
            self._indexed_ids.clear()

    # --- Persistence & Atomic Reloading ---

    def save_to_disk(self) -> None:
        """Safely saves the FAISS index and metadata to disk using atomic replacement.

        Raises:
            ValueError: If no index directory was configured on this instance.
            VectorStoreError: If writing to disk fails.
        """
        with self._lock:
            if not self._index_dir:
                raise ValueError("Cannot save index: no index_dir configured on VectorStore instance.")

            self._index_dir.mkdir(parents=True, exist_ok=True)
            index_file = self._index_dir / self.DEFAULT_INDEX_FILENAME
            meta_file = self._index_dir / self.DEFAULT_META_FILENAME

            # Atomic write pattern: write to temporary files in the same directory, then rename
            tmp_idx_path: Optional[Path] = None
            tmp_meta_path: Optional[Path] = None

            try:
                with tempfile.NamedTemporaryFile(dir=self._index_dir, delete=False, suffix=".tmp") as tmp_idx:
                    tmp_idx_path = Path(tmp_idx.name)

                with tempfile.NamedTemporaryFile(dir=self._index_dir, delete=False, suffix=".tmp", mode="w") as tmp_meta:
                    tmp_meta_path = Path(tmp_meta.name)

                # Write FAISS binary
                faiss.write_index(self._index, str(tmp_idx_path))

                # Write JSON metadata
                meta = {
                    "format_version": self.CURRENT_FORMAT_VERSION,
                    "index_type": self.INDEX_TYPE,
                    "metric": self.METRIC,
                    "normalized": True,
                    "dimension": self._dimension,
                    "model_name": self._model_name,
                    "total_vectors": self.total_vectors,
                    "indexed_ids": sorted(list(self._indexed_ids)),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                with open(tmp_meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)

                # Atomic replacement
                os.replace(tmp_idx_path, index_file)
                os.replace(tmp_meta_path, meta_file)

            except Exception as e:
                # Cleanup leftover temp files on error
                if tmp_idx_path and tmp_idx_path.exists():
                    tmp_idx_path.unlink()
                if tmp_meta_path and tmp_meta_path.exists():
                    tmp_meta_path.unlink()
                raise VectorStoreError(f"Failed to persist vector index atomically: {e}") from e

    def load_from_disk(self) -> None:
        """Loads a persisted FAISS index and metadata from disk, validating compatibility and consistency.

        Raises:
            ValueError: If no index directory was configured on this instance.
            IndexNotFoundError: If the index file or metadata file is missing.
            CorruptIndexError: If the index file or metadata JSON is corrupt, missing required
                keys, has an unsupported format version, or exhibits state mismatches.
            DimensionMismatchError: If the persisted index dimension does not match.
            ModelMismatchError: If the persisted model name does not match.
        """
        with self._lock:
            if not self._index_dir:
                raise ValueError("Cannot load index: no index_dir configured on VectorStore instance.")

            index_file = self._index_dir / self.DEFAULT_INDEX_FILENAME
            meta_file = self._index_dir / self.DEFAULT_META_FILENAME

            if not index_file.exists() or not meta_file.exists():
                raise IndexNotFoundError(f"Index or metadata file not found in directory: {self._index_dir}")

            # 1. Read and validate metadata
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                raise CorruptIndexError(f"Failed to parse metadata JSON at '{meta_file}': {e}") from e

            # Validate required metadata keys
            missing_keys = self.REQUIRED_METADATA_KEYS - set(meta.keys())
            if missing_keys:
                raise CorruptIndexError(
                    f"Metadata file at '{meta_file}' is missing required key(s): {sorted(list(missing_keys))}"
                )

            # Validate format version
            format_version = meta["format_version"]
            if format_version not in self.SUPPORTED_FORMAT_VERSIONS:
                raise CorruptIndexError(
                    f"Unsupported index format version '{format_version}'. "
                    f"Supported versions: {self.SUPPORTED_FORMAT_VERSIONS}"
                )

            # Validate index structure and metrics
            if meta["index_type"] != self.INDEX_TYPE:
                raise CorruptIndexError(
                    f"Unsupported index type '{meta['index_type']}', expected '{self.INDEX_TYPE}'"
                )

            if meta["metric"] != self.METRIC:
                raise CorruptIndexError(
                    f"Unsupported metric '{meta['metric']}', expected '{self.METRIC}'"
                )

            if not meta["normalized"]:
                raise CorruptIndexError("Persisted metadata indicates vectors are not normalized.")

            persisted_dim = meta["dimension"]
            if persisted_dim != self._dimension:
                raise DimensionMismatchError(
                    f"Persisted index dimension ({persisted_dim}) does not match configured dimension ({self._dimension})"
                )

            persisted_model = meta["model_name"]
            if persisted_model != self._model_name:
                raise ModelMismatchError(
                    f"Persisted index model '{persisted_model}' does not match configured model '{self._model_name}'"
                )

            # 2. Read FAISS index
            try:
                raw_loaded = faiss.read_index(str(index_file))
            except Exception as e:
                raise CorruptIndexError(f"Failed to deserialize FAISS index from '{index_file}': {e}") from e

            # Validate that the loaded index has correct dimension and structure
            if raw_loaded.d != self._dimension:
                raise DimensionMismatchError(
                    f"Deserialized FAISS index has dimension {raw_loaded.d}, expected {self._dimension}"
                )

            if not isinstance(raw_loaded, faiss.IndexIDMap2):
                raise CorruptIndexError(
                    f"Deserialized FAISS index is not an IndexIDMap2 (got {type(raw_loaded).__name__})"
                )

            # 3. Cross-validate consistency between FAISS index and metadata
            if raw_loaded.ntotal != meta["total_vectors"]:
                raise CorruptIndexError(
                    f"Index vector count mismatch: FAISS binary has {raw_loaded.ntotal} vectors, "
                    f"but metadata records {meta['total_vectors']}"
                )

            faiss_ids = set(faiss.vector_to_array(raw_loaded.id_map).tolist())
            meta_ids = set(meta["indexed_ids"])
            if faiss_ids != meta_ids:
                raise CorruptIndexError(
                    f"Index ID set mismatch between FAISS index and metadata: "
                    f"found {len(faiss_ids)} IDs in FAISS vs {len(meta_ids)} in metadata"
                )

            self._index = raw_loaded
            self._indexed_ids = faiss_ids
            self._is_ready = True


_vector_store_instance: Optional[VectorStore] = None
_vector_store_singleton_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    """Returns the shared VectorStore singleton instance, loading from disk if an index exists."""
    global _vector_store_instance
    if _vector_store_instance is None:
        with _vector_store_singleton_lock:
            if _vector_store_instance is None:
                from app.core.config import settings
                import logging

                store = VectorStore(
                    dimension=384,
                    model_name=settings.EMBEDDING_MODEL_NAME,
                    index_dir=settings.VECTOR_INDEX_DIR,
                )
                if (
                    store.index_path
                    and store.index_path.exists()
                    and store.metadata_path
                    and store.metadata_path.exists()
                ):
                    try:
                        store.load_from_disk()
                    except Exception as err:
                        logging.getLogger(__name__).warning(
                            "Failed to load existing vector store from disk: %s", err
                        )
                        store.set_ready(False)
                _vector_store_instance = store
    return _vector_store_instance


def reset_vector_store() -> None:
    """Resets the cached VectorStore singleton (useful for test fixtures)."""
    global _vector_store_instance
    with _vector_store_singleton_lock:
        _vector_store_instance = None

