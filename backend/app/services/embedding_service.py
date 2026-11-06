"""Local vector embedding service using Sentence Transformers for ResearchLens AI."""
import math
import os
import threading
from typing import List, Optional, Union
import numpy as np

def configure_hf_offline_mode() -> None:
    """Configures HuggingFace and Transformers offline environment settings.

    Rules:
    - If HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE is explicitly configured in the environment
      (e.g. "1", "0", "true", "false"), that explicit configuration is strictly preserved.
    - In test/testing environments (ENVIRONMENT in ('test', 'testing')), offline mode is
      enabled by default to prevent unit tests from making unexpected outbound network requests.
    - In development and production environments, offline mode is NOT forced by default,
      allowing SentenceTransformers to download model weights on initial launch if not cached.
    """
    env = (os.getenv("ENVIRONMENT") or "").strip().lower()
    if env in ("test", "testing"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


configure_hf_offline_mode()


class EmbeddingServiceError(Exception):
    """Base exception for all embedding service failures."""
    pass


class ModelLoadError(EmbeddingServiceError):
    """Raised when the embedding model fails to initialize or load weights."""
    pass


class EncodingError(EmbeddingServiceError):
    """Raised when vector generation or encoding fails."""
    pass


class EmbeddingService:
    """Service providing local, dense vector embeddings using Sentence Transformers.

    Design Principles:
    1. Local & Free: Uses sentence-transformers (default: all-MiniLM-L6-v2) for offline execution.
    2. Lazy Initialization: Model weights are not loaded into memory until the first call to
       embed_text, embed_batch, dimension, or max_seq_length.
    3. Thread-Safe Single Instance: Utilizes a thread lock so model weights are loaded exactly
       once per service instance.
    4. Predictable Output: Guarantees finite (no NaN/Inf), L2-normalized 1D or 2D float vectors
       matching the model's native dimension.
    5. Strict Input Validation: Non-strings and empty/whitespace texts raise clear exceptions
       rather than silently generating degenerate vectors.
    """

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        """Initializes the embedding service configuration without loading model weights.

        Args:
            model_name: HuggingFace model identifier or local path.
            device: Computing device ('cpu', 'cuda', 'mps', or None for auto-detection).
            normalize_embeddings: When True, output vectors are L2-normalized (unit length).
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string.")

        self._model_name: str = model_name
        self._device: Optional[str] = device
        self._normalize_embeddings: bool = normalize_embeddings

        self._model = None
        self._dimension: Optional[int] = None
        self._max_seq_length: Optional[int] = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        """Returns the configured model name."""
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """Returns True if the underlying model is currently loaded in memory."""
        return self._model is not None

    def _get_model(self):
        """Thread-safe lazy loader for the SentenceTransformer model."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer

                        loaded_model = SentenceTransformer(
                            self._model_name,
                            device=self._device,
                        )
                        self._model = loaded_model

                        # Dynamically inspect embedding dimension from loaded model
                        dim = loaded_model.get_sentence_embedding_dimension()
                        if dim is None or dim <= 0:
                            raise ModelLoadError(
                                f"Model '{self._model_name}' reported invalid dimension: {dim}"
                            )
                        self._dimension = int(dim)

                        # Inspect tokenizer maximum sequence length
                        self._max_seq_length = getattr(loaded_model, "max_seq_length", None)

                    except Exception as e:
                        if isinstance(e, ModelLoadError):
                            raise
                        raise ModelLoadError(
                            f"Failed to load SentenceTransformer model '{self._model_name}': {e}"
                        ) from e

        return self._model

    @property
    def dimension(self) -> int:
        """Returns the vector embedding dimension dynamically retrieved from the loaded model.

        Example: all-MiniLM-L6-v2 produces 384-dimensional vectors.
        """
        if self._dimension is None:
            self._get_model()
        return self._dimension  # type: ignore

    @property
    def max_seq_length(self) -> Optional[int]:
        """Returns the maximum sequence length (in tokenizer tokens) accepted by the model.

        Truncation Risk Documentation:
        - Transformer models have a hard limit on input sequence length (typically 256 or 512 tokens).
        - For all-MiniLM-L6-v2, max_seq_length is 256 wordpiece tokens.
        - Text exceeding this limit is silently truncated by the tokenizer at inference time.
        - Upstream chunking strategies should keep chunk lengths aligned with this threshold to prevent
          information loss.
        """
        if self._max_seq_length is None:
            self._get_model()
        return self._max_seq_length

    def _validate_single_text(self, text: Union[str, any], param_name: str = "text") -> str:
        """Validates that text is a non-empty, non-whitespace string."""
        if not isinstance(text, str):
            raise TypeError(f"{param_name} must be a string, got {type(text).__name__}")
        cleaned = text.strip()
        if not cleaned:
            raise ValueError(f"{param_name} cannot be empty or whitespace-only")
        return cleaned

    def embed_text(self, text: str) -> List[float]:
        """Generates an L2-normalized vector embedding for a single text string.

        Args:
            text: Non-empty string to embed (e.g. user search query or passage).

        Returns:
            List of Python floats representing the dense vector of length self.dimension.

        Raises:
            TypeError: If text is not a string.
            ValueError: If text is empty or whitespace-only.
            EncodingError: If vector generation fails or yields non-finite values.
        """
        self._validate_single_text(text, param_name="text")
        batch_result = self.embed_batch([text])
        return batch_result[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates L2-normalized vector embeddings for a list of text strings.

        Args:
            texts: List of non-empty strings to embed.

        Returns:
            List of vectors, where each vector is a List[float] of length self.dimension.
            If texts is an empty list ([]), returns an empty list ([]).

        Raises:
            TypeError: If texts is not a list or contains non-string items.
            ValueError: If any item in texts is empty or whitespace-only.
            EncodingError: If vector generation fails or yields non-finite values.
        """
        if not isinstance(texts, list):
            raise TypeError(f"texts must be a list of strings, got {type(texts).__name__}")

        # Empty batch returns empty list without calling model
        if len(texts) == 0:
            return []

        # Validate all items in batch
        validated_texts: List[str] = []
        for idx, item in enumerate(texts):
            valid_item = self._validate_single_text(item, param_name=f"texts[{idx}]")
            validated_texts.append(valid_item)

        model = self._get_model()

        try:
            raw_embeddings = model.encode(
                validated_texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=self._normalize_embeddings,
                convert_to_numpy=True,
            )
        except Exception as e:
            raise EncodingError(f"Failed to generate embeddings: {e}") from e

        # Ensure 2D numpy array
        embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        expected_rows = len(validated_texts)
        expected_cols = self.dimension

        if embeddings.shape != (expected_rows, expected_cols):
            raise EncodingError(
                f"Generated embedding shape mismatch. Expected ({expected_rows}, {expected_cols}), "
                f"got {embeddings.shape}"
            )

        # Verify all values are finite (no NaN, +Inf, -Inf)
        if not np.isfinite(embeddings).all():
            raise EncodingError("Model encoding produced non-finite values (NaN or Inf).")

        # L2-normalization verification and fallback
        if self._normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Guard against zero-norm division (extremely rare with real embeddings)
            zero_norm_mask = (norms == 0).flatten()
            if np.any(zero_norm_mask):
                raise EncodingError("One or more embeddings produced zero L2 norm; cannot normalize.")

            # If not perfectly normalized by backend, apply explicit normalization
            if not np.allclose(norms, 1.0, atol=1e-4):
                embeddings = embeddings / norms

        # Convert to list of list of native Python floats
        return embeddings.tolist()


_embedding_service_instance: Optional[EmbeddingService] = None
_embedding_singleton_lock = threading.Lock()


def get_embedding_service() -> EmbeddingService:
    """Returns the shared EmbeddingService singleton instance."""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        with _embedding_singleton_lock:
            if _embedding_service_instance is None:
                from app.core.config import settings

                _embedding_service_instance = EmbeddingService(
                    model_name=settings.EMBEDDING_MODEL_NAME
                )
    return _embedding_service_instance


def reset_embedding_service() -> None:
    """Resets the cached EmbeddingService singleton (useful for test fixtures)."""
    global _embedding_service_instance
    with _embedding_singleton_lock:
        _embedding_service_instance = None

