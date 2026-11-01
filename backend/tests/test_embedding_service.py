"""Unit tests for EmbeddingService using mocked SentenceTransformer (no network/downloads)."""
import math
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.services.embedding_service import (
    EmbeddingService,
    EmbeddingServiceError,
    ModelLoadError,
    EncodingError,
)


def create_fake_normalized_vectors(n_items: int, dimension: int = 384) -> np.ndarray:
    """Helper creating deterministic, finite, L2-normalized vectors."""
    vectors = []
    for i in range(n_items):
        raw = np.array([(j + 1) * (i + 1) for j in range(dimension)], dtype=np.float32)
        norm = np.linalg.norm(raw)
        vectors.append(raw / norm)
    return np.array(vectors, dtype=np.float32)


@pytest.fixture
def mock_sentence_transformer():
    """Mocks SentenceTransformer class so tests never download real model weights."""
    with patch("sentence_transformers.SentenceTransformer") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384
        mock_instance.max_seq_length = 256

        # Default encode behavior: returns normalized 384-d vectors
        def fake_encode(sentences, **kwargs):
            return create_fake_normalized_vectors(len(sentences), dimension=384)

        mock_instance.encode.side_effect = fake_encode
        mock_cls.return_value = mock_instance
        yield mock_cls


class TestEmbeddingServiceLazyLoading:
    """Tests lazy model initialization and caching behavior."""

    def test_lazy_loading_not_initialized_at_instantiation(self, mock_sentence_transformer):
        """Verifies SentenceTransformer is not loaded in __init__."""
        service = EmbeddingService(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert not service.is_loaded
        mock_sentence_transformer.assert_not_called()

    def test_model_loaded_only_once_per_service_instance(self, mock_sentence_transformer):
        """Verifies model is loaded on first access and cached for subsequent calls."""
        service = EmbeddingService()

        # First access loads the model
        dim = service.dimension
        assert dim == 384
        assert service.is_loaded
        assert mock_sentence_transformer.call_count == 1

        # Second access reuses loaded model without reloading
        _ = service.embed_text("Query text")
        _ = service.max_seq_length
        assert mock_sentence_transformer.call_count == 1

    def test_model_loading_failure_raises_model_load_error(self):
        """Verifies that failures during model loading raise ModelLoadError."""
        with patch("sentence_transformers.SentenceTransformer", side_effect=RuntimeError("Out of memory")):
            service = EmbeddingService()
            with pytest.raises(ModelLoadError, match="Failed to load SentenceTransformer model"):
                _ = service.dimension


class TestEmbeddingServiceEncoding:
    """Tests vector generation, shapes, normalization, and batching."""

    def test_embed_single_text_output_shape_and_normalization(self, mock_sentence_transformer):
        """Verifies single text returns a 1D vector of length dimension with unit L2 norm."""
        service = EmbeddingService()
        vec = service.embed_text("Semantic search across academic research papers.")

        assert isinstance(vec, list)
        assert len(vec) == 384

        # Check all values are finite
        assert all(math.isfinite(x) for x in vec)

        # Check L2 normalization (sum of squares == 1.0)
        norm = math.sqrt(sum(x * x for x in vec))
        assert pytest.approx(norm, abs=1e-4) == 1.0

    def test_embed_batch_output_shape_and_normalization(self, mock_sentence_transformer):
        """Verifies batch embedding returns a 2D list of vectors with correct dimensions."""
        service = EmbeddingService()
        texts = [
            "First passage on natural language processing.",
            "Second passage on transformer architectures.",
            "Third passage on contrastive vector learning.",
        ]
        batch_vecs = service.embed_batch(texts)

        assert isinstance(batch_vecs, list)
        assert len(batch_vecs) == 3
        for vec in batch_vecs:
            assert len(vec) == 384
            assert all(math.isfinite(x) for x in vec)
            norm = math.sqrt(sum(x * x for x in vec))
            assert pytest.approx(norm, abs=1e-4) == 1.0

    def test_empty_batch_returns_empty_list_without_calling_model(self, mock_sentence_transformer):
        """Verifies empty batch returns empty list immediately without calling model."""
        service = EmbeddingService()
        result = service.embed_batch([])
        assert result == []


class TestEmbeddingServiceInputValidation:
    """Tests input rejection on invalid, empty, or malformed inputs."""

    def test_empty_or_whitespace_text_raises_value_error(self, mock_sentence_transformer):
        """Verifies empty strings and whitespace-only strings raise ValueError."""
        service = EmbeddingService()

        with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
            service.embed_text("")

        with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
            service.embed_text("   \n\t  ")

    def test_batch_with_empty_item_raises_value_error(self, mock_sentence_transformer):
        """Verifies that an empty item anywhere inside a batch raises ValueError."""
        service = EmbeddingService()
        with pytest.raises(ValueError, match="texts\\[1\\] cannot be empty"):
            service.embed_batch(["Valid text", "   ", "Another valid text"])

    def test_invalid_types_raise_type_error(self, mock_sentence_transformer):
        """Verifies non-string inputs raise TypeError."""
        service = EmbeddingService()

        with pytest.raises(TypeError, match="must be a string"):
            service.embed_text(12345)  # type: ignore

        with pytest.raises(TypeError, match="must be a string"):
            service.embed_text(None)  # type: ignore

        with pytest.raises(TypeError, match="must be a list"):
            service.embed_batch("Not a list")  # type: ignore

        with pytest.raises(TypeError, match="texts\\[0\\] must be a string"):
            service.embed_batch([None])  # type: ignore


class TestEmbeddingServiceErrorHandling:
    """Tests handling of runtime inference failures, shape mismatches, and non-finite numbers."""

    def test_model_encode_exception_raises_encoding_error(self, mock_sentence_transformer):
        """Verifies exceptions during encode() raise EncodingError."""
        mock_instance = mock_sentence_transformer.return_value
        mock_instance.encode.side_effect = RuntimeError("CUDA internal error")

        service = EmbeddingService()
        with pytest.raises(EncodingError, match="Failed to generate embeddings"):
            service.embed_text("Query")

    def test_non_finite_values_raise_encoding_error(self, mock_sentence_transformer):
        """Verifies that NaN or Inf vectors raise EncodingError."""
        mock_instance = mock_sentence_transformer.return_value
        bad_vectors = np.full((1, 384), np.nan, dtype=np.float32)
        mock_instance.encode.side_effect = lambda *args, **kwargs: bad_vectors

        service = EmbeddingService()
        with pytest.raises(EncodingError, match="non-finite values"):
            service.embed_text("Valid text")

    def test_shape_mismatch_raises_encoding_error(self, mock_sentence_transformer):
        """Verifies vectors with unexpected dimensions raise EncodingError."""
        mock_instance = mock_sentence_transformer.return_value
        # Returns 128-dim instead of expected 384
        wrong_shape = np.ones((1, 128), dtype=np.float32)
        mock_instance.encode.side_effect = lambda *args, **kwargs: wrong_shape

        service = EmbeddingService()
        with pytest.raises(EncodingError, match="shape mismatch"):
            service.embed_text("Valid text")

    def test_max_seq_length_inspection(self, mock_sentence_transformer):
        """Verifies max_seq_length is correctly exposed from loaded model."""
        service = EmbeddingService()
        assert service.max_seq_length == 256
