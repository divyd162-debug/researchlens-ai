"""Optional real-model integration smoke test for EmbeddingService.

This test is kept strictly separate from the default unit test suite.
It is SKIPPED by default to prevent downloading model weights or making network calls
during regular development and CI test runs.

To run this test explicitly:
    RUN_REAL_MODEL_TESTS=1 pytest backend/tests/test_embedding_integration.py
"""
import os
import math
import pytest

from app.services.embedding_service import EmbeddingService


@pytest.mark.skipif(
    os.getenv("RUN_REAL_MODEL_TESTS") != "1",
    reason="Real-model smoke test skipped by default. Run with RUN_REAL_MODEL_TESTS=1 to execute.",
)
def test_real_model_smoke():
    """Optional smoke test verifying real all-MiniLM-L6-v2 model inference and normalization."""
    service = EmbeddingService(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Verify model dimension and max sequence length
    assert service.dimension == 384
    assert service.max_seq_length == 256

    # Test single query embedding
    query_vec = service.embed_text("How do transformer attention mechanisms scale with sequence length?")
    assert len(query_vec) == 384
    assert all(math.isfinite(x) for x in query_vec)

    query_norm = math.sqrt(sum(x * x for x in query_vec))
    assert pytest.approx(query_norm, abs=1e-4) == 1.0

    # Test batch embedding
    passages = [
        "Self-attention calculates alignment scores between queries and keys.",
        "The quadratic complexity in sequence length poses computational challenges.",
    ]
    batch_vecs = service.embed_batch(passages)
    assert len(batch_vecs) == 2
    for vec in batch_vecs:
        assert len(vec) == 384
        vec_norm = math.sqrt(sum(x * x for x in vec))
        assert pytest.approx(vec_norm, abs=1e-4) == 1.0
