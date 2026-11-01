"""Unit tests for VectorStore verifying FAISS indexing, similarity ranking, persistence, and concurrency."""
import concurrent.futures
import json
from pathlib import Path
import numpy as np
import pytest

from app.services.vector_store import (
    VectorStore,
    SearchResult,
    DuplicateIDError,
    DimensionMismatchError,
    InvalidVectorError,
    InvalidIDError,
    CorruptIndexError,
    ModelMismatchError,
    IndexNotFoundError,
)


def make_unit_vectors(num_vectors: int, dimension: int = 4) -> np.ndarray:
    """Helper creating reproducible unit-normalized vectors."""
    rng = np.random.default_rng(seed=42)
    raw = rng.standard_normal((num_vectors, dimension), dtype=np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


@pytest.fixture
def small_store(tmp_path: Path) -> VectorStore:
    """Fixture providing a 4-dimensional VectorStore for exact geometric assertions."""
    return VectorStore(dimension=4, model_name="test-model", index_dir=tmp_path)


class TestVectorStoreBasicOps:
    """Tests for vector insertion, counting, and presence checking."""

    def test_add_and_count_vectors(self, small_store: VectorStore):
        """Verifies adding vectors updates total count and ID tracking."""
        vecs = make_unit_vectors(3, dimension=4)
        ids = [101, 102, 103]

        added = small_store.add_vectors(ids, vecs)
        assert added == 3
        assert small_store.total_vectors == 3
        assert small_store.contains_id(101)
        assert small_store.contains_id(102)
        assert small_store.contains_id(103)
        assert not small_store.contains_id(999)

    def test_empty_batch_add(self, small_store: VectorStore):
        """Verifies passing empty lists returns 0 without modifying store."""
        added = small_store.add_vectors([], np.empty((0, 4), dtype=np.float32))
        assert added == 0
        assert small_store.total_vectors == 0

    def test_clear_resets_store(self, small_store: VectorStore):
        """Verifies clear() purges all vectors and resets ID set."""
        vecs = make_unit_vectors(2, dimension=4)
        small_store.add_vectors([1, 2], vecs)
        assert small_store.total_vectors == 2

        small_store.clear()
        assert small_store.total_vectors == 0
        assert not small_store.contains_id(1)


class TestSimilarityRanking:
    """Tests verifying cosine similarity ranking and score correctness."""

    def test_cosine_similarity_ranking_order(self, small_store: VectorStore):
        """Verifies exact inner product ranking with known orthogonal and parallel vectors."""
        # Define 3 orthogonal vectors in 4D space
        # v1 is parallel to query [1, 0, 0, 0] (score ~ 1.0)
        # v2 is orthogonal [0, 1, 0, 0] (score ~ 0.0)
        # v3 is diagonal [0.7071, 0.7071, 0, 0] (score ~ 0.7071)
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float32)

        vectors = np.vstack([v1, v2, v3])
        chunk_ids = [1, 2, 3]

        small_store.add_vectors(chunk_ids, vectors)

        # Query matching v1 exactly
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = small_store.search(query, top_k=3)

        assert len(results) == 3
        # Most similar must be chunk 1 (score 1.0)
        assert results[0].chunk_id == 1
        assert pytest.approx(results[0].score, abs=1e-4) == 1.0

        # Second must be chunk 3 (score ~0.7071)
        assert results[1].chunk_id == 3
        assert pytest.approx(results[1].score, abs=1e-4) == 0.7071

        # Third must be chunk 2 (score ~0.0)
        assert results[2].chunk_id == 2
        assert pytest.approx(results[2].score, abs=1e-4) == 0.0

    def test_search_empty_store_returns_empty_list(self, small_store: VectorStore):
        """Verifies search on an empty store returns [] without errors."""
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = small_store.search(query, top_k=5)
        assert results == []

    def test_top_k_bounds(self, small_store: VectorStore):
        """Verifies search respects top_k limit."""
        vecs = make_unit_vectors(5, dimension=4)
        small_store.add_vectors([1, 2, 3, 4, 5], vecs)

        results = small_store.search(vecs[0], top_k=2)
        assert len(results) == 2

    def test_unnormalized_vectors_are_normalized_by_service(self, small_store: VectorStore):
        """Verifies vectors with arbitrary norms are normalized so similarity is in [-1.0, 1.0]."""
        unnorm_v1 = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        unnorm_v2 = np.array([0.0, 50.0, 0.0, 0.0], dtype=np.float32)
        small_store.add_vectors([1, 2], np.vstack([unnorm_v1, unnorm_v2]))

        # Search with an unnormalized query in the same direction as v1
        unnorm_query = np.array([7.0, 0.0, 0.0, 0.0], dtype=np.float32)
        results = small_store.search(unnorm_query, top_k=2)

        assert len(results) == 2
        assert results[0].chunk_id == 1
        assert pytest.approx(results[0].score, abs=1e-4) == 1.0
        assert results[1].chunk_id == 2
        assert pytest.approx(results[1].score, abs=1e-4) == 0.0

    def test_search_with_allowed_ids_filters_candidates(self, small_store: VectorStore):
        """Verifies allowed_ids restricts search candidate pool, resolving dilution."""
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32)
        v3 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        small_store.add_vectors([10, 20, 30], np.vstack([v1, v2, v3]))

        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # Restrict to only 20 and 30 (exclude 10 despite having highest similarity)
        res = small_store.search(query, top_k=2, allowed_ids=[20, 30])
        assert len(res) == 2
        assert res[0].chunk_id == 20
        assert res[1].chunk_id == 30
        assert all(r.chunk_id != 10 for r in res)

    def test_search_with_empty_or_unindexed_allowed_ids(self, small_store: VectorStore):
        """Verifies empty or unindexed allowed_ids returns empty list safely."""
        v = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        small_store.add_vectors([1], v)

        assert small_store.search(v[0], top_k=5, allowed_ids=[]) == []
        assert small_store.search(v[0], top_k=5, allowed_ids=[999, 888]) == []



class TestValidationAndRejection:
    """Tests ensuring duplicate IDs, dimension mismatches, and corrupt inputs fail clearly."""

    def test_duplicate_id_in_same_batch_raises_duplicate_error(self, small_store: VectorStore):
        """Verifies duplicate ID within the same input batch is rejected."""
        vecs = make_unit_vectors(2, dimension=4)
        with pytest.raises(DuplicateIDError, match="Duplicate ID 5 detected within the provided batch"):
            small_store.add_vectors([5, 5], vecs)

        # Verify nothing was added
        assert small_store.total_vectors == 0

    def test_duplicate_id_across_batches_raises_duplicate_error(self, small_store: VectorStore):
        """Verifies inserting an ID that already exists in the store is rejected."""
        vecs = make_unit_vectors(2, dimension=4)
        small_store.add_vectors([10], vecs[0:1])

        with pytest.raises(DuplicateIDError, match="Chunk ID 10 already exists in the vector store"):
            small_store.add_vectors([10], vecs[1:2])

        assert small_store.total_vectors == 1

    def test_invalid_id_values_rejected(self, small_store: VectorStore):
        """Verifies non-positive or non-integer IDs raise InvalidIDError."""
        vecs = make_unit_vectors(1, dimension=4)

        with pytest.raises(InvalidIDError, match="must be a positive integer"):
            small_store.add_vectors([0], vecs)

        with pytest.raises(InvalidIDError, match="must be a positive integer"):
            small_store.add_vectors([-5], vecs)

        with pytest.raises(InvalidIDError, match="must be an integer"):
            small_store.add_vectors(["string_id"], vecs)  # type: ignore

    def test_dimension_mismatch_rejected(self, small_store: VectorStore):
        """Verifies passing vectors with incorrect dimension raises DimensionMismatchError."""
        wrong_vecs = make_unit_vectors(2, dimension=8)  # 8D instead of 4D
        with pytest.raises(DimensionMismatchError, match="expected 4, got 8"):
            small_store.add_vectors([1, 2], wrong_vecs)

        with pytest.raises(DimensionMismatchError, match="expected \\(1, 4\\)"):
            small_store.search([1.0, 2.0])  # 2D query instead of 4D

    def test_non_finite_vectors_rejected(self, small_store: VectorStore):
        """Verifies vectors containing NaN or Inf raise InvalidVectorError."""
        nan_vec = np.array([[np.nan, 0.0, 0.0, 0.0]], dtype=np.float32)
        with pytest.raises(InvalidVectorError, match="non-finite"):
            small_store.add_vectors([1], nan_vec)

        inf_query = [np.inf, 0.0, 0.0, 0.0]
        with pytest.raises(InvalidVectorError, match="non-finite"):
            small_store.search(inf_query)

    def test_zero_magnitude_vector_rejected(self, small_store: VectorStore):
        """Verifies vector of all zeros cannot be normalized."""
        zero_vec = np.zeros((1, 4), dtype=np.float32)
        with pytest.raises(InvalidVectorError, match="zero magnitude"):
            small_store.add_vectors([1], zero_vec)


class TestRemovalAndDeletion:
    """Tests removing IDs and deleting vectors from the index."""

    def test_remove_existing_ids(self, small_store: VectorStore):
        """Verifies removing existing IDs decreases total count and updates search results."""
        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        small_store.add_vectors([10, 20], np.vstack([v1, v2]))

        removed = small_store.remove_ids([10])
        assert removed == 1
        assert small_store.total_vectors == 1
        assert not small_store.contains_id(10)
        assert small_store.contains_id(20)

        # Search should no longer return chunk 10
        results = small_store.search(v1, top_k=2)
        assert len(results) == 1
        assert results[0].chunk_id == 20

    def test_remove_non_existent_ids_safe_noop(self, small_store: VectorStore):
        """Verifies removing non-existent IDs returns 0 and does not error."""
        v = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        small_store.add_vectors([1], v)

        removed = small_store.remove_ids([999, 888])
        assert removed == 0
        assert small_store.total_vectors == 1


class TestPersistenceAndReloading:
    """Tests saving to disk, atomic replacement, metadata checks, and corruption detection."""

    def test_save_and_reload_persisted_index(self, tmp_path: Path):
        """Verifies index serializes to disk atomically and reloads faithfully."""
        store1 = VectorStore(dimension=4, model_name="all-MiniLM-L6-v2", index_dir=tmp_path)
        vecs = make_unit_vectors(3, dimension=4)
        store1.add_vectors([101, 102, 103], vecs)
        store1.save_to_disk()

        assert (tmp_path / VectorStore.DEFAULT_INDEX_FILENAME).exists()
        assert (tmp_path / VectorStore.DEFAULT_META_FILENAME).exists()

        # Create a fresh store instance and load from the same directory
        store2 = VectorStore(dimension=4, model_name="all-MiniLM-L6-v2", index_dir=tmp_path)
        store2.load_from_disk()

        assert store2.total_vectors == 3
        assert store2.contains_id(101)
        assert store2.contains_id(102)
        assert store2.contains_id(103)

        # Search in loaded store should produce identical results
        res1 = store1.search(vecs[0], top_k=3)
        res2 = store2.search(vecs[0], top_k=3)
        assert [r.chunk_id for r in res1] == [r.chunk_id for r in res2]
        assert [r.score for r in res1] == pytest.approx([r.score for r in res2], abs=1e-5)

    def test_missing_files_raises_index_not_found(self, tmp_path: Path):
        """Verifies load_from_disk on empty directory raises IndexNotFoundError."""
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        store = VectorStore(dimension=4, model_name="model-x", index_dir=empty_dir)

        with pytest.raises(IndexNotFoundError, match="not found in directory"):
            store.load_from_disk()

    def test_corrupt_index_file_raises_corrupt_error(self, tmp_path: Path):
        """Verifies corrupt binary index raises CorruptIndexError."""
        store = VectorStore(dimension=4, model_name="model-x", index_dir=tmp_path)
        store.add_vectors([1], make_unit_vectors(1, 4))
        store.save_to_disk()

        # Corrupt the binary index file
        (tmp_path / VectorStore.DEFAULT_INDEX_FILENAME).write_bytes(b"CORRUPTED_BINARY_DATA")

        with pytest.raises(CorruptIndexError, match="Failed to deserialize FAISS index"):
            store.load_from_disk()

    def test_corrupt_metadata_file_raises_corrupt_error(self, tmp_path: Path):
        """Verifies unparseable metadata JSON raises CorruptIndexError."""
        store = VectorStore(dimension=4, model_name="model-x", index_dir=tmp_path)
        store.add_vectors([1], make_unit_vectors(1, 4))
        store.save_to_disk()

        # Corrupt metadata JSON
        (tmp_path / VectorStore.DEFAULT_META_FILENAME).write_text("INVALID_JSON{")

        with pytest.raises(CorruptIndexError, match="Failed to parse metadata JSON"):
            store.load_from_disk()

    def test_model_mismatch_raises_model_mismatch_error(self, tmp_path: Path):
        """Verifies attempting to load index from a different model raises ModelMismatchError."""
        store1 = VectorStore(dimension=4, model_name="model-alpha", index_dir=tmp_path)
        store1.add_vectors([1], make_unit_vectors(1, 4))
        store1.save_to_disk()

        store2 = VectorStore(dimension=4, model_name="model-beta", index_dir=tmp_path)
        with pytest.raises(ModelMismatchError, match="does not match configured model"):
            store2.load_from_disk()

    def test_save_and_load_without_index_dir_raises_value_error(self):
        """Verifies saving or loading without configured index_dir raises ValueError."""
        store = VectorStore(dimension=4, model_name="test-model", index_dir=None)
        with pytest.raises(ValueError, match="no index_dir configured"):
            store.save_to_disk()

        with pytest.raises(ValueError, match="no index_dir configured"):
            store.load_from_disk()

    def test_persistence_count_mismatch_raises_corrupt_error(self, tmp_path: Path):
        """Verifies tampering with total_vectors in metadata triggers CorruptIndexError."""
        store = VectorStore(dimension=4, model_name="model-x", index_dir=tmp_path)
        store.add_vectors([1, 2], make_unit_vectors(2, 4))
        store.save_to_disk()

        meta_path = tmp_path / VectorStore.DEFAULT_META_FILENAME
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["total_vectors"] = 99  # Mismatch with actual 2 vectors
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        fresh_store = VectorStore(dimension=4, model_name="model-x", index_dir=tmp_path)
        with pytest.raises(CorruptIndexError, match="Index vector count mismatch"):
            fresh_store.load_from_disk()

    def test_persistence_id_mismatch_raises_corrupt_error(self, tmp_path: Path):
        """Verifies tampering with indexed_ids in metadata triggers CorruptIndexError."""
        store = VectorStore(dimension=4, model_name="model-x", index_dir=tmp_path)
        store.add_vectors([1, 2], make_unit_vectors(2, 4))
        store.save_to_disk()

        meta_path = tmp_path / VectorStore.DEFAULT_META_FILENAME
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["indexed_ids"] = [1, 999]  # Mismatch with FAISS embedded IDs [1, 2]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        fresh_store = VectorStore(dimension=4, model_name="model-x", index_dir=tmp_path)
        with pytest.raises(CorruptIndexError, match="Index ID set mismatch"):
            fresh_store.load_from_disk()

    def test_unsupported_format_version_raises_corrupt_error(self, tmp_path: Path):
        """Verifies unsupported format_version raises CorruptIndexError."""
        store = VectorStore(dimension=4, model_name="model-x", index_dir=tmp_path)
        store.add_vectors([1], make_unit_vectors(1, 4))
        store.save_to_disk()

        meta_path = tmp_path / VectorStore.DEFAULT_META_FILENAME
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["format_version"] = 999
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        fresh_store = VectorStore(dimension=4, model_name="model-x", index_dir=tmp_path)
        with pytest.raises(CorruptIndexError, match="Unsupported index format version"):
            fresh_store.load_from_disk()

    def test_missing_required_metadata_key_raises_corrupt_error(self, tmp_path: Path):
        """Verifies missing required key in metadata raises CorruptIndexError."""
        store = VectorStore(dimension=4, model_name="model-x", index_dir=tmp_path)
        store.add_vectors([1], make_unit_vectors(1, 4))
        store.save_to_disk()

        meta_path = tmp_path / VectorStore.DEFAULT_META_FILENAME
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        del meta["metric"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        fresh_store = VectorStore(dimension=4, model_name="model-x", index_dir=tmp_path)
        with pytest.raises(CorruptIndexError, match="missing required key"):
            fresh_store.load_from_disk()

    def test_unsupported_index_type_or_metric_raises_corrupt_error(self, tmp_path: Path):
        """Verifies incompatible index_type or unnormalized flag raises CorruptIndexError."""
        store = VectorStore(dimension=4, model_name="model-x", index_dir=tmp_path)
        store.add_vectors([1], make_unit_vectors(1, 4))
        store.save_to_disk()

        meta_path = tmp_path / VectorStore.DEFAULT_META_FILENAME
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["normalized"] = False
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

        fresh_store = VectorStore(dimension=4, model_name="model-x", index_dir=tmp_path)
        with pytest.raises(CorruptIndexError, match="indicates vectors are not normalized"):
            fresh_store.load_from_disk()



class TestConcurrency:
    """Tests verifying thread safety under concurrent vector additions."""

    def test_concurrent_vector_additions(self, small_store: VectorStore):
        """Verifies multiple threads adding disjoint vectors concurrently without corruption."""
        num_threads = 6
        items_per_thread = 10
        total_items = num_threads * items_per_thread

        def worker(thread_idx: int):
            start_id = thread_idx * items_per_thread + 1
            chunk_ids = list(range(start_id, start_id + items_per_thread))
            vecs = make_unit_vectors(items_per_thread, dimension=4)
            return small_store.add_vectors(chunk_ids, vecs)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            results = [f.result() for f in futures]

        assert sum(results) == total_items
        assert small_store.total_vectors == total_items
        for cid in range(1, total_items + 1):
            assert small_store.contains_id(cid)
