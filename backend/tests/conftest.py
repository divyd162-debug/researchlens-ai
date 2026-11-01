"""Global pytest configuration and isolation fixtures for ResearchLens AI backend tests."""
import os
from pathlib import Path
import pytest

from app.core.config import settings
from app.core.worker_lock import reset_worker_lock
from app.services.vector_store import reset_vector_store


@pytest.fixture(autouse=True)
def isolate_test_data_dir(tmp_path_factory, monkeypatch):
    """Isolates data and vector store paths for every test run.

    Prevents tests from locking or modifying the live development server's
    data/faiss_worker.lock and data/faiss_index.bin.
    """
    test_dir = tmp_path_factory.mktemp("backend_test_data")
    monkeypatch.setattr(settings, "DATA_DIR", str(test_dir))
    monkeypatch.setattr(settings, "VECTOR_INDEX_DIR", str(test_dir))
    reset_worker_lock()
    reset_vector_store()
    yield
    reset_worker_lock()
    reset_vector_store()
