"""Smoke, liveness, and readiness check tests for FastAPI backend."""
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.deps import get_db
from app.core.database import Base
from app.main import app
from app.services.vector_store import get_vector_store

client = TestClient(app)


@pytest.fixture
def test_db_session():
    """Provides an isolated SQLite in-memory database session."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_root_endpoint():
    """Verify that root endpoint returns HTTP 200 and expected metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "ResearchLens AI"
    assert data["status"] == "online"
    assert "version" in data


def test_liveness_endpoint():
    """Verify that liveness check endpoint responds with healthy status without checking dependencies."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "environment" in data


def test_readiness_all_dependencies_healthy(test_db_session: Session):
    """Verify that readiness check returns HTTP 200 ready when DB, vector store, and upload dir are healthy."""
    def _override_get_db():
        yield test_db_session

    app.dependency_overrides[get_db] = _override_get_db
    try:
        vs = get_vector_store()
        vs.set_ready(True)

        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}
    finally:
        app.dependency_overrides.clear()


def test_readiness_fails_when_database_is_unreachable():
    """Verify that readiness returns HTTP 503 and reports database failure without leaking credentials."""
    def _failing_get_db():
        mock_session = MagicMock()
        mock_session.execute.side_effect = sa.exc.OperationalError(
            "FATAL: password authentication failed for user 'secret_user'",
            params=None,
            orig=Exception("Connection refused"),
        )
        yield mock_session

    app.dependency_overrides[get_db] = _failing_get_db
    try:
        response = client.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert "database" in data["failures"]
        # Ensure credentials/sensitive info are NOT exposed in response body
        assert "secret_user" not in str(data)
        assert "password" not in str(data)
    finally:
        app.dependency_overrides.clear()


def test_readiness_fails_when_vector_store_is_not_ready(test_db_session: Session):
    """Verify that readiness returns HTTP 503 when vector store is in reloading/recovery mode."""
    def _override_get_db():
        yield test_db_session

    app.dependency_overrides[get_db] = _override_get_db
    vs = get_vector_store()
    original_state = vs.is_ready
    try:
        vs.set_ready(False)
        response = client.get("/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "not_ready"
        assert "vector_store" in data["failures"]
    finally:
        vs.set_ready(original_state)
        app.dependency_overrides.clear()


def test_readiness_fails_when_upload_directory_is_inaccessible(test_db_session: Session):
    """Verify that readiness returns HTTP 503 when upload directory does not exist or lacks permissions."""
    def _override_get_db():
        yield test_db_session

    app.dependency_overrides[get_db] = _override_get_db
    try:
        with patch("pathlib.Path.exists", return_value=False):
            response = client.get("/health/ready")
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "not_ready"
            assert "upload_directory" in data["failures"]
    finally:
        app.dependency_overrides.clear()


def test_liveness_survives_database_outage():
    """Verify that /health liveness probe remains 200 OK even when database is completely down."""
    def _failing_get_db():
        raise RuntimeError("Database completely crashed!")

    app.dependency_overrides[get_db] = _failing_get_db
    try:
        # Liveness check must continue returning 200
        liveness_response = client.get("/health")
        assert liveness_response.status_code == 200
        assert liveness_response.json()["status"] == "healthy"

        # Readiness check fails with 503
        readiness_response = client.get("/health/ready")
        assert readiness_response.status_code == 503
        assert "database" in readiness_response.json()["failures"]
    finally:
        app.dependency_overrides.clear()
