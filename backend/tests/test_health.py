"""Basic smoke and health check tests for FastAPI backend."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Verify that root endpoint returns HTTP 200 and expected metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "ResearchLens AI"
    assert data["status"] == "online"
    assert "version" in data

def test_health_check_endpoint():
    """Verify that health check endpoint responds with healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "environment" in data
