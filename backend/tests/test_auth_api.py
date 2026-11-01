"""Integration tests for FastAPI authentication routes and current-user dependency."""
from datetime import timedelta
from uuid import uuid4
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.core.security import create_access_token, verify_password
from app.main import app
from app.models.user import User


@pytest.fixture
def db_session():
    """Creates an isolated in-memory SQLite database session."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @sa.event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def client(db_session: Session):
    """Provides a TestClient with get_db overridden to use the isolated test session."""
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestRegisterEndpoint:
    """Tests for POST /api/v1/auth/register."""

    def test_register_success(self, client: TestClient, db_session: Session):
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "  NewScientist@Example.COM  ",
                "password": "StrongPassword123!",
            },
        )
        assert response.status_code == 201
        data = response.json()

        assert "id" in data
        assert data["email"] == "newscientist@example.com"
        assert data["is_active"] is True
        assert data["is_verified"] is False
        assert "created_at" in data
        assert "updated_at" in data

        # Security check: password hashes must NEVER be exposed in the response
        assert "password" not in data
        assert "hashed_password" not in data

        # Verify database state
        db_user = db_session.query(User).filter(User.email == "newscientist@example.com").first()
        assert db_user is not None
        assert db_user.hashed_password != "StrongPassword123!"
        assert verify_password("StrongPassword123!", db_user.hashed_password) is True

    def test_register_duplicate_email_returns_409(self, client: TestClient):
        # First registration
        res1 = client.post(
            "/api/v1/auth/register",
            json={"email": "duplicate@example.com", "password": "Password123!"},
        )
        assert res1.status_code == 201

        # Second registration with case variation
        res2 = client.post(
            "/api/v1/auth/register",
            json={"email": "DUPLICATE@example.com", "password": "DifferentPassword123!"},
        )
        assert res2.status_code == 409
        data = res2.json()
        assert "already exists" in data["detail"].lower()

    def test_register_short_password_returns_422(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "short@example.com", "password": "short"},
        )
        assert response.status_code == 422

    def test_register_multibyte_password_exceeding_72_bytes_returns_422(self, client: TestClient):
        # 20 lock emoji = 80 UTF-8 bytes
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "emoji@example.com", "password": "🔒" * 20},
        )
        assert response.status_code == 422

    def test_register_invalid_email_format_returns_error(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/register",
            json={"email": "not-an-email", "password": "ValidPassword123!"},
        )
        assert response.status_code in (400, 422)


class TestLoginEndpoint:
    """Tests for POST /api/v1/auth/login."""

    @pytest.fixture(autouse=True)
    def create_test_users(self, client: TestClient, db_session: Session):
        # Active user
        client.post(
            "/api/v1/auth/register",
            json={"email": "user@example.com", "password": "UserPassword123!"},
        )
        # Inactive user
        client.post(
            "/api/v1/auth/register",
            json={"email": "inactive@example.com", "password": "InactivePassword123!"},
        )
        inactive = db_session.query(User).filter(User.email == "inactive@example.com").first()
        inactive.is_active = False
        db_session.commit()

    def test_login_success(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "  USER@Example.COM  ", "password": "UserPassword123!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_login_wrong_password_returns_generic_401(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "user@example.com", "password": "WrongPassword123!"},
        )
        assert response.status_code == 401
        assert response.headers.get("www-authenticate") == "Bearer"
        assert response.json()["detail"] == "Invalid email or password."

    def test_login_unknown_email_returns_generic_401(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "unknown@example.com", "password": "AnyPassword123!"},
        )
        assert response.status_code == 401
        assert response.headers.get("www-authenticate") == "Bearer"
        assert response.json()["detail"] == "Invalid email or password."

    def test_login_inactive_user_with_correct_password_returns_generic_401(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "inactive@example.com", "password": "InactivePassword123!"},
        )
        assert response.status_code == 401
        assert response.headers.get("www-authenticate") == "Bearer"
        # Must not disclose that the account exists
        assert response.json()["detail"] == "Invalid email or password."

    def test_login_inactive_user_with_wrong_password_returns_generic_401(self, client: TestClient):
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "inactive@example.com", "password": "WrongPassword123!"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid email or password."


class TestCurrentUserEndpoint:
    """Tests for GET /api/v1/auth/me."""

    @pytest.fixture
    def active_user_and_token(self, client: TestClient, db_session: Session):
        reg_res = client.post(
            "/api/v1/auth/register",
            json={"email": "me_user@example.com", "password": "MySecretPassword123!"},
        )
        user_id = reg_res.json()["id"]
        login_res = client.post(
            "/api/v1/auth/login",
            json={"email": "me_user@example.com", "password": "MySecretPassword123!"},
        )
        token = login_res.json()["access_token"]
        return user_id, token

    def test_get_me_success(self, client: TestClient, active_user_and_token):
        user_id, token = active_user_and_token
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == user_id
        assert data["email"] == "me_user@example.com"
        assert data["is_active"] is True
        assert "hashed_password" not in data

    def test_get_me_missing_auth_header_returns_401(self, client: TestClient):
        response = client.get("/api/v1/auth/me")
        assert response.status_code == 401
        assert response.headers.get("www-authenticate") == "Bearer"
        assert response.json()["detail"] == "Not authenticated"

    def test_get_me_invalid_auth_scheme_returns_401(self, client: TestClient, active_user_and_token):
        _, token = active_user_and_token
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Basic {token}"},
        )
        assert response.status_code == 401
        assert response.headers.get("www-authenticate") == "Bearer"
        assert response.json()["detail"] == "Not authenticated"

    def test_get_me_malformed_token_returns_401(self, client: TestClient):
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer not-a-valid-jwt-token"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Could not validate credentials"

    def test_get_me_expired_token_returns_401(self, client: TestClient, db_session: Session):
        import jwt
        from datetime import datetime, timezone
        from app.core.config import settings

        # Create user
        reg_res = client.post(
            "/api/v1/auth/register",
            json={"email": "expired@example.com", "password": "Password123!"},
        )
        user_id = reg_res.json()["id"]

        # Manually encode an expired token (exp in the past)
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user_id),
            "type": "access",
            "iat": int((now - timedelta(hours=2)).timestamp()),
            "exp": int((now - timedelta(hours=1)).timestamp()),
        }
        expired_token = jwt.encode(payload, settings.get_jwt_secret_key(), algorithm=settings.JWT_ALGORITHM)

        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert response.status_code == 401
        assert response.headers.get("www-authenticate") == "Bearer"
        assert response.json()["detail"] == "Could not validate credentials"

    def test_get_me_nonexistent_user_subject_returns_401(self, client: TestClient):
        # Token with random non-existent UUID subject
        orphan_token = create_access_token(subject=uuid4())
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {orphan_token}"},
        )
        assert response.status_code == 401
        assert response.headers.get("www-authenticate") == "Bearer"
        assert response.json()["detail"] == "Could not validate credentials"

    def test_get_me_inactive_user_returns_401(self, client: TestClient, db_session: Session):
        from uuid import UUID

        reg_res = client.post(
            "/api/v1/auth/register",
            json={"email": "deactivated@example.com", "password": "Password123!"},
        )
        user_id = reg_res.json()["id"]
        user_uuid = UUID(user_id)
        token = create_access_token(subject=user_uuid)

        # Deactivate user
        user = db_session.query(User).filter(User.id == user_uuid).first()
        user.is_active = False
        db_session.commit()

        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 401
        assert response.headers.get("www-authenticate") == "Bearer"
        assert response.json()["detail"] == "Could not validate credentials"




class TestOpenAPISpecification:
    """Verifies that OpenAPI exposes authentication endpoints and the bearer security scheme."""

    def test_openapi_schema_contains_auth_routes_and_security_scheme(self, client: TestClient):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()

        paths = schema.get("paths", {})
        assert "/api/v1/auth/register" in paths
        assert "post" in paths["/api/v1/auth/register"]

        assert "/api/v1/auth/login" in paths
        assert "post" in paths["/api/v1/auth/login"]

        assert "/api/v1/auth/me" in paths
        assert "get" in paths["/api/v1/auth/me"]

        # Verify that /me route requires security
        me_get = paths["/api/v1/auth/me"]["get"]
        assert "security" in me_get
        security_requirements = me_get["security"]
        assert any("HTTPBearer" in req for req in security_requirements)

        # Verify components security schemes
        components = schema.get("components", {})
        security_schemes = components.get("securitySchemes", {})
        assert "HTTPBearer" in security_schemes
        bearer_scheme = security_schemes["HTTPBearer"]
        assert bearer_scheme["type"] == "http"
        assert bearer_scheme["scheme"] == "bearer"
