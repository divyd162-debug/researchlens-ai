"""Unit tests for AuthService: registration, authentication, timing mitigation, and session handling."""
from datetime import datetime
from unittest.mock import patch
from uuid import UUID

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.database import Base
from app.core.security import (
    DUMMY_PASSWORD_HASH,
    decode_access_token,
    verify_password,
)
from app.models.user import User
from app.schemas.auth import (
    AuthResult,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)
from app.services.auth_service import (
    AuthService,
    DuplicateEmailError,
    InactiveUserError,
    InvalidCredentialsError,
    InvalidEmailError,
    WeakPasswordError,
    auth_service,
    normalize_email,
    validate_password_complexity,
)


@pytest.fixture
def db_session():
    """Provides an isolated in-memory SQLite database session."""
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


class TestEmailNormalization:
    """Tests for email normalization and structure validation."""

    def test_email_lowercased_and_trimmed(self):
        result = normalize_email("  Researcher@Example.COM  ")
        assert result == "researcher@example.com"

    def test_email_with_subdomain_and_plus(self):
        result = normalize_email("user+tag@department.university.edu")
        assert result == "user+tag@department.university.edu"

    @pytest.mark.parametrize(
        "invalid_email",
        [
            "",
            "   ",
            "not-an-email",
            "@example.com",
            "user@",
            "user@domain",
            "user @example.com",
            "user@ example.com",
            "user@@example.com",
            "user@.com",
            "user@domain.",
            "a" * 315 + "@example.com",  # Exceeds 320 chars
        ],
    )
    def test_invalid_email_formats_rejected(self, invalid_email):
        with pytest.raises(InvalidEmailError):
            normalize_email(invalid_email)


class TestPasswordComplexity:
    """Tests for password complexity and bcrypt UTF-8 byte boundary enforcement."""

    def test_valid_passwords(self):
        validate_password_complexity("correct-horse-battery-staple")
        validate_password_complexity("12345678")

    def test_password_shorter_than_eight_chars_rejected(self):
        with pytest.raises(WeakPasswordError, match="at least 8 characters"):
            validate_password_complexity("short")

    def test_empty_password_rejected(self):
        with pytest.raises(WeakPasswordError):
            validate_password_complexity("")

    def test_multibyte_unicode_password_exceeding_72_bytes_rejected(self):
        # 20 lock emoji = 20 * 4 = 80 UTF-8 bytes (len is 20 chars, but 80 bytes)
        emoji_pw = "🔒" * 20
        assert len(emoji_pw) == 20
        assert len(emoji_pw.encode("utf-8")) == 80
        with pytest.raises(WeakPasswordError, match="cannot exceed 72 bytes"):
            validate_password_complexity(emoji_pw)

    def test_multibyte_unicode_password_within_72_bytes_accepted(self):
        # 10 lock emoji = 40 UTF-8 bytes, 10 chars >= 8
        emoji_pw = "🔒" * 10
        assert len(emoji_pw) == 10
        assert len(emoji_pw.encode("utf-8")) == 40
        validate_password_complexity(emoji_pw)


class TestUserRegistration:
    """Tests for user account registration via AuthService."""

    def test_register_user_success(self, db_session: Session):
        request = UserRegisterRequest(
            email="  NewUser@Example.Com  ",
            password="SecurePassword123!",
        )
        response = auth_service.register_user(db_session, request)

        assert isinstance(response, UserResponse)
        assert isinstance(response.id, UUID)
        assert response.email == "newuser@example.com"
        assert response.is_active is True
        assert response.is_verified is False
        assert isinstance(response.created_at, datetime)
        assert isinstance(response.updated_at, datetime)

        # Verify database record
        db_user = db_session.query(User).filter(User.id == response.id).first()
        assert db_user is not None
        assert db_user.email == "newuser@example.com"
        assert db_user.hashed_password != "SecurePassword123!"
        assert verify_password("SecurePassword123!", db_user.hashed_password) is True

        # Verify response DTO never contains hashed_password
        assert not hasattr(response, "hashed_password")

    def test_register_user_from_dict(self, db_session: Session):
        response = auth_service.register_user(
            db_session,
            {"email": "dict_user@example.com", "password": "Password123!"},
        )
        assert response.email == "dict_user@example.com"

    def test_register_duplicate_email_raises_duplicate_error(self, db_session: Session):
        auth_service.register_user(
            db_session,
            {"email": "duplicate@example.com", "password": "Password123!"},
        )
        # Attempt registering same email with different case
        with pytest.raises(DuplicateEmailError, match="already exists"):
            auth_service.register_user(
                db_session,
                {"email": "DUPLICATE@EXAMPLE.COM", "password": "DifferentPassword123!"},
            )

    def test_register_handles_uniqueness_race_and_rolls_back(self, db_session: Session):
        """Simulates concurrent insert race where db.commit() raises IntegrityError."""
        service = AuthService()

        # Mock db.commit to raise IntegrityError simulating a concurrent transaction commit
        with patch.object(db_session, "commit", side_effect=IntegrityError("duplicate", {}, None)):
            with pytest.raises(DuplicateEmailError, match="already exists"):
                service.register_user(
                    db_session,
                    {"email": "racer@example.com", "password": "Password123!"},
                )

        # Verify session is clean and operable after rollback
        users_count = db_session.query(User).count()
        assert users_count == 0

    def test_register_weak_password_rejected(self, db_session: Session):
        with pytest.raises(WeakPasswordError):
            auth_service.register_user(
                db_session,
                {"email": "user@example.com", "password": "short"},
            )

    def test_register_invalid_email_rejected(self, db_session: Session):
        with pytest.raises(InvalidEmailError):
            auth_service.register_user(
                db_session,
                {"email": "not-an-email", "password": "ValidPassword123!"},
            )


class TestUserAuthentication:
    """Tests for user login, timing side-channel mitigation, and token issuance."""

    @pytest.fixture(autouse=True)
    def setup_users(self, db_session: Session):
        # Register an active user
        auth_service.register_user(
            db_session,
            {"email": "active@example.com", "password": "ActiveSecret123!"},
        )
        # Register and then deactivate an inactive user
        inactive = auth_service.register_user(
            db_session,
            {"email": "inactive@example.com", "password": "InactiveSecret123!"},
        )
        db_user = db_session.query(User).filter(User.id == inactive.id).first()
        db_user.is_active = False
        db_session.commit()

    def test_authenticate_success(self, db_session: Session):
        request = UserLoginRequest(
            email="ACTIVE@EXAMPLE.COM",  # Case normalization
            password="ActiveSecret123!",
        )
        result = auth_service.authenticate_user(db_session, request)

        assert isinstance(result, AuthResult)
        assert isinstance(result.user, UserResponse)
        assert result.user.email == "active@example.com"
        assert isinstance(result.token, TokenResponse)
        assert result.token.token_type == "bearer"
        assert result.token.expires_in == settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60

        # Decode and verify the issued JWT token
        payload = decode_access_token(result.token.access_token)
        assert payload.sub == result.user.id
        assert payload.type == "access"

    def test_authenticate_from_dict(self, db_session: Session):
        result = auth_service.authenticate_user(
            db_session,
            {"email": "active@example.com", "password": "ActiveSecret123!"},
        )
        assert result.user.email == "active@example.com"

    def test_authenticate_wrong_password_raises_generic_error(self, db_session: Session):
        with pytest.raises(InvalidCredentialsError) as exc_info:
            auth_service.authenticate_user(
                db_session,
                {"email": "active@example.com", "password": "WrongPassword123!"},
            )
        assert str(exc_info.value) == "Invalid email or password."

    def test_authenticate_unknown_email_calls_dummy_hash_and_raises_generic_error(
        self, db_session: Session
    ):
        """Verifies that unknown email triggers verify_password against DUMMY_PASSWORD_HASH for constant timing."""
        with patch("app.services.auth_service.verify_password", wraps=verify_password) as mock_verify:
            with pytest.raises(InvalidCredentialsError) as exc_info:
                auth_service.authenticate_user(
                    db_session,
                    {"email": "nonexistent@example.com", "password": "AnyPassword123!"},
                )
            assert str(exc_info.value) == "Invalid email or password."
            # Verify dummy verification was executed with DUMMY_PASSWORD_HASH
            mock_verify.assert_called_once_with("AnyPassword123!", DUMMY_PASSWORD_HASH)

    def test_authenticate_malformed_email_calls_dummy_hash_and_raises_generic_error(
        self, db_session: Session
    ):
        """Verifies malformed email triggers dummy hash verification to preserve constant timing."""
        with patch("app.services.auth_service.verify_password", wraps=verify_password) as mock_verify:
            with pytest.raises(InvalidCredentialsError) as exc_info:
                auth_service.authenticate_user(
                    db_session,
                    {"email": "not-an-email", "password": "AnyPassword123!"},
                )
            assert str(exc_info.value) == "Invalid email or password."
            mock_verify.assert_called_once_with("AnyPassword123!", DUMMY_PASSWORD_HASH)

    def test_authenticate_inactive_user_with_correct_password_raises_generic_error(
        self, db_session: Session
    ):
        """Verifies inactive user is rejected with identical generic error message."""
        with pytest.raises(InactiveUserError) as exc_info:
            auth_service.authenticate_user(
                db_session,
                {"email": "inactive@example.com", "password": "InactiveSecret123!"},
            )
        # Message must not disclose account existence
        assert str(exc_info.value) == "Invalid email or password."
        # Must be an instance of InvalidCredentialsError
        assert isinstance(exc_info.value, InvalidCredentialsError)

    def test_authenticate_inactive_user_with_wrong_password_raises_generic_error(
        self, db_session: Session
    ):
        with pytest.raises(InvalidCredentialsError) as exc_info:
            auth_service.authenticate_user(
                db_session,
                {"email": "inactive@example.com", "password": "WrongPassword123!"},
            )
        assert str(exc_info.value) == "Invalid email or password."

    def test_dummy_password_hash_safely_returns_false(self):
        """Explicitly proves DUMMY_PASSWORD_HASH returns False for arbitrary passwords."""
        assert verify_password("Password123!", DUMMY_PASSWORD_HASH) is False
        assert verify_password("", DUMMY_PASSWORD_HASH) is False
        assert verify_password("wrong", DUMMY_PASSWORD_HASH) is False


class TestUserLookup:
    """Tests for user lookup helpers by ID and normalized email."""

    def test_get_user_by_id_and_email(self, db_session: Session):
        created = auth_service.register_user(
            db_session,
            {"email": "lookup@example.com", "password": "LookupSecret123!"},
        )

        found_by_id = auth_service.get_user_by_id(db_session, created.id)
        assert found_by_id is not None
        assert found_by_id.email == "lookup@example.com"

        found_by_email = auth_service.get_user_by_email(db_session, "  LOOKUP@example.com  ")
        assert found_by_email is not None
        assert found_by_email.id == created.id

    def test_get_user_not_found_returns_none(self, db_session: Session):
        import uuid
        assert auth_service.get_user_by_id(db_session, uuid.uuid4()) is None
        assert auth_service.get_user_by_email(db_session, "unknown@example.com") is None
        assert auth_service.get_user_by_email(db_session, "invalid-email") is None
