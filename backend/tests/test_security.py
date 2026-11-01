"""Unit tests for password hashing, JWT lifecycle, claims validation, and configuration security."""
from datetime import datetime, timedelta, timezone
import uuid
import jwt
import pytest

from app.core.config import DEV_DEFAULT_JWT_SECRET, settings
from app.core.security import (
    DUMMY_PASSWORD_HASH,
    MAX_ACCESS_TOKEN_LIFETIME,
    InvalidTokenError,
    SecurityConfigurationError,
    TokenExpiredError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.main import create_application


class TestPasswordHashing:
    """Tests for secure password hashing and verification using pwdlib[bcrypt]."""

    def test_hash_password_produces_bcrypt_hash(self):
        """Verifies password hashing returns a well-formed bcrypt hash."""
        password = "CorrectHorseBatteryStaple!"
        hashed = hash_password(password)

        assert isinstance(hashed, str)
        assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
        assert hashed != password

    def test_hash_password_salts_uniquely(self):
        """Verifies successive hashes of the same password produce distinct salts and hashes."""
        password = "SuperSecretPassword123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        assert hash1 != hash2
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True

    def test_verify_password_correct(self):
        """Verifies correct plain password matches its hash."""
        password = "ValidPassword_2026!"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True

    def test_verify_password_wrong(self):
        """Verifies wrong plain password fails verification."""
        password = "ValidPassword_2026!"
        hashed = hash_password(password)
        assert verify_password("WrongPassword_2026!", hashed) is False

    def test_verify_password_case_sensitive(self):
        """Verifies password verification is strictly case-sensitive."""
        password = "CaseSensitivePassword"
        hashed = hash_password(password)
        assert verify_password("casesensitivepassword", hashed) is False

    def test_hash_password_empty_or_invalid_raises(self):
        """Verifies empty or non-string passwords raise ValueError."""
        with pytest.raises(ValueError, match="Password must be a non-empty string"):
            hash_password("")

        with pytest.raises(ValueError, match="Password must be a non-empty string"):
            hash_password(None)  # type: ignore

    def test_hash_password_exceeds_72_bytes_raises(self):
        """Verifies ASCII passwords exceeding bcrypt's 72-byte limit fail fast with a clear ValueError."""
        long_password = "a" * 73
        with pytest.raises(ValueError, match="cannot exceed 72 bytes"):
            hash_password(long_password)

    def test_hash_password_multibyte_unicode_exceeds_72_bytes_raises(self):
        """Verifies multibyte Unicode passwords exceeding 72 UTF-8 bytes (even if under 72 chars) are rejected."""
        # 20 emojis of 4 bytes each = 80 UTF-8 bytes (only 20 characters in length)
        multibyte_password = "🔒" * 20
        assert len(multibyte_password) == 20  # Under 72 characters
        assert len(multibyte_password.encode("utf-8")) == 80  # Exceeds 72 bytes

        with pytest.raises(ValueError, match="cannot exceed 72 bytes"):
            hash_password(multibyte_password)

    def test_hash_password_multibyte_unicode_within_72_bytes_succeeds(self):
        """Verifies multibyte Unicode passwords within 72 UTF-8 bytes are accepted."""
        # 18 emojis of 4 bytes each = exactly 72 UTF-8 bytes
        multibyte_password = "🔒" * 18
        assert len(multibyte_password.encode("utf-8")) == 72
        hashed = hash_password(multibyte_password)
        assert verify_password(multibyte_password, hashed) is True

    def test_verify_password_safe_on_invalid_inputs(self):
        """Verifies verify_password fails closed without raising exceptions on malformed inputs."""
        assert verify_password("", "some_hash") is False
        assert verify_password(None, "some_hash") is False  # type: ignore
        assert verify_password("password", "") is False
        assert verify_password("password", None) is False  # type: ignore
        assert verify_password("password", "invalid_not_a_bcrypt_hash") is False

    def test_dummy_password_hash_verifies_safely_to_false(self):
        """Verifies DUMMY_PASSWORD_HASH verifies safely to False to support Milestone 1B timing mitigation."""
        assert verify_password("any_submitted_password", DUMMY_PASSWORD_HASH) is False


class TestJWTCreation:
    """Tests for JWT access token generation and claims creation."""

    def test_create_access_token_with_uuid(self):
        """Verifies access token creation from a UUID object subject."""
        user_id = uuid.uuid4()
        token = create_access_token(subject=user_id)
        assert isinstance(token, str)
        assert len(token.split(".")) == 3  # Header.Payload.Signature

    def test_create_access_token_with_uuid_string(self):
        """Verifies access token creation from a valid UUID string subject."""
        user_id_str = str(uuid.uuid4())
        token = create_access_token(subject=user_id_str)
        payload = decode_access_token(token)
        assert str(payload.sub) == user_id_str

    def test_create_access_token_invalid_uuid_raises(self):
        """Verifies non-UUID subject inputs raise ValueError."""
        with pytest.raises(ValueError, match="Subject must be a valid UUID"):
            create_access_token(subject="not-a-uuid")

        with pytest.raises(ValueError, match="Subject must be a UUID"):
            create_access_token(subject=12345)  # type: ignore

    def test_create_access_token_non_positive_expiry_raises(self):
        """Verifies negative or zero expires_delta raises ValueError."""
        user_id = uuid.uuid4()
        with pytest.raises(ValueError, match="expires_delta must be a positive timedelta"):
            create_access_token(subject=user_id, expires_delta=timedelta(seconds=0))

        with pytest.raises(ValueError, match="expires_delta must be a positive timedelta"):
            create_access_token(subject=user_id, expires_delta=timedelta(seconds=-60))

    def test_create_access_token_exceeding_max_lifetime_raises(self):
        """Verifies expires_delta exceeding 24 hours raises ValueError."""
        user_id = uuid.uuid4()
        with pytest.raises(ValueError, match="cannot exceed maximum allowed lifetime"):
            create_access_token(subject=user_id, expires_delta=timedelta(hours=25))

    def test_create_access_token_with_additional_claims(self):
        """Verifies non-reserved custom claims are embedded in payload."""
        user_id = uuid.uuid4()
        token = create_access_token(
            subject=user_id,
            additional_claims={"tenant_role": "admin", "org_id": "lab-42"},
        )
        payload = decode_access_token(token)
        assert payload.extra_claims["tenant_role"] == "admin"
        assert payload.extra_claims["org_id"] == "lab-42"

    def test_create_access_token_reserved_claim_conflict_raises(self):
        """Verifies attempting to override reserved claims ('sub', 'exp', etc.) raises ValueError."""
        user_id = uuid.uuid4()
        with pytest.raises(ValueError, match="cannot override reserved token claims"):
            create_access_token(
                subject=user_id,
                additional_claims={"sub": "hijacked_id"},
            )

        with pytest.raises(ValueError, match="cannot override reserved token claims"):
            create_access_token(
                subject=user_id,
                additional_claims={"exp": 9999999999},
            )


class TestJWTDecodingAndValidation:
    """Tests for strict token decoding, signature verification, and claims validation."""

    def test_decode_access_token_success(self):
        """Verifies decoding a valid access token returns strongly-typed claims."""
        user_id = uuid.uuid4()
        token = create_access_token(subject=user_id)
        payload = decode_access_token(token)

        assert payload.sub == user_id
        assert payload.type == "access"
        assert payload.iat.tzinfo is not None
        assert payload.exp.tzinfo is not None
        assert payload.exp > payload.iat

    def test_decode_access_token_expired_raises_token_expired_error(self):
        """Verifies expired token raises TokenExpiredError."""
        user_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        expired_payload = {
            "sub": str(user_id),
            "type": "access",
            "iat": int((now - timedelta(minutes=10)).timestamp()),
            "exp": int((now - timedelta(minutes=5)).timestamp()),  # expired 5 min ago
        }
        secret = settings.get_jwt_secret_key()
        token = jwt.encode(expired_payload, secret, algorithm="HS256")

        with pytest.raises(TokenExpiredError, match="Access token has expired"):
            decode_access_token(token)

    def test_decode_access_token_future_iat_raises_invalid_token_error(self):
        """Verifies a correctly signed token with a future iat is rejected by decode_access_token."""
        user_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        future_payload = {
            "sub": str(user_id),
            "type": "access",
            "iat": int((now + timedelta(hours=1)).timestamp()),  # 1 hour in the future
            "exp": int((now + timedelta(hours=2)).timestamp()),
        }
        secret = settings.get_jwt_secret_key()
        token = jwt.encode(future_payload, secret, algorithm="HS256")

        with pytest.raises(InvalidTokenError, match="Token decoding failed"):
            decode_access_token(token)

    def test_decode_access_token_inverted_exp_iat_raises_invalid_token_error(self):
        """Verifies tokens where exp <= iat are rejected with InvalidTokenError."""
        user_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        # iat is valid (in past), but exp is earlier than iat
        inverted_payload = {
            "sub": str(user_id),
            "type": "access",
            "iat": int((now - timedelta(minutes=10)).timestamp()),
            "exp": int((now - timedelta(minutes=20)).timestamp()),  # exp <= iat
        }
        secret = settings.get_jwt_secret_key()
        token = jwt.encode(inverted_payload, secret, algorithm="HS256")

        # PyJWT ExpiredSignatureError or domain InvalidTokenError
        with pytest.raises((InvalidTokenError, TokenExpiredError)):
            decode_access_token(token)

    def test_decode_access_token_exceeding_max_lifetime_raises_invalid_token_error(self):
        """Verifies tokens with a lifetime exceeding 24 hours are rejected by decode_access_token."""
        user_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        # Token valid for 48 hours
        long_lived_payload = {
            "sub": str(user_id),
            "type": "access",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=48)).timestamp()),
        }
        secret = settings.get_jwt_secret_key()
        token = jwt.encode(long_lived_payload, secret, algorithm="HS256")

        with pytest.raises(InvalidTokenError, match="exceeds maximum allowed lifetime"):
            decode_access_token(token)

    def test_decode_access_token_wrong_secret_raises_invalid_token_error(self):
        """Verifies token signed with wrong secret raises InvalidTokenError."""
        user_id = uuid.uuid4()
        token = create_access_token(
            subject=user_id,
            secret_key="secret-key-one-which-is-at-least-32-chars-long",
        )

        with pytest.raises(InvalidTokenError, match="Token decoding failed"):
            decode_access_token(
                token,
                secret_key="secret-key-two-which-is-at-least-32-chars-long",
            )

    def test_decode_access_token_malformed_string_raises_invalid_token_error(self):
        """Verifies non-JWT malformed strings raise InvalidTokenError."""
        with pytest.raises(InvalidTokenError):
            decode_access_token("this.is.not.a.jwt")

        with pytest.raises(InvalidTokenError):
            decode_access_token("garbage_token")

    def test_decode_access_token_empty_or_none_raises_invalid_token_error(self):
        """Verifies empty or non-string token inputs raise InvalidTokenError."""
        with pytest.raises(InvalidTokenError, match="Token must be a non-empty string"):
            decode_access_token("")

        with pytest.raises(InvalidTokenError, match="Token must be a non-empty string"):
            decode_access_token("   ")

        with pytest.raises(InvalidTokenError, match="Token must be a non-empty string"):
            decode_access_token(None)  # type: ignore

    def test_decode_access_token_wrong_type_raises_invalid_token_error(self):
        """Verifies tokens with type other than 'access' (e.g. 'refresh') are rejected."""
        user_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        refresh_payload = {
            "sub": str(user_id),
            "type": "refresh",  # wrong type
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=1)).timestamp()),
        }
        secret = settings.get_jwt_secret_key()
        token = jwt.encode(refresh_payload, secret, algorithm="HS256")

        with pytest.raises(InvalidTokenError, match="Invalid token type: expected 'access', got 'refresh'"):
            decode_access_token(token)

    def test_decode_access_token_invalid_subject_raises_invalid_token_error(self):
        """Verifies tokens with a non-UUID subject fail strict validation."""
        now = datetime.now(timezone.utc)
        bad_sub_payload = {
            "sub": "not_a_valid_uuid_format",
            "type": "access",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=1)).timestamp()),
        }
        secret = settings.get_jwt_secret_key()
        token = jwt.encode(bad_sub_payload, secret, algorithm="HS256")

        with pytest.raises(InvalidTokenError, match="is not a valid UUID"):
            decode_access_token(token)

    def test_decode_access_token_wrong_algorithm_raises_invalid_token_error(self):
        """Verifies token signed with unconfigured algorithm (e.g. HS384 when HS256 expected) is rejected."""
        user_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        payload = {
            "sub": str(user_id),
            "type": "access",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=1)).timestamp()),
        }
        secret = settings.get_jwt_secret_key()
        # Sign with HS384 while settings.JWT_ALGORITHM is HS256
        token = jwt.encode(payload, secret, algorithm="HS384")

        with pytest.raises(InvalidTokenError, match="Token decoding failed"):
            decode_access_token(token)

    def test_decode_access_token_missing_required_claims_raises(self):
        """Verifies tokens missing standard claims ('sub', 'type', 'exp', 'iat') are rejected."""
        secret = settings.get_jwt_secret_key()

        # Missing 'sub'
        payload_no_sub = {
            "type": "access",
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
        }
        token1 = jwt.encode(payload_no_sub, secret, algorithm="HS256")
        with pytest.raises(InvalidTokenError):
            decode_access_token(token1)

        # Missing 'type'
        payload_no_type = {
            "sub": str(uuid.uuid4()),
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "exp": int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
        }
        token2 = jwt.encode(payload_no_type, secret, algorithm="HS256")
        with pytest.raises(InvalidTokenError):
            decode_access_token(token2)


class TestSecurityConfiguration:
    """Tests for JWT secret configuration, environment checks, and fail-closed validation."""

    def test_dev_environment_uses_default_when_unset(self, monkeypatch):
        """Verifies development environment falls back to documented default when secret is unset."""
        monkeypatch.setattr(settings, "ENVIRONMENT", "development")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", None)

        secret = settings.get_jwt_secret_key()
        assert secret == DEV_DEFAULT_JWT_SECRET

    def test_dev_environment_uses_configured_secret(self, monkeypatch):
        """Verifies development environment respects explicitly set secret."""
        custom_secret = "custom-development-secret-key-12345"
        monkeypatch.setattr(settings, "ENVIRONMENT", "development")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", custom_secret)

        secret = settings.get_jwt_secret_key()
        assert secret == custom_secret

    def test_test_environment_uses_default_when_unset(self, monkeypatch):
        """Verifies test environment falls back to documented default when secret is unset."""
        monkeypatch.setattr(settings, "ENVIRONMENT", "test")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", None)

        secret = settings.get_jwt_secret_key()
        assert secret == DEV_DEFAULT_JWT_SECRET

    def test_production_environment_missing_secret_fails_closed(self, monkeypatch):
        """Verifies production environment raises SecurityConfigurationError when secret is missing."""
        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", None)

        with pytest.raises(SecurityConfigurationError, match="JWT_SECRET_KEY is not configured"):
            settings.get_jwt_secret_key()

    def test_production_environment_empty_secret_fails_closed(self, monkeypatch):
        """Verifies production environment raises SecurityConfigurationError on empty/whitespace secret."""
        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", "   ")

        with pytest.raises(SecurityConfigurationError, match="JWT_SECRET_KEY is not configured"):
            settings.get_jwt_secret_key()

    def test_production_environment_dev_secret_fails_closed(self, monkeypatch):
        """Verifies production environment refuses to run with the development default secret."""
        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", DEV_DEFAULT_JWT_SECRET)

        with pytest.raises(SecurityConfigurationError, match="cannot use the default development secret"):
            settings.get_jwt_secret_key()

    def test_production_environment_short_secret_fails_closed(self, monkeypatch):
        """Verifies production environment requires at least 32 characters for adequate entropy."""
        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", "too-short-secret-under-32-chars")

        with pytest.raises(SecurityConfigurationError, match="must be at least 32 characters"):
            settings.get_jwt_secret_key()

    def test_production_environment_valid_secret_succeeds(self, monkeypatch):
        """Verifies production environment accepts a valid, strong 32+ character secret."""
        prod_secret = "a-very-strong-production-jwt-secret-key-with-256-bit-entropy"
        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", prod_secret)

        secret = settings.get_jwt_secret_key()
        assert secret == prod_secret

    def test_unrecognized_environment_fails_closed(self, monkeypatch):
        """Verifies unrecognized environments (e.g. 'staging', 'qa') fail closed like production."""
        monkeypatch.setattr(settings, "ENVIRONMENT", "staging")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", None)

        with pytest.raises(SecurityConfigurationError, match="JWT_SECRET_KEY is not configured"):
            settings.get_jwt_secret_key()

    def test_validate_jwt_settings_non_positive_expiry(self, monkeypatch):
        """Verifies validate_jwt_settings raises on zero or negative token expiry duration."""
        monkeypatch.setattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES", 0)

        with pytest.raises(SecurityConfigurationError, match="ACCESS_TOKEN_EXPIRE_MINUTES must be positive"):
            settings.validate_jwt_settings()

    def test_validate_jwt_settings_exceeding_24_hours_raises(self, monkeypatch):
        """Verifies validate_jwt_settings raises when expiry exceeds 1440 minutes (24 hours)."""
        monkeypatch.setattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES", 1500)

        with pytest.raises(SecurityConfigurationError, match="cannot exceed 1440 minutes"):
            settings.validate_jwt_settings()

    def test_validate_jwt_settings_unsupported_algorithm(self, monkeypatch):
        """Verifies validate_jwt_settings rejects unsupported or dangerous algorithms like 'none'."""
        monkeypatch.setattr(settings, "JWT_ALGORITHM", "none")

        with pytest.raises(SecurityConfigurationError, match="Unsupported JWT_ALGORITHM"):
            settings.validate_jwt_settings()


class TestApplicationStartupSecurity:
    """Tests proving FastAPI application startup eagerly rejects invalid production configuration."""

    def test_startup_rejects_missing_secret_in_production(self, monkeypatch):
        """Verifies application startup fails when JWT_SECRET_KEY is missing in production."""
        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", None)

        with pytest.raises(SecurityConfigurationError, match="JWT_SECRET_KEY is not configured"):
            create_application()

    def test_startup_rejects_empty_secret_in_production(self, monkeypatch):
        """Verifies application startup fails when JWT_SECRET_KEY is empty/whitespace in production."""
        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", "   ")

        with pytest.raises(SecurityConfigurationError, match="JWT_SECRET_KEY is not configured"):
            create_application()

    def test_startup_rejects_dev_default_secret_in_production(self, monkeypatch):
        """Verifies application startup fails when JWT_SECRET_KEY uses development default in production."""
        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", DEV_DEFAULT_JWT_SECRET)

        with pytest.raises(SecurityConfigurationError, match="cannot use the default development secret"):
            create_application()

    def test_startup_rejects_weak_short_secret_in_production(self, monkeypatch):
        """Verifies application startup fails when JWT_SECRET_KEY is under 32 characters in production."""
        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", "short-key-under-32-chars")

        with pytest.raises(SecurityConfigurationError, match="must be at least 32 characters"):
            create_application()

    def test_startup_succeeds_with_strong_secret_in_production(self, monkeypatch):
        """Verifies application startup succeeds in production when a strong 32+ character key is provided."""
        prod_secret = "a-very-strong-production-jwt-secret-key-with-256-bit-entropy"
        monkeypatch.setattr(settings, "ENVIRONMENT", "production")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", prod_secret)

        app = create_application()
        assert app is not None
        assert app.title == settings.PROJECT_NAME

    def test_startup_succeeds_in_development_without_configured_secret(self, monkeypatch):
        """Verifies application startup succeeds in development without a configured secret (using dev fallback)."""
        monkeypatch.setattr(settings, "ENVIRONMENT", "development")
        monkeypatch.setattr(settings, "JWT_SECRET_KEY", None)

        app = create_application()
        assert app is not None
        assert app.title == settings.PROJECT_NAME
