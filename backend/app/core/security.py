"""Core security module for ResearchLens AI providing password hashing and JWT lifecycle management.

Architecture & Security Principles:
1. Modern Password Hashing:
   - Uses pwdlib with BcryptHasher to guarantee modern, clean Python 3.13 compatibility
     without deprecated standard library dependencies.
   - Salt generation and work factors (rounds=12 default) are managed safely per hash.
   - Bcrypt's 72-byte ceiling is strictly validated on UTF-8 encoded bytes to prevent silent truncation.
2. Strict JWT Verification:
   - Tokens are cryptographically signed with HMAC-SHA256 (HS256).
   - Signatures, issuance (iat), and expiration (exp) are strictly verified.
   - Tokens with future iat, exp <= iat, or lifetime exceeding 24 hours (1440m) are rejected.
   - Expected token type is strictly enforced ('access').
   - Subject claim (sub) is strictly validated as a well-formed UUID.
3. Fail-Closed Security Exceptions:
   - Security-specific exceptions provide predictable, typed errors.
   - Passwords, plaintext credentials, secrets, and raw hashes are never logged.
4. Timing Attack Mitigation for Authentication (Milestone 1B):
   - When querying users during login, the service must verify plain passwords against
     DUMMY_PASSWORD_HASH if no matching user is found, ensuring constant-time latency.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Union
from uuid import UUID
import jwt
from pydantic import BaseModel, Field, model_validator
from pwdlib import PasswordHash
from pwdlib.hashers.bcrypt import BcryptHasher

from app.core.config import settings

# --- Domain Constants ---

# Maximum allowed access token lifetime (24 hours)
MAX_ACCESS_TOKEN_LIFETIME = timedelta(hours=24)

# Milestone 1B Security Requirement:
# To prevent username-enumeration timing side-channels during authentication, the upcoming
# authentication service (Milestone 1B) must verify the submitted password against
# DUMMY_PASSWORD_HASH whenever an email lookup returns no user. This guarantees that
# password verification latency (~70-100ms) remains constant regardless of whether
# the user account exists.
DUMMY_PASSWORD_HASH = "$2b$12$lI1lRhu1FGovstdbSUS/RuE/TLmuk6hLaTlrfKMFHAfWUW9PQerp."


# --- Domain Exceptions ---

class SecurityError(Exception):
    """Base exception for all security layer failures."""
    pass


class SecurityConfigurationError(SecurityError):
    """Raised when security or JWT configuration is invalid, missing, or unsafe."""
    pass


class TokenError(SecurityError):
    """Base exception for all token-related failures."""
    pass


class TokenExpiredError(TokenError):
    """Raised when a valid token has passed its expiration time."""
    pass


class InvalidTokenError(TokenError):
    """Raised when a token fails signature, claim, algorithm, or structural validation."""
    pass


# --- Token Payload Model ---

class TokenPayload(BaseModel):
    """Structured, validated claims extracted from an authenticated JWT access token."""
    sub: UUID = Field(..., description="Authenticated user ID as valid UUID")
    type: str = Field(..., description="Token type, must be 'access'")
    iat: datetime = Field(..., description="Timestamp when token was issued (timezone-aware UTC)")
    exp: datetime = Field(..., description="Timestamp when token expires (timezone-aware UTC)")
    extra_claims: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional non-reserved claims carried in the token payload",
    )

    @model_validator(mode="after")
    def validate_lifetime(self) -> "TokenPayload":
        """Ensures expiration is strictly after issuance and within maximum access-token lifetime."""
        if self.exp <= self.iat:
            raise ValueError(
                f"Invalid token lifetime: expiration ({self.exp.isoformat()}) must be strictly after issuance ({self.iat.isoformat()})."
            )
        if (self.exp - self.iat) > MAX_ACCESS_TOKEN_LIFETIME:
            raise ValueError(
                f"Token lifetime of {(self.exp - self.iat)} exceeds maximum allowed lifetime of {MAX_ACCESS_TOKEN_LIFETIME}."
            )
        return self


# --- Password Hashing Setup ---

_password_hasher = PasswordHash((BcryptHasher(),))


def hash_password(plain_password: str) -> str:
    """Computes a secure bcrypt hash for the provided plain text password.

    Args:
        plain_password: Raw password string.

    Returns:
        Securely hashed string representation including salt and parameters.

    Raises:
        ValueError: If password is empty, non-string, or exceeds bcrypt's 72-byte limit.
    """
    if not plain_password or not isinstance(plain_password, str):
        raise ValueError("Password must be a non-empty string.")

    encoded = plain_password.encode("utf-8")
    if len(encoded) > 72:
        raise ValueError("Password cannot exceed 72 bytes due to bcrypt algorithmic constraints.")

    return _password_hasher.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain text password against a stored bcrypt hash.

    Args:
        plain_password: Raw password string provided during authentication.
        hashed_password: Stored hash from the database.

    Returns:
        True if the password matches the hash; False otherwise.
        Fails safely and returns False on malformed inputs without throwing exceptions.
    """
    if not plain_password or not isinstance(plain_password, str):
        return False
    if not hashed_password or not isinstance(hashed_password, str):
        return False

    try:
        return _password_hasher.verify(plain_password, hashed_password)
    except Exception:
        # Fail closed on corrupted hashes or unexpected verification failures
        return False


# --- JWT Token Lifecycle ---

def create_access_token(
    subject: Union[UUID, str],
    expires_delta: Optional[timedelta] = None,
    additional_claims: Optional[Dict[str, Any]] = None,
    secret_key: Optional[str] = None,
) -> str:
    """Generates a signed JWT access token for an authenticated user subject.

    Args:
        subject: Target user UUID (as UUID instance or valid UUID string).
        expires_delta: Optional custom lifetime duration. Defaults to settings.ACCESS_TOKEN_EXPIRE_MINUTES.
        additional_claims: Optional dictionary of extra non-reserved claims to embed.
        secret_key: Optional explicit signing secret. Defaults to settings.get_jwt_secret_key().

    Returns:
        Compact signed JWT string.

    Raises:
        ValueError: If subject is not a valid UUID, expires_delta is non-positive or exceeds 24h,
                    or additional_claims attempts to overwrite reserved claims.
        SecurityConfigurationError: If secret_key is unconfigured or unsafe in production.
    """
    if isinstance(subject, str):
        try:
            subject_uuid = UUID(subject)
        except (ValueError, TypeError, AttributeError):
            raise ValueError(f"Subject must be a valid UUID, got {subject!r}") from None
    elif isinstance(subject, UUID):
        subject_uuid = subject
    else:
        raise ValueError(f"Subject must be a UUID or valid UUID string, got {type(subject).__name__}")

    now = datetime.now(timezone.utc)

    if expires_delta is not None:
        if not isinstance(expires_delta, timedelta) or expires_delta.total_seconds() <= 0:
            raise ValueError("expires_delta must be a positive timedelta.")
        if expires_delta > MAX_ACCESS_TOKEN_LIFETIME:
            raise ValueError(f"expires_delta cannot exceed maximum allowed lifetime of {MAX_ACCESS_TOKEN_LIFETIME}.")
        exp_dt = now + expires_delta
    else:
        configured_lifetime = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        if configured_lifetime > MAX_ACCESS_TOKEN_LIFETIME:
            raise ValueError(
                f"Configured ACCESS_TOKEN_EXPIRE_MINUTES ({settings.ACCESS_TOKEN_EXPIRE_MINUTES}m) "
                f"exceeds maximum allowed lifetime of {MAX_ACCESS_TOKEN_LIFETIME}."
            )
        exp_dt = now + configured_lifetime

    payload: Dict[str, Any] = {
        "sub": str(subject_uuid),
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(exp_dt.timestamp()),
    }

    if additional_claims:
        reserved_keys = {"sub", "type", "iat", "exp"}
        conflicts = reserved_keys.intersection(additional_claims.keys())
        if conflicts:
            raise ValueError(f"additional_claims cannot override reserved token claims: {sorted(list(conflicts))}")
        payload.update(additional_claims)

    secret = secret_key or settings.get_jwt_secret_key()
    algorithm = settings.JWT_ALGORITHM

    token = jwt.encode(payload, secret, algorithm=algorithm)
    return token


def decode_access_token(
    token: str,
    secret_key: Optional[str] = None,
) -> TokenPayload:
    """Decodes, validates, and parses a signed JWT access token.

    Enforces:
    - Signature validity using configured algorithm.
    - Token expiration (exp) and issued-at (iat) validity.
    - Token lifetime bounds (exp strictly after iat, maximum lifetime 24 hours).
    - Required standard claims presence ('sub', 'type', 'exp', 'iat').
    - Strict access token type ('type == "access"').
    - Well-formed UUID subject format.

    Args:
        token: Compact JWT string.
        secret_key: Optional explicit secret key. Defaults to settings.get_jwt_secret_key().

    Returns:
        TokenPayload containing typed and validated claims.

    Raises:
        TokenExpiredError: If the token signature is valid but expiration time has passed.
        InvalidTokenError: If the token is malformed, signature is invalid, algorithm mismatches,
                           claims are missing, token type is not 'access', subject is not a UUID,
                           or token lifetime is invalid / exceeds 24 hours.
        SecurityConfigurationError: If the signing secret is improperly configured.
    """
    if not token or not isinstance(token, str) or not token.strip():
        raise InvalidTokenError("Token must be a non-empty string.")

    secret = secret_key or settings.get_jwt_secret_key()
    algorithm = settings.JWT_ALGORITHM

    try:
        raw_payload = jwt.decode(
            token.strip(),
            secret,
            algorithms=[algorithm],
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "require": ["sub", "type", "exp", "iat"],
            },
        )
    except jwt.ExpiredSignatureError:
        raise TokenExpiredError("Access token has expired.") from None
    except jwt.PyJWTError as e:
        raise InvalidTokenError(f"Token decoding failed: {e}") from None

    # Enforce strict token type
    token_type = raw_payload.get("type")
    if token_type != "access":
        raise InvalidTokenError(f"Invalid token type: expected 'access', got '{token_type}'.")

    # Enforce valid UUID subject
    sub_raw = raw_payload.get("sub")
    try:
        sub_uuid = UUID(str(sub_raw))
    except (ValueError, TypeError, AttributeError):
        raise InvalidTokenError(f"Invalid token subject: '{sub_raw}' is not a valid UUID.") from None

    iat_dt = datetime.fromtimestamp(raw_payload["iat"], tz=timezone.utc)
    exp_dt = datetime.fromtimestamp(raw_payload["exp"], tz=timezone.utc)

    # Enforce strict lifetime bounds
    if exp_dt <= iat_dt:
        raise InvalidTokenError(
            f"Invalid token lifetime: expiration ({exp_dt.isoformat()}) must be strictly after issued-at ({iat_dt.isoformat()})."
        )
    if (exp_dt - iat_dt) > MAX_ACCESS_TOKEN_LIFETIME:
        raise InvalidTokenError(
            f"Token lifetime of {(exp_dt - iat_dt)} exceeds maximum allowed lifetime of {MAX_ACCESS_TOKEN_LIFETIME}."
        )

    extra_claims = {
        k: v for k, v in raw_payload.items()
        if k not in ("sub", "type", "iat", "exp")
    }

    try:
        return TokenPayload(
            sub=sub_uuid,
            type="access",
            iat=iat_dt,
            exp=exp_dt,
            extra_claims=extra_claims,
        )
    except Exception as e:
        raise InvalidTokenError(f"Token payload validation failed: {e}") from None
