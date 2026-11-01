"""Pydantic schemas for authentication, credentials, and user account representations."""
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, field_validator


class InvalidEmailError(ValueError):
    """Raised when an email address format is invalid."""
    pass


def normalize_email(email: str) -> str:
    """Normalizes and validates an email address.

    Trims leading/trailing whitespace, converts to lowercase, and verifies basic structure.

    Args:
        email: Raw email string.

    Returns:
        Normalized email string (trimmed, lowercased).

    Raises:
        InvalidEmailError: If the email is empty, contains whitespace, or has an invalid structure.
    """
    if not email or not isinstance(email, str):
        raise InvalidEmailError("Email must be a non-empty string.")

    cleaned = email.strip().lower()
    if not cleaned:
        raise InvalidEmailError("Email cannot be empty or whitespace-only.")

    if any(c.isspace() for c in cleaned):
        raise InvalidEmailError("Email cannot contain whitespace characters.")

    parts = cleaned.split("@")
    if len(parts) != 2:
        raise InvalidEmailError("Email must contain exactly one '@' separator.")

    local_part, domain_part = parts
    if not local_part or not domain_part:
        raise InvalidEmailError("Email must have non-empty username and domain parts.")

    if "." not in domain_part or domain_part.startswith(".") or domain_part.endswith("."):
        raise InvalidEmailError("Email domain must contain a valid top-level domain.")

    if len(cleaned) > 320:
        raise InvalidEmailError("Email address cannot exceed 320 characters.")

    return cleaned


class UserRegisterRequest(BaseModel):
    """Payload for registering a new user account."""
    model_config = ConfigDict(frozen=True)

    email: str = Field(
        ...,
        min_length=3,
        max_length=320,
        description="User email address",
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=72,
        description="Plain text password (8-72 characters, max 72 UTF-8 bytes)",
    )

    @field_validator("email")
    @classmethod
    def validate_and_normalize_email(cls, v: str) -> str:
        return normalize_email(v)

    @field_validator("password")
    @classmethod
    def validate_password_bytes(cls, v: str) -> str:
        if len(v.encode("utf-8")) > 72:
            raise ValueError("Password cannot exceed 72 bytes due to bcrypt constraints.")
        return v



class UserLoginRequest(BaseModel):
    """Payload for authenticating existing user credentials."""
    model_config = ConfigDict(frozen=True)

    email: str = Field(
        ...,
        min_length=1,
        max_length=320,
        description="User email address",
    )
    password: str = Field(
        ...,
        min_length=1,
        max_length=72,
        description="Plain text password",
    )


class UserResponse(BaseModel):
    """Public user account details, excluding credential hashes."""
    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: UUID = Field(..., description="Unique user identifier")
    email: str = Field(..., description="Normalized user email address")
    is_active: bool = Field(..., description="Whether user account is active")
    is_verified: bool = Field(..., description="Whether user email is verified")
    created_at: datetime = Field(..., description="Account creation timestamp")
    updated_at: datetime = Field(..., description="Last account update timestamp")


class TokenResponse(BaseModel):
    """JWT access token response model."""
    model_config = ConfigDict(frozen=True)

    access_token: str = Field(..., description="Signed JWT access token")
    token_type: str = Field(default="bearer", description="Token authorization type")
    expires_in: int = Field(..., gt=0, description="Token lifetime in seconds")


class AuthResult(BaseModel):
    """Result of a successful authentication operation."""
    model_config = ConfigDict(frozen=True)

    user: UserResponse = Field(..., description="Authenticated user account details")
    token: TokenResponse = Field(..., description="Issued access token details")
