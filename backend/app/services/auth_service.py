"""Authentication service module for ResearchLens AI.

Architecture & Security Principles:
1. Timing Attack Mitigation:
   - Always verify passwords against DUMMY_PASSWORD_HASH when no user matches the email,
     when email normalization fails, or when credentials are unknown.
   - Ensures constant verification latency regardless of account existence.
2. Safe Failure & Information Disclosure:
   - Login failures return a generic error message ('Invalid email or password.')
     for nonexistent users, bad passwords, and inactive accounts alike.
   - Inactive accounts fail with InactiveUserError (a subclass of InvalidCredentialsError)
     preserving identical client-facing messages without revealing account existence.
3. Clean State & Concurrency Safety:
   - Email uniqueness races are caught via IntegrityError, followed by an immediate
     session.rollback() to leave the SQLAlchemy session in a clean transaction state.
4. Credential & Secret Protection:
   - Passwords and token payloads are never written to application logs or error messages.
   - Password hashes are never exposed through service responses or user DTOs.
"""
from typing import Any, Dict, Optional, Union
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import (
    DUMMY_PASSWORD_HASH,
    create_access_token,
    hash_password,
    verify_password,
)
from app.models.user import User
from app.schemas.auth import (
    AuthResult,
    InvalidEmailError,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
    normalize_email,
)



# --- Domain Exceptions ---

class AuthServiceError(Exception):
    """Base exception for all authentication service errors."""
    pass


class RegistrationError(AuthServiceError):
    """Base exception for user registration errors."""
    pass


class DuplicateEmailError(RegistrationError):
    """Raised when an email address is already registered."""
    pass


class WeakPasswordError(RegistrationError):
    """Raised when a password does not meet complexity or length constraints."""
    pass


class AuthenticationError(AuthServiceError):
    """Base exception for authentication failures."""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Raised when authentication fails due to incorrect credentials or unknown account.

    Security Note: Message must remain generic ('Invalid email or password.')
    to avoid account enumeration side-channels.
    """
    DEFAULT_MESSAGE = "Invalid email or password."

    def __init__(self, message: Optional[str] = None):
        super().__init__(message or self.DEFAULT_MESSAGE)


class InactiveUserError(InvalidCredentialsError):
    """Raised when an authenticating user account is deactivated or inactive.

    Inherits from InvalidCredentialsError so callers catching InvalidCredentialsError
    automatically handle inactive accounts with identical generic messaging.
    """
    pass


# --- Helper Validation Functions ---



def validate_password_complexity(password: str) -> None:
    """Validates password length and UTF-8 byte bounds for bcrypt compatibility.

    Args:
        password: Plain text password string.

    Raises:
        WeakPasswordError: If password is empty, fewer than 8 characters, or exceeds 72 UTF-8 bytes.
    """
    if not password or not isinstance(password, str):
        raise WeakPasswordError("Password must be a non-empty string.")

    if len(password) < 8:
        raise WeakPasswordError("Password must be at least 8 characters long.")

    encoded = password.encode("utf-8")
    if len(encoded) > 72:
        raise WeakPasswordError("Password cannot exceed 72 bytes due to bcrypt constraints.")


# --- AuthService Implementation ---

class AuthService:
    """Service providing secure user registration, credential authentication, and user lookup."""

    def register_user(
        self,
        db: Session,
        request: Union[UserRegisterRequest, Dict[str, Any]],
    ) -> UserResponse:
        """Registers a new user account with a securely hashed password.

        Args:
            db: Active SQLAlchemy database session.
            request: UserRegisterRequest instance or dictionary containing 'email' and 'password'.

        Returns:
            UserResponse containing public user account information (excluding password hash).

        Raises:
            WeakPasswordError: If the password does not meet length/byte constraints.
            InvalidEmailError: If the email address format is invalid.
            DuplicateEmailError: If the normalized email is already registered.
            RegistrationError: For other registration failures.
        """
        if isinstance(request, dict):
            try:
                req = UserRegisterRequest.model_validate(request)
            except ValidationError as exc:
                for err in exc.errors():
                    loc = err.get("loc", ())
                    if "password" in loc:
                        raise WeakPasswordError(f"Invalid password: {err.get('msg')}") from exc
                    if "email" in loc:
                        raise InvalidEmailError(f"Invalid email: {err.get('msg')}") from exc
                raise RegistrationError(f"Registration validation failed: {exc}") from exc
        elif isinstance(request, UserRegisterRequest):
            req = request
        else:
            raise ValueError(f"Expected UserRegisterRequest or dict, got {type(request).__name__}")

        # Normalize and validate email
        normalized_email = normalize_email(req.email)

        # Validate password complexity
        validate_password_complexity(req.password)

        # Pre-check for duplicate email
        existing_user = db.query(User).filter(User.email == normalized_email).first()
        if existing_user is not None:
            raise DuplicateEmailError("An account with this email already exists.")

        # Hash password securely using existing core security utility
        hashed_password = hash_password(req.password)

        # Create new user entity
        new_user = User(
            email=normalized_email,
            hashed_password=hashed_password,
            is_active=True,
            is_verified=False,
        )
        db.add(new_user)

        # Commit with transaction rollback safety on race conditions
        try:
            db.commit()
            db.refresh(new_user)
        except IntegrityError as exc:
            db.rollback()
            raise DuplicateEmailError("An account with this email already exists.") from exc
        except Exception:
            db.rollback()
            raise

        return UserResponse.model_validate(new_user)

    def authenticate_user(
        self,
        db: Session,
        request: Union[UserLoginRequest, Dict[str, Any]],
    ) -> AuthResult:
        """Authenticates user credentials and issues a signed JWT access token.

        Guarantees constant-time verification latency against DUMMY_PASSWORD_HASH
        when the account is not found, when email format is invalid, or when credentials fail.

        Args:
            db: Active SQLAlchemy database session.
            request: UserLoginRequest instance or dictionary containing 'email' and 'password'.

        Returns:
            AuthResult containing public user profile and signed JWT TokenResponse.

        Raises:
            InvalidCredentialsError: On incorrect credentials or nonexistent user ('Invalid email or password.').
            InactiveUserError: If user account is deactivated ('Invalid email or password.').
        """
        if isinstance(request, dict):
            try:
                req = UserLoginRequest.model_validate(request)
            except ValidationError:
                # Perform dummy verification to maintain constant timing on malformed payloads
                raw_pw = str(request.get("password", "") if isinstance(request, dict) else "")
                verify_password(raw_pw, DUMMY_PASSWORD_HASH)
                raise InvalidCredentialsError("Invalid email or password.")
        elif isinstance(request, UserLoginRequest):
            req = request
        else:
            raise ValueError(f"Expected UserLoginRequest or dict, got {type(request).__name__}")

        raw_email = req.email
        password = req.password

        # Normalize email. If format is invalid, perform dummy verification and fail generically
        try:
            normalized_email = normalize_email(raw_email)
        except InvalidEmailError:
            verify_password(password, DUMMY_PASSWORD_HASH)
            raise InvalidCredentialsError("Invalid email or password.")

        # Query user by normalized email
        user = db.query(User).filter(User.email == normalized_email).first()

        # If user is not found, verify against dummy hash to prevent username enumeration timing attacks
        if user is None:
            verify_password(password, DUMMY_PASSWORD_HASH)
            raise InvalidCredentialsError("Invalid email or password.")

        # Found user: verify submitted password against stored hash
        password_matches = verify_password(password, user.hashed_password)
        if not password_matches:
            raise InvalidCredentialsError("Invalid email or password.")

        # Check account activation status
        if not user.is_active:
            raise InactiveUserError("Invalid email or password.")

        # Issue access token
        access_token = create_access_token(subject=user.id)
        expires_in = settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60

        return AuthResult(
            user=UserResponse.model_validate(user),
            token=TokenResponse(
                access_token=access_token,
                token_type="bearer",
                expires_in=expires_in,
            ),
        )

    def get_user_by_id(self, db: Session, user_id: UUID) -> Optional[User]:
        """Looks up a user by primary key UUID."""
        return db.query(User).filter(User.id == user_id).first()

    def get_user_by_email(self, db: Session, email: str) -> Optional[User]:
        """Looks up a user by normalized email address."""
        try:
            normalized = normalize_email(email)
        except InvalidEmailError:
            return None
        return db.query(User).filter(User.email == normalized).first()


# Export default singleton service instance
auth_service = AuthService()
