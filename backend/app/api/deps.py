"""FastAPI dependencies for authentication, database sessions, and tenant context."""
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import (
    InvalidTokenError,
    TokenError,
    TokenExpiredError,
    decode_access_token,
)
from app.models.user import User

# Standard HTTP Bearer token scheme for OpenAPI documentation
oauth2_scheme = HTTPBearer(
    auto_error=False,
    description="JWT Bearer token in 'Authorization: Bearer <token>' format",
)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Extracts, validates, and decodes the JWT access token and resolves the active User.

    Enforces:
    - Presence of Bearer token in Authorization header.
    - Token signature and structural validity via decode_access_token.
    - Expiration and lifetime bounds.
    - UUID subject lookup in database.
    - Active account status (is_active == True).

    Raises:
        HTTPException(401): If token is missing, expired, invalid, or user is inactive/not found.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials.strip()

    try:
        payload = decode_access_token(token)
    except (TokenError, Exception):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == payload.sub).first()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Alias dependency for explicit semantic active user verification."""
    return current_user
