"""Database connection and session lifecycle management for ResearchLens AI.

Architecture & Design Principles:
1. Lazy Initialization: Importing this module does NOT connect to the database.
   Connections are only checked out from the pool when an actual database operation occurs.
2. Safe Credentials: Passwords in database URLs are never logged or exposed in error messages.
3. Explicit Session Lifecycle:
   - FastAPI dependency `get_db()` provides request-scoped sessions with guaranteed cleanup.
   - Context manager `db_session_scope()` provides transactional boundaries with automatic commit/rollback.
4. Safe Engine Defaults:
   - `pool_pre_ping=True`: Verifies connection liveness before checkout to prevent stale socket errors.
   - `pool_size=5`, `max_overflow=10`, `pool_recycle=1800`: Conservative connection pooling.
   - `expire_on_commit=False`: Preserves model state after commit for clean serialization.
"""
from contextlib import contextmanager
import logging
import re
from typing import Generator, Optional
from sqlalchemy import Engine, create_engine, pool
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for all database-related failures."""
    pass


class DatabaseConfigurationError(DatabaseError):
    """Raised when DATABASE_URL is missing, unconfigured, or improperly formatted."""
    pass


def mask_database_url(url: Optional[str]) -> str:
    """Returns a sanitized version of the database URL with credentials hidden.

    Never prints or logs unmasked passwords.
    """
    if not url:
        return "<not configured>"
    try:
        sa_url = make_url(url)
        return sa_url.render_as_string(hide_password=True)
    except Exception:
        # Fallback to greedy regex masking if SQLAlchemy make_url fails to parse the string
        sanitized = re.sub(r"(://[^:]+:)(.*)(@[^@/]+)", r"\1***\3", str(url))
        if "***@" in sanitized:
            return sanitized
        return "<malformed database URL>"


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy 2.0 ORM models in ResearchLens AI."""
    pass


def create_db_engine(database_url: Optional[str] = None, **engine_kwargs) -> Engine:
    """Creates a SQLAlchemy Engine configured with safe connection-pooling defaults.

    Args:
        database_url: Optional explicit connection string. Defaults to settings.DATABASE_URL.
        **engine_kwargs: Additional keyword arguments forwarded to create_engine.

    Returns:
        SQLAlchemy Engine instance.

    Raises:
        DatabaseConfigurationError: If no database URL is configured or if the URL is invalid.
    """
    url_str = database_url or settings.DATABASE_URL
    if not url_str or not url_str.strip():
        raise DatabaseConfigurationError(
            "DATABASE_URL is not configured. Set DATABASE_URL in your .env file or environment variables."
        )

    try:
        parsed_url = make_url(url_str)
    except Exception as e:
        safe_msg = re.sub(r"(://[^:]+:)(.*)(@[^@/]+)", r"\1***\3", str(e))
        raise DatabaseConfigurationError(
            f"Failed to parse database connection URL: {safe_msg}"
        ) from None

    # Safe defaults tailored to database dialect
    kwargs = {
        "echo": False,  # Never leak queries or parameter values to stdout by default
    }

    if parsed_url.get_backend_name() == "sqlite":
        # SQLite specific settings (e.g. for testing)
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # Production PostgreSQL pooling defaults
        kwargs.update({
            "pool_pre_ping": True,    # Test connection health before using from pool
            "pool_recycle": 1800,     # Recycle connections every 30m to avoid timeout drops
        })
        # pool_size and max_overflow are only supported on QueuePool, not NullPool
        poolclass = engine_kwargs.get("poolclass")
        if poolclass is None or not (isinstance(poolclass, type) and issubclass(poolclass, pool.NullPool)):
            kwargs["pool_size"] = 5
            kwargs["max_overflow"] = 10

    # Allow caller kwargs to override defaults (useful for testing, e.g. StaticPool)
    kwargs.update(engine_kwargs)

    try:
        return create_engine(parsed_url, **kwargs)
    except Exception as e:
        safe_msg = re.sub(r"(://[^:]+:)(.*)(@[^@/]+)", r"\1***\3", str(e))
        raise DatabaseConfigurationError(
            f"Failed to create database engine for '{mask_database_url(url_str)}': {safe_msg}"
        ) from None


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Creates a sessionmaker configured with safe transactional defaults.

    - autocommit=False: Requires explicit commits/rollbacks (SQLAlchemy 2.0 standard).
    - autoflush=False: Avoids premature flushing before query boundaries.
    - expire_on_commit=False: Keeps committed object attributes accessible without re-querying.
    """
    return sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )


# --- Module-Level Singletons (Lazy Initialized) ---

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker[Session]] = None


def get_engine(database_url: Optional[str] = None) -> Engine:
    """Returns the globally cached engine, creating it lazily if not yet initialized."""
    global _engine
    if _engine is None:
        _engine = create_db_engine(database_url)
    return _engine


def get_session_factory(database_url: Optional[str] = None) -> sessionmaker[Session]:
    """Returns the globally cached session factory, creating it lazily if not yet initialized."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine(database_url)
        _session_factory = create_session_factory(engine)
    return _session_factory


def reset_database_state() -> None:
    """Disposes of the current engine connection pool and resets global singletons.

    Primarily intended for test fixtures and environment reconfiguration.
    """
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
        _engine = None
    _session_factory = None


# --- Session Lifecycle Helpers ---

@contextmanager
def db_session_scope(
    session_factory: Optional[sessionmaker[Session]] = None,
) -> Generator[Session, None, None]:
    """Transactional context manager for explicit session scope.

    Commits on normal exit, rolls back on any exception, and guarantees closure.
    Ideal for background scripts, tasks, and batch operations.
    """
    factory = session_factory or get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding an active database session for a single request.

    Guarantees cleanup and connection return to pool after request completion.
    Rolls back if an unhandled exception occurs.
    """
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
