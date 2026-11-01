"""Unit tests for database configuration, engine creation, and session lifecycle.

Design Note:
These tests use a temporary SQLite file or SQLite memory engine to verify session
lifecycle, dependency injection, exception rollback, and cleanup mechanics.
As noted in the project architecture, testing with SQLite validates SQLAlchemy
lifecycle semantics but does NOT prove PostgreSQL-specific dialect compatibility
or live database availability. No running PostgreSQL server is required.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy import pool, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.core.database import (
    Base,
    DatabaseConfigurationError,
    DatabaseError,
    create_db_engine,
    create_session_factory,
    db_session_scope,
    get_db,
    get_engine,
    get_session_factory,
    mask_database_url,
    reset_database_state,
)


@pytest.fixture(autouse=True)
def clean_database_state():
    """Ensures a clean global database state before and after every test."""
    reset_database_state()
    yield
    reset_database_state()


@pytest.fixture
def test_engine(tmp_path: Path):
    """Provides a temporary file-backed SQLite engine for test isolation."""
    db_path = tmp_path / "test.db"
    engine = create_db_engine(f"sqlite:///{db_path}")
    # Create a simple table for verifying transaction semantics
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT NOT NULL)"))
    yield engine
    engine.dispose()


@pytest.fixture
def test_session_factory(test_engine):
    """Provides a session factory bound to the test engine."""
    return create_session_factory(test_engine)


class TestDatabaseURLMasking:
    """Tests verifying credentials in connection URLs are never exposed in logs or errors."""

    def test_mask_postgresql_url_with_password(self):
        """Verifies passwords in PostgreSQL URLs are masked with asterisks."""
        url = "postgresql://dbuser:super_secret_pw123@dbhost.internal:5432/researchlens"
        masked = mask_database_url(url)
        assert "super_secret_pw123" not in masked
        assert masked == "postgresql://dbuser:***@dbhost.internal:5432/researchlens"

    def test_mask_psycopg3_url_with_password(self):
        """Verifies psycopg 3 driver URLs are also correctly masked."""
        url = "postgresql+psycopg://app_user:another_secret@localhost:5432/proddb"
        masked = mask_database_url(url)
        assert "another_secret" not in masked
        assert masked == "postgresql+psycopg://app_user:***@localhost:5432/proddb"

    def test_mask_none_or_empty_url(self):
        """Verifies None or empty URLs return safe placeholders."""
        assert mask_database_url(None) == "<not configured>"
        assert mask_database_url("") == "<not configured>"

    def test_mask_malformed_url_does_not_crash(self):
        """Verifies completely malformed inputs do not raise unhandled exceptions."""
        assert mask_database_url("not-a-valid-url:::") == "<malformed database URL>"

    def test_mask_malformed_url_with_password_uses_regex_fallback(self):
        """Verifies passwords in structurally unparseable URLs are still masked via fallback regex."""
        bad_url = "postgresql+psycopg://dbuser:super_secret_pw@badhost:notaport/db"
        masked = mask_database_url(bad_url)
        assert "super_secret_pw" not in masked
        assert "dbuser:***@badhost:notaport/db" in masked

    def test_mask_url_with_special_characters_encoded_and_raw(self):
        """Verifies passwords containing %, @, : (both URL-encoded and raw) are never exposed."""
        # 1. URL-encoded password for 'p@ss%word:123' -> 'p%40ss%25word%3A123'
        encoded_pw = "p%40ss%25word%3A123"
        url1 = f"postgresql+psycopg://myuser:{encoded_pw}@localhost:5432/mydb"
        masked1 = mask_database_url(url1)
        assert encoded_pw not in masked1
        assert "p@ss%word:123" not in masked1
        assert masked1 == "postgresql+psycopg://myuser:***@localhost:5432/mydb"

        # 2. Raw special characters in password causing parse issues handled safely
        raw_pw_url = "postgresql+psycopg://myuser:raw@pass:word%123@localhost:5432/mydb"
        masked2 = mask_database_url(raw_pw_url)
        assert "pass" not in masked2
        assert "word" not in masked2
        assert "***@" in masked2




class TestDatabaseConfigurationAndEngine:
    """Tests for engine creation, configuration validation, and safe defaults."""

    def test_missing_database_url_raises_configuration_error(self, monkeypatch):
        """Verifies calling create_db_engine without a configured URL raises DatabaseConfigurationError."""
        # Ensure settings.DATABASE_URL is None
        monkeypatch.setattr("app.core.database.settings.DATABASE_URL", None)

        with pytest.raises(DatabaseConfigurationError, match="DATABASE_URL is not configured"):
            create_db_engine(None)

    def test_empty_string_database_url_raises_configuration_error(self):
        """Verifies whitespace or empty string URL raises DatabaseConfigurationError."""
        with pytest.raises(DatabaseConfigurationError, match="DATABASE_URL is not configured"):
            create_db_engine("   ")

    def test_invalid_url_scheme_raises_configuration_error(self):
        """Verifies invalid database URL scheme raises DatabaseConfigurationError without exposing secrets."""
        with pytest.raises(DatabaseConfigurationError, match="Failed to create database engine"):
            create_db_engine("http://invalid-scheme-for-db")

    def test_create_engine_postgresql_defaults(self):
        """Verifies PostgreSQL engine is created with pool_pre_ping and disabled echo."""
        url = "postgresql+psycopg://user:pass@localhost:5432/testdb"
        engine = create_db_engine(url)
        try:
            assert engine.pool._pre_ping is True
            assert engine.echo is False
        finally:
            engine.dispose()

    def test_create_engine_nullpool_compatibility(self):
        """Verifies create_db_engine handles NullPool (used by Alembic) without passing invalid pool_size."""
        url = "postgresql+psycopg://user:p%40ss%25word%3A123@localhost:5432/testdb"
        engine = create_db_engine(url, poolclass=pool.NullPool)
        try:
            assert isinstance(engine.pool, pool.NullPool)
            assert engine.pool._pre_ping is True
        finally:
            engine.dispose()


    def test_create_session_factory_safe_defaults(self, test_engine):
        """Verifies session factory enforces autocommit=False and expire_on_commit=False."""
        factory = create_session_factory(test_engine)
        session: Session = factory()
        try:
            assert session.autoflush is False
            assert session.expire_on_commit is False
        finally:
            session.close()

    def test_declarative_base_is_instantiable(self):
        """Verifies SQLAlchemy 2.0 Base class is defined and ready for future model inheritance."""
        assert issubclass(Base, object)
        assert hasattr(Base, "metadata")


class TestSessionLifecycle:
    """Tests verifying explicit transaction scoping, commit, rollback, and cleanup."""

    def test_db_session_scope_commits_on_success(self, test_session_factory):
        """Verifies db_session_scope commits transactional changes when no error occurs."""
        with db_session_scope(test_session_factory) as session:
            session.execute(text("INSERT INTO items (id, name) VALUES (1, 'Paper A')"))

        # In a separate session, verify data persisted
        with db_session_scope(test_session_factory) as session:
            result = session.execute(text("SELECT name FROM items WHERE id = 1")).scalar()
            assert result == "Paper A"

    def test_db_session_scope_rolls_back_on_exception(self, test_session_factory):
        """Verifies db_session_scope rolls back changes when an exception is raised."""
        class CustomTestError(Exception):
            pass

        with pytest.raises(CustomTestError):
            with db_session_scope(test_session_factory) as session:
                session.execute(text("INSERT INTO items (id, name) VALUES (2, 'Paper B')"))
                raise CustomTestError("Simulated failure during processing")

        # Verify row 2 was NOT committed to the database
        with db_session_scope(test_session_factory) as session:
            result = session.execute(text("SELECT name FROM items WHERE id = 2")).scalar()
            assert result is None

    def test_get_db_generator_yields_and_closes_session(self, test_session_factory, monkeypatch):
        """Verifies get_db yields an active session and guarantees its closure."""
        # Inject our test session factory
        monkeypatch.setattr("app.core.database.get_session_factory", lambda: test_session_factory)

        db_gen = get_db()
        session = next(db_gen)
        assert isinstance(session, Session)

        # Run query inside the session
        session.execute(text("INSERT INTO items (id, name) VALUES (3, 'Paper C')"))
        session.commit()

        # Wrap session.close to verify it gets called when generator finishes
        with patch.object(session, "close", wraps=session.close) as mock_close:
            with pytest.raises(StopIteration):
                next(db_gen)
            assert mock_close.called is True
            assert session.in_transaction() is False

    def test_get_db_generator_rolls_back_on_unhandled_exception(self, test_session_factory, monkeypatch):
        """Verifies get_db rolls back the active transaction if an exception is thrown through it."""
        monkeypatch.setattr("app.core.database.get_session_factory", lambda: test_session_factory)

        db_gen = get_db()
        session = next(db_gen)
        session.execute(text("INSERT INTO items (id, name) VALUES (4, 'Paper D')"))
        assert session.in_transaction() is True

        with patch.object(session, "close", wraps=session.close) as mock_close:
            # Simulate FastAPI route raising an unhandled exception
            with pytest.raises(RuntimeError, match="Route handler exploded"):
                db_gen.throw(RuntimeError("Route handler exploded"))
            assert mock_close.called is True
            assert session.in_transaction() is False

        # Verify row was rolled back and never committed to database
        with db_session_scope(test_session_factory) as verify_session:
            result = verify_session.execute(text("SELECT name FROM items WHERE id = 4")).scalar()
            assert result is None

    def test_db_session_scope_commit_failure_triggers_rollback_and_closes(self):
        """Verifies an exception during session.commit triggers rollback and closes session."""
        mock_session = MagicMock()
        mock_session.commit.side_effect = RuntimeError("Commit failed")
        mock_factory = lambda: mock_session

        with pytest.raises(RuntimeError, match="Commit failed"):
            with db_session_scope(mock_factory):
                pass

        assert mock_session.rollback.called is True
        assert mock_session.close.called is True


class TestApplicationImportSafety:
    """Verifies that importing the application never opens a database connection."""

    def test_importing_main_app_does_not_connect_to_db(self, monkeypatch):
        """Verifies FastAPI application can be instantiated even when DATABASE_URL is unset."""
        # Unset DATABASE_URL to verify no connection attempt occurs
        monkeypatch.setattr("app.core.database.settings.DATABASE_URL", None)
        from app.main import create_application
        app = create_application()
        assert app.title == "ResearchLens AI"

    def test_import_with_unreachable_postgres_url_does_not_connect(self, monkeypatch):
        """Verifies setting an unreachable PostgreSQL URL does not attempt a connection on app creation."""
        unreachable_url = "postgresql+psycopg://user:pass@non-existent-host.invalid:5432/researchlens"
        monkeypatch.setattr("app.core.database.settings.DATABASE_URL", unreachable_url)
        from app.main import create_application
        app = create_application()
        assert app.title == "ResearchLens AI"

    def test_reset_database_state_clears_singletons(self, tmp_path: Path):
        """Verifies global singletons can be cleanly initialized and reset."""
        db_url = f"sqlite:///{tmp_path}/reset_test.db"
        engine = get_engine(db_url)
        assert engine is not None
        factory = get_session_factory(db_url)
        assert factory is not None

        reset_database_state()
        from app.core import database
        assert database._engine is None
        assert database._session_factory is None
