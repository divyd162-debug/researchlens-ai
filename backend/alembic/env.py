"""Alembic environment configuration for ResearchLens AI.

Architecture & Design Principles:
1. Dynamic Configuration: Reads DATABASE_URL from app.core.config (settings)
   rather than hardcoding credentials in alembic.ini.
2. Metadata Integration: Uses Base.metadata from app.core.database for autogenerate.
3. Safe Logging & Error Handling: Database URLs with passwords are masked and
   never logged or emitted in error outputs.
4. Lazy / Conditional Connection: Importing this module does NOT connect to the
   database. Connections are only established in run_migrations_online().
"""
from logging.config import fileConfig
import os
from pathlib import Path
import sys

from sqlalchemy import pool
from alembic import context

# Ensure backend directory is in sys.path so 'app' can be imported cleanly
# regardless of working directory
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.core.database import Base, create_db_engine, mask_database_url
import app.models  # Ensures all ORM models are registered with Base.metadata

# Alembic Config object (present when invoked by Alembic runner)
try:
    config = context.config
except (AttributeError, NameError):
    config = None

# Interpret the config file for Python logging
if config is not None and config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Point target_metadata to our SQLAlchemy DeclarativeBase metadata
# When models are defined in Milestone 2B, their tables will be registered here
target_metadata = Base.metadata


def get_database_url(cfg=None) -> str:
    """Retrieves the database connection string from arguments, settings, or config.

    Raises:
        ValueError: If no database URL is configured.
    """
    active_cfg = cfg or config
    x_args = {}
    try:
        x_args = context.get_x_argument(as_dictionary=True)
    except (AttributeError, NameError):
        pass

    main_url = active_cfg.get_main_option("sqlalchemy.url") if active_cfg else None
    if main_url and not str(main_url).strip():
        main_url = None

    url = (
        x_args.get("url")
        or main_url
        or settings.DATABASE_URL
    )

    if not url or not str(url).strip():
        raise ValueError(
            "DATABASE_URL is not configured. Set DATABASE_URL in your .env file or "
            "provide it via command-line: alembic -x url=<DATABASE_URL>"
        )

    return str(url).strip()


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Generates SQL scripts without establishing a live database connection.
    Useful for code review, dry runs, and air-gapped deployments.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Establishes a connection to the database and applies migrations within
    an explicit transaction.
    """
    url = get_database_url()

    # Reuse application's create_db_engine to benefit from pool_pre_ping,
    # safe defaults, and masked error messages.
    connectable = create_db_engine(database_url=url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


is_offline = None
try:
    is_offline = context.is_offline_mode()
except (AttributeError, NameError):
    # env.py was imported directly outside an active Alembic migration context
    pass

if is_offline is True:
    run_migrations_offline()
elif is_offline is False:
    run_migrations_online()

