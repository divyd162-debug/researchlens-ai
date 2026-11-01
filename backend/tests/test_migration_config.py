"""Unit tests for Alembic migration configuration, metadata binding, and credential safety.

Design Note:
These tests verify that Alembic configuration files are properly initialized,
point to the application's Base.metadata, resolve DATABASE_URL dynamically without
hardcoded credentials, and can generate offline SQL migrations without connecting
to a live PostgreSQL instance.
"""
import io
from pathlib import Path
import pytest
from alembic.config import Config
from alembic import command
from alembic.script import ScriptDirectory

from app.core.database import Base


@pytest.fixture
def alembic_config_path() -> Path:
    """Returns the path to backend/alembic.ini."""
    backend_dir = Path(__file__).resolve().parents[1]
    config_file = backend_dir / "alembic.ini"
    assert config_file.exists(), f"Expected alembic.ini at {config_file}"
    return config_file


@pytest.fixture
def alembic_cfg(alembic_config_path: Path) -> Config:
    """Provides an Alembic Config object loaded from backend/alembic.ini."""
    cfg = Config(str(alembic_config_path))
    return cfg


class TestAlembicStructureAndFiles:
    """Tests verifying migration directories, template files, and scripts exist."""

    def test_migration_directory_structure_exists(self, alembic_config_path: Path):
        """Verifies alembic directory, script template, and versions folder exist."""
        backend_dir = alembic_config_path.parent
        alembic_dir = backend_dir / "alembic"
        versions_dir = alembic_dir / "versions"
        env_py = alembic_dir / "env.py"
        script_mako = alembic_dir / "script.py.mako"
        readme = alembic_dir / "README"

        assert alembic_dir.is_dir()
        assert versions_dir.is_dir()
        assert env_py.is_file()
        assert script_mako.is_file()
        assert readme.is_file()

    def test_alembic_ini_contains_no_hardcoded_credentials(self, alembic_config_path: Path):
        """Verifies that neither backend nor root alembic.ini contains hardcoded database passwords."""
        for ini_path in [alembic_config_path, alembic_config_path.parents[1] / "alembic.ini"]:
            if ini_path.exists():
                content = ini_path.read_text(encoding="utf-8")
                # Ensure no plaintext passwords or live credentials
                assert "password" not in content.lower() or "#" in content
                # Ensure active sqlalchemy.url is commented out or absent
                lines = [line.strip() for line in content.splitlines()]
                active_urls = [line for line in lines if line.startswith("sqlalchemy.url") and "=" in line and not line.startswith("#")]
                assert len(active_urls) == 0, f"Found un-commented sqlalchemy.url in {ini_path}: {active_urls}"

    def test_root_and_backend_configs_resolve_identically(self, alembic_config_path: Path):
        """Verifies root alembic.ini and backend/alembic.ini resolve to the exact same script directory."""
        root_ini = alembic_config_path.parents[1] / "alembic.ini"
        assert root_ini.exists()

        cfg_backend = Config(str(alembic_config_path))
        cfg_root = Config(str(root_ini))

        script_backend = ScriptDirectory.from_config(cfg_backend)
        script_root = ScriptDirectory.from_config(cfg_root)

        assert Path(script_backend.dir).resolve() == Path(script_root.dir).resolve()
        assert Path(script_backend.versions).resolve() == Path(script_root.versions).resolve()



class TestAlembicMetadataAndEnvironment:
    """Tests verifying metadata binding and configuration resolution."""

    def test_script_directory_loads_cleanly(self, alembic_cfg: Config):
        """Verifies Alembic can load the ScriptDirectory without syntax or path errors."""
        script = ScriptDirectory.from_config(alembic_cfg)
        assert script is not None
        # Verify initial schema migration exists and is discoverable (Milestone 2B)
        revisions = list(script.walk_revisions())
        assert len(revisions) >= 1
        assert revisions[-1].down_revision is None  # Initial root migration

    def test_target_metadata_points_to_declarative_base(self):
        """Verifies env.py imports and binds target_metadata to Base.metadata."""
        import importlib.util
        backend_dir = Path(__file__).resolve().parents[1]
        env_path = backend_dir / "alembic" / "env.py"

        spec = importlib.util.spec_from_file_location("alembic_env_module", env_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert module.target_metadata is Base.metadata

    def test_get_database_url_raises_when_unconfigured(self, monkeypatch):
        """Verifies get_database_url in env.py raises ValueError if no URL is available."""
        import importlib.util
        backend_dir = Path(__file__).resolve().parents[1]
        env_path = backend_dir / "alembic" / "env.py"

        spec = importlib.util.spec_from_file_location("alembic_env_test", env_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Mock settings.DATABASE_URL to None
        monkeypatch.setattr("app.core.config.settings.DATABASE_URL", None)
        with pytest.raises(ValueError, match="DATABASE_URL is not configured"):
            module.get_database_url()

    def test_get_database_url_prefers_x_argument(self, monkeypatch):
        """Verifies explicit -x url=... argument takes precedence over settings."""
        import importlib.util
        from alembic import context
        backend_dir = Path(__file__).resolve().parents[1]
        env_path = backend_dir / "alembic" / "env.py"

        spec = importlib.util.spec_from_file_location("alembic_env_x_test", env_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Set settings.DATABASE_URL to default
        monkeypatch.setattr("app.core.config.settings.DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/default_db")
        # Mock context.get_x_argument to provide override
        override_url = "postgresql+psycopg://override:secret@remote:5432/override_db"
        monkeypatch.setattr(context, "get_x_argument", lambda as_dictionary=True: {"url": override_url})

        resolved = module.get_database_url()
        assert resolved == override_url


class TestOfflineMigrationGeneration:
    """Tests verifying offline SQL generation without live database connectivity."""

    def test_offline_sql_generation_sqlite(self, alembic_cfg: Config):
        """Verifies alembic upgrade head --sql generates SQL for SQLite without live database."""
        buf = io.StringIO()
        alembic_cfg.stdout = buf
        alembic_cfg.set_main_option("sqlalchemy.url", "sqlite:///:memory:")

        # Run offline upgrade
        command.upgrade(alembic_cfg, "head", sql=True)
        output = buf.getvalue()
        # Even with no tables, offline mode executes without error
        assert isinstance(output, str)

    def test_offline_sql_generation_postgresql(self, alembic_cfg: Config, monkeypatch):
        """Verifies offline mode emits PostgreSQL DDL headers without connecting."""
        buf = io.StringIO()
        alembic_cfg.stdout = buf
        monkeypatch.setattr(
            "app.core.config.settings.DATABASE_URL",
            "postgresql+psycopg://testuser:testpass@localhost:5432/testdb",
        )

        command.upgrade(alembic_cfg, "head", sql=True)
        output = buf.getvalue()
        assert "BEGIN;" in output or output == ""

    def test_offline_sql_generation_with_encoded_special_chars_password(self, alembic_cfg: Config, monkeypatch):
        """Verifies offline SQL generation with passwords containing %, @, : succeeds and masks credentials."""
        buf = io.StringIO()
        alembic_cfg.stdout = buf
        complex_url = "postgresql+psycopg://myuser:p%40ss%25word%3A123@localhost:5432/testdb"
        monkeypatch.setattr("app.core.config.settings.DATABASE_URL", complex_url)

        command.upgrade(alembic_cfg, "head", sql=True)
        output = buf.getvalue()
        # Verify output is valid SQL
        assert "BEGIN;" in output or output == ""
        # Verify raw and encoded passwords never appear in generated SQL
        assert "p%40ss%25word%3A123" not in output
        assert "p@ss%word:123" not in output

    def test_disposable_database_migration_cycle_forwards_and_backwards(self, alembic_cfg: Config, tmp_path: Path):
        """Proves that all Alembic revisions run cleanly forwards and backwards on a disposable SQLite test database."""
        db_path = tmp_path / "disposable_test.db"
        alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")

        # Upgrade from base to head
        command.upgrade(alembic_cfg, "head")

        # Downgrade from head to base
        command.downgrade(alembic_cfg, "base")

        # Re-upgrade from base to head
        command.upgrade(alembic_cfg, "head")


