"""Core application configuration and settings."""
import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv


def load_app_env(override: bool = False) -> None:
    """Loads environment variables with robust path resolution across directories.

    Search priority (highest to lowest precedence):
    1. Explicit custom ENV_FILE if provided in the environment.
    2. backend/.env (canonical backend configuration location).
    3. .env at project root workspace.
    4. Fallback search in current working directory (if different from backend and root).
    """
    backend_dir = Path(__file__).resolve().parent.parent.parent  # backend/
    root_dir = backend_dir.parent

    # 1. Base project root .env
    root_env = root_dir / ".env"
    if root_env.is_file():
        load_dotenv(dotenv_path=root_env, override=override)

    # 2. Canonical backend/.env (overrides root .env)
    backend_env = backend_dir / ".env"
    if backend_env.is_file():
        load_dotenv(dotenv_path=backend_env, override=override)

    # 3. Fallback for cwd if outside backend/ and root/
    cwd_env = Path.cwd() / ".env"
    if cwd_env.is_file() and cwd_env != backend_env and cwd_env != root_env:
        load_dotenv(dotenv_path=cwd_env, override=override)

    # 4. Explicit custom ENV_FILE (highest priority file)
    env_file_custom = os.getenv("ENV_FILE")
    if env_file_custom and Path(env_file_custom).is_file():
        load_dotenv(dotenv_path=Path(env_file_custom), override=override)


load_app_env(override=False)

# Standard development secret (at least 32 characters, safe only for local development/testing)
DEV_DEFAULT_JWT_SECRET = "dev-insecure-jwt-secret-do-not-use-in-production-min-32-chars"
DEFAULT_DATA_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "data")
DEFAULT_UPLOAD_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "uploads")


class Settings:
    """Application settings loaded from environment variables."""
    PROJECT_NAME: str = os.getenv("APP_NAME", "ResearchLens AI")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # Parse CORS origins from comma-separated string
    CORS_ORIGINS: List[str] = [
        origin.strip() 
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if origin.strip()
    ]

    # Relational Database connection URL (Phase 3)
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")

    # LLM Provider Configuration (OmniRoute OpenAI-Compatible Gateway)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mock")
    OMNIROUTE_BASE_URL: str = os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
    OMNIROUTE_API_KEY: Optional[str] = os.getenv("OMNIROUTE_API_KEY")
    OMNIROUTE_MODEL: str = os.getenv("OMNIROUTE_MODEL", "gemini-fallback-combo")
    OMNIROUTE_TIMEOUT_SECONDS: float = float(os.getenv("OMNIROUTE_TIMEOUT_SECONDS", "60.0"))
    OMNIROUTE_MAX_RETRIES: int = int(os.getenv("OMNIROUTE_MAX_RETRIES", "1"))

    # JWT Authentication Configuration (Phase 5, Milestone 1A)
    JWT_SECRET_KEY: Optional[str] = os.getenv("JWT_SECRET_KEY")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    # Storage & Upload Configuration (Phase 5, Milestone 2)
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", DEFAULT_UPLOAD_DIR)

    # Maximum allowed page count for uploaded PDF documents (default: 100 pages).
    # Mitigates resource exhaustion and decompression DoS from oversized documents.
    MAX_DOCUMENT_PAGES: int = int(os.getenv("MAX_DOCUMENT_PAGES", "100"))

    # Maximum allowed size for the PDF file payload itself (default: 25 MB = 26,214,400 bytes).
    # Enforced strictly by DocumentService during file streaming and database persistence.
    MAX_UPLOAD_SIZE_BYTES: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024)))

    # Bounded buffer for HTTP multipart/form-data encoding overhead (boundaries, headers, form fields like title).
    MULTIPART_OVERHEAD_BYTES: int = int(os.getenv("MULTIPART_OVERHEAD_BYTES", str(256 * 1024)))  # 256 KB

    # Maximum allowed size for the entire HTTP request body at the ASGI boundary.
    # Enforced by RequestSizeLimitMiddleware to prevent unbounded spooling while accommodating multipart overhead.
    MAX_REQUEST_BODY_SIZE_BYTES: int = int(
        os.getenv("MAX_REQUEST_BODY_SIZE_BYTES", str(MAX_UPLOAD_SIZE_BYTES + MULTIPART_OVERHEAD_BYTES))
    )

    # Vector Store & Embedding Configuration (Phase 7, Milestone 1)
    DATA_DIR: str = os.getenv("DATA_DIR", DEFAULT_DATA_DIR)
    VECTOR_INDEX_DIR: str = os.getenv("VECTOR_INDEX_DIR", DEFAULT_DATA_DIR)
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

    # Ingestion timeout threshold for stale in-flight documents (default: 300 seconds / 5 minutes).
    # In-flight documents (EXTRACTING, CHUNKING, EMBEDDING) older than this threshold are considered
    # stranded due to worker process crashes and are eligible for recovery/retry.
    STALE_INGESTION_TIMEOUT_SECONDS: int = int(os.getenv("STALE_INGESTION_TIMEOUT_SECONDS", "300"))

    # Multi-turn conversational history configuration (Stage 2)
    CHAT_MAX_HISTORY_TURNS: int = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "10"))
    CHAT_MAX_HISTORY_CHARS: int = int(os.getenv("CHAT_MAX_HISTORY_CHARS", "4000"))

    def __init__(self) -> None:
        self.reload(load_env=False)

    def reload(self, load_env: bool = False, override: bool = False) -> None:
        """Reloads settings from environment variables."""
        if load_env:
            load_app_env(override=override)

        self.PROJECT_NAME = os.getenv("APP_NAME", "ResearchLens AI")
        self.ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", "8000"))
        self.CORS_ORIGINS = [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
            if origin.strip()
        ]
        self.DATABASE_URL = os.getenv("DATABASE_URL")
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
        self.OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
        self.OMNIROUTE_API_KEY = os.getenv("OMNIROUTE_API_KEY")
        self.OMNIROUTE_MODEL = os.getenv("OMNIROUTE_MODEL", "gemini-fallback-combo")
        self.OMNIROUTE_TIMEOUT_SECONDS = float(os.getenv("OMNIROUTE_TIMEOUT_SECONDS", "60.0"))
        self.OMNIROUTE_MAX_RETRIES = int(os.getenv("OMNIROUTE_MAX_RETRIES", "1"))
        self.JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
        self.JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
        self.UPLOAD_DIR = os.getenv("UPLOAD_DIR", DEFAULT_UPLOAD_DIR)
        self.MAX_DOCUMENT_PAGES = int(os.getenv("MAX_DOCUMENT_PAGES", "100"))
        self.MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024)))
        self.MULTIPART_OVERHEAD_BYTES = int(os.getenv("MULTIPART_OVERHEAD_BYTES", str(256 * 1024)))
        self.MAX_REQUEST_BODY_SIZE_BYTES = int(
            os.getenv("MAX_REQUEST_BODY_SIZE_BYTES", str(self.MAX_UPLOAD_SIZE_BYTES + self.MULTIPART_OVERHEAD_BYTES))
        )
        self.DATA_DIR = os.getenv("DATA_DIR", DEFAULT_DATA_DIR)
        self.VECTOR_INDEX_DIR = os.getenv("VECTOR_INDEX_DIR", DEFAULT_DATA_DIR)
        self.EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.STALE_INGESTION_TIMEOUT_SECONDS = int(os.getenv("STALE_INGESTION_TIMEOUT_SECONDS", "300"))
        self.CHAT_MAX_HISTORY_TURNS = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "10"))
        self.CHAT_MAX_HISTORY_CHARS = int(os.getenv("CHAT_MAX_HISTORY_CHARS", "4000"))

    def get_jwt_secret_key(self) -> str:
        """Returns the effective JWT secret key with strict environment validation.

        Rules:
        - In production (ENVIRONMENT == 'production'):
          * Must not be None, empty, or whitespace.
          * Must be at least 32 characters long (256-bit entropy recommendation).
          * Must NOT match the development fallback key.
          * Fails closed (raises SecurityConfigurationError) if unsafe.
        - In development/testing (ENVIRONMENT in ('development', 'test', 'testing')):
          * Returns JWT_SECRET_KEY if configured.
          * Otherwise safely falls back to DEV_DEFAULT_JWT_SECRET.
        - In any unrecognized environment:
          * Treated as production (strict fail-closed).
        """
        from app.core.security import SecurityConfigurationError

        env = (self.ENVIRONMENT or "development").strip().lower()
        secret = self.JWT_SECRET_KEY

        if env in ("development", "test", "testing"):
            if secret and secret.strip():
                return secret.strip()
            return DEV_DEFAULT_JWT_SECRET

        # Production and any unrecognized environment (strict fail-closed)
        if not secret or not secret.strip():
            raise SecurityConfigurationError(
                "JWT_SECRET_KEY is not configured. A strong secret of at least 32 characters is required in production."
            )

        trimmed_secret = secret.strip()
        if trimmed_secret == DEV_DEFAULT_JWT_SECRET:
            raise SecurityConfigurationError(
                "JWT_SECRET_KEY cannot use the default development secret in production."
            )

        if len(trimmed_secret) < 32:
            raise SecurityConfigurationError(
                f"JWT_SECRET_KEY must be at least 32 characters long in production (got {len(trimmed_secret)})."
            )

        return trimmed_secret

    def validate_jwt_settings(self) -> None:
        """Validates all JWT settings for the current environment.

        Raises:
            SecurityConfigurationError: If any setting is invalid.
        """
        from app.core.security import SecurityConfigurationError

        if self.ACCESS_TOKEN_EXPIRE_MINUTES <= 0:
            raise SecurityConfigurationError(
                f"ACCESS_TOKEN_EXPIRE_MINUTES must be positive, got {self.ACCESS_TOKEN_EXPIRE_MINUTES}."
            )
        if self.ACCESS_TOKEN_EXPIRE_MINUTES > 1440:
            raise SecurityConfigurationError(
                f"ACCESS_TOKEN_EXPIRE_MINUTES cannot exceed 1440 minutes (24 hours), got {self.ACCESS_TOKEN_EXPIRE_MINUTES}."
            )

        supported_algorithms = ["HS256", "HS384", "HS512"]
        if not self.JWT_ALGORITHM or self.JWT_ALGORITHM.upper() not in supported_algorithms:
            raise SecurityConfigurationError(
                f"Unsupported JWT_ALGORITHM '{self.JWT_ALGORITHM}'. Must be one of {supported_algorithms}."
            )

        # Validates secret key requirements for active environment
        self.get_jwt_secret_key()

    def validate_llm_settings(self) -> None:
        """Validates LLM provider settings.

        Raises:
            LLMConfigurationError: If LLM_PROVIDER is invalid or required credentials are missing.
        """
        from app.services.llm.exceptions import LLMConfigurationError

        provider = (self.LLM_PROVIDER or "").strip().lower()
        if not provider:
            raise LLMConfigurationError("LLM_PROVIDER must not be empty.")

        if provider == "mock":
            return

        if provider == "omniroute":
            if not self.OMNIROUTE_BASE_URL or not str(self.OMNIROUTE_BASE_URL).strip():
                raise LLMConfigurationError("OMNIROUTE_BASE_URL cannot be empty when LLM_PROVIDER is 'omniroute'.")
            if not str(self.OMNIROUTE_BASE_URL).strip().startswith(("http://", "https://")):
                raise LLMConfigurationError(
                    f"OMNIROUTE_BASE_URL must be a valid HTTP or HTTPS URL, got '{self.OMNIROUTE_BASE_URL}'."
                )
            if not self.OMNIROUTE_MODEL or not str(self.OMNIROUTE_MODEL).strip():
                raise LLMConfigurationError("OMNIROUTE_MODEL cannot be empty when LLM_PROVIDER is 'omniroute'.")
            if self.OMNIROUTE_TIMEOUT_SECONDS <= 0:
                raise LLMConfigurationError(
                    f"OMNIROUTE_TIMEOUT_SECONDS must be positive, got {self.OMNIROUTE_TIMEOUT_SECONDS}."
                )
            if self.OMNIROUTE_MAX_RETRIES < 0:
                raise LLMConfigurationError(
                    f"OMNIROUTE_MAX_RETRIES must be non-negative, got {self.OMNIROUTE_MAX_RETRIES}."
                )
            return

        raise LLMConfigurationError(
            f"Unsupported LLM_PROVIDER '{provider}'. Supported providers: ['mock', 'omniroute']."
        )

    def validate_document_settings(self) -> None:
        """Validates document upload configuration settings.

        Raises:
            ValueError: If MAX_DOCUMENT_PAGES or MAX_UPLOAD_SIZE_BYTES is not positive.
        """
        if self.MAX_DOCUMENT_PAGES <= 0:
            raise ValueError(
                f"MAX_DOCUMENT_PAGES must be a positive integer, got {self.MAX_DOCUMENT_PAGES}."
            )
        if self.MAX_UPLOAD_SIZE_BYTES <= 0:
            raise ValueError(
                f"MAX_UPLOAD_SIZE_BYTES must be a positive integer, got {self.MAX_UPLOAD_SIZE_BYTES}."
            )


settings = Settings()
