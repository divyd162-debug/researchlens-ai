"""Main entrypoint for ResearchLens AI FastAPI application."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import logging
import os

from app.api import api_v1_router
from app.core.config import settings
from app.core.middleware import RequestSizeLimitMiddleware
from app.core.worker_lock import get_worker_lock
from app.services.index_recovery import verify_and_recover_vector_index

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown lifecycle."""
    # 1. Eagerly validate security, JWT, LLM, and document configuration on server startup
    settings.validate_jwt_settings()
    settings.validate_llm_settings()
    settings.validate_document_settings()

    # 2. Worker concurrency safeguard: acquire advisory file lock
    lock = get_worker_lock(settings.DATA_DIR)
    lock.acquire()

    try:
        from app.services.llm.factory import get_llm_provider
        active_llm = get_llm_provider()
        logger.info(
            "ResearchLens AI starting in single-worker mode (PID %d). "
            "Local FAISS index directory: %s. Active LLM provider: '%s', model: '%s'.",
            os.getpid(),
            settings.VECTOR_INDEX_DIR,
            active_llm.provider_name,
            active_llm.model_name,
        )

        # 3. Clean up stale temporary upload files older than 1 hour from interrupted uploads
        try:
            from app.services.document_service import cleanup_stale_temp_files
            cleanup_stale_temp_files(settings.UPLOAD_DIR)
        except Exception as cleanup_exc:
            logger.warning("Startup cleanup of stale temporary upload files failed: %s", cleanup_exc)

        # 4. FAISS index verification and deterministic startup recovery
        try:
            verify_and_recover_vector_index()
        except Exception as exc:
            logger.error("FAISS vector store startup recovery failed: %s", exc, exc_info=True)
            # Fail fast so server is never served with corrupt/inconsistent index
            raise

        yield
    finally:
        # Guarantee advisory worker lock release on either normal shutdown or startup failure
        try:
            lock.release()
        except Exception as release_exc:
            logger.warning("Error releasing advisory worker lock during shutdown: %s", release_exc)


def create_application() -> FastAPI:
    """Factory function to configure and return the FastAPI application.

    Eagerly validates security and LLM configuration at factory creation time
    to fail fast if running with missing or insecure credentials.
    """
    settings.validate_jwt_settings()
    settings.validate_llm_settings()
    settings.validate_document_settings()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="Grounded AI Research Assistant API with verifiable citations",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure request size limiting middleware (Phase 5, Milestone 2)
    app.add_middleware(RequestSizeLimitMiddleware)

    # Configure CORS for frontend access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register versioned API routes
    app.include_router(api_v1_router, prefix="/api/v1")

    @app.get("/", tags=["General"])
    def root():
        """Root endpoint returning basic application information."""
        return {
            "name": settings.PROJECT_NAME,
            "status": "online",
            "version": "0.1.0",
            "docs": "/docs",
        }

    @app.get("/health", tags=["General"])
    def health_check():
        """Liveness check endpoint returning basic process health without external dependencies."""
        return {
            "status": "healthy",
            "environment": settings.ENVIRONMENT,
        }

    @app.get("/health/ready", tags=["General"])
    def readiness_check():
        """Readiness check endpoint verifying database, vector store, and upload directory."""
        from fastapi.responses import JSONResponse
        from pathlib import Path

        failures = []

        # 1. Database readiness check (lightweight SELECT 1 with bounded connection)
        try:
            from sqlalchemy import text
            from app.core.database import get_db, get_session_factory
            if get_db in app.dependency_overrides:
                override = app.dependency_overrides[get_db]
                gen = override()
                session = next(gen)
                try:
                    session.execute(text("SELECT 1"))
                finally:
                    try:
                        next(gen)
                    except StopIteration:
                        pass
            else:
                factory = get_session_factory()
                with factory() as session:
                    session.execute(text("SELECT 1"))
        except Exception as exc:
            logger.warning("Readiness check: database dependency failure (%s)", type(exc).__name__)
            failures.append("database")

        # 2. Vector store readiness check (in-memory readiness flag without loading embedding model)
        try:
            from app.services.vector_store import get_vector_store
            vs = get_vector_store()
            if not vs.is_ready:
                failures.append("vector_store")
        except Exception as exc:
            logger.warning("Readiness check: vector store dependency failure (%s)", type(exc).__name__)
            failures.append("vector_store")

        # 3. Upload directory readiness check (filesystem existence and permissions without test files)
        try:
            upload_path = Path(settings.UPLOAD_DIR)
            if not (upload_path.exists() and upload_path.is_dir() and os.access(upload_path, os.R_OK | os.W_OK)):
                failures.append("upload_directory")
        except Exception as exc:
            logger.warning("Readiness check: upload directory dependency failure (%s)", type(exc).__name__)
            failures.append("upload_directory")

        if failures:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "failures": failures},
            )

        return {"status": "ready"}


    return app



app = create_application()
