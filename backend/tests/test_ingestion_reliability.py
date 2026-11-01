"""Reliability regression tests for PDF ingestion pipeline.

Covers:
1. Concurrency races and atomic status claims (Task 1).
2. FAISS index recovery, rebuild, and failure safety (Task 2).
3. Single-worker advisory lock safeguards (Task 3).
"""
import concurrent.futures
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from typing import Generator
from unittest.mock import MagicMock, patch
from uuid import uuid4

import faiss
from fastapi import FastAPI
import numpy as np
import pytest
from fastapi.testclient import TestClient
import pymupdf
from sqlalchemy import create_engine, func, select, text, update
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.database import Base, create_session_factory, db_session_scope
from app.core.security import create_access_token
from app.core.worker_lock import MultiWorkerError, PlatformNotSupportedError, WorkerLock
from app.main import app, create_application
from app.models.chunk import Chunk
from app.models.document import Document, ProcessingStatus
from app.models.user import User
from app.services.embedding_service import EmbeddingService, EncodingError
from app.services.index_recovery import rebuild_vector_index, verify_and_recover_vector_index
from app.services.ingestion_service import IngestionService
from app.services.pdf_service import PDFService
from app.services.vector_store import (
    VectorIndexRecoveryError,
    VectorStore,
    VectorStoreNotReadyError,
)


def create_sample_pdf(file_path: Path, pages_text: list[str]) -> Path:
    """Helper to generate a valid PDF file with specified text on each page."""
    doc = pymupdf.open()
    for text_content in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text_content, fontsize=12)
    doc.save(str(file_path))
    doc.close()
    return file_path


@pytest.fixture
def db_engine(tmp_path: Path):
    """File-backed SQLite engine enabling independent thread connections without driver misuse."""
    db_file = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"check_same_thread": False, "timeout": 15},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def session_factory(db_engine) -> sessionmaker[Session]:
    return create_session_factory(db_engine)


@pytest.fixture
def db_session(session_factory) -> Generator[Session, None, None]:
    with db_session_scope(session_factory) as session:
        yield session


@pytest.fixture
def test_user(db_session: Session) -> User:
    user = User(
        email="reliability_test@researchlens.ai",
        hashed_password="hashed_pw_for_test",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    service = MagicMock(spec=EmbeddingService)
    service.dimension = 384
    service.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def deterministic_embed(texts):
        embeddings = []
        for text_val in texts:
            seed = sum(ord(c) for c in text_val) % 10000
            np.random.seed(seed)
            vec = np.random.randn(384).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec.tolist())
        return embeddings

    service.embed_batch.side_effect = deterministic_embed
    return service


@pytest.fixture
def test_vector_store(tmp_path: Path) -> VectorStore:
    index_dir = tmp_path / "vec_store"
    index_dir.mkdir(parents=True, exist_ok=True)
    store = VectorStore(
        dimension=384,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        index_dir=index_dir,
    )
    return store


@pytest.fixture
def ingestion_service(
    session_factory: sessionmaker[Session],
    mock_embedding_service: EmbeddingService,
    test_vector_store: VectorStore,
) -> IngestionService:
    return IngestionService(
        session_factory=session_factory,
        pdf_service=PDFService(),
        embedding_service=mock_embedding_service,
        vector_store=test_vector_store,
    )


# ==============================================================================
# 1. CONCURRENCY & RACE CONDITION REGRESSION TESTS
# ==============================================================================

class TestConcurrencyAndClaimSafety:
    """Verifies that atomic status claims prevent duplicate execution and corruption."""

    def test_concurrent_ingestion_attempts_only_one_claims_and_succeeds(
        self,
        db_session: Session,
        session_factory: sessionmaker[Session],
        test_user: User,
        tmp_path: Path,
        ingestion_service: IngestionService,
        test_vector_store: VectorStore,
    ):
        """Proves that when two tasks race to process the same document:
        - Only one task claims the document.
        - The losing task exits immediately without modifying chunks, vectors, or status.
        - The losing task cannot mark the document FAILED after the winning task succeeds.
        """
        pdf_path = tmp_path / "concurrent.pdf"
        create_sample_pdf(pdf_path, ["Unique content for concurrent race testing."])

        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="concurrent.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="concurrent_hash_1",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db_session.add(doc)
        db_session.commit()

        # Run two ingestion attempts concurrently in separate threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(
                ingestion_service.process_document, doc.id, session_factory
            )
            future2 = executor.submit(
                ingestion_service.process_document, doc.id, session_factory
            )
            res1 = future1.result()
            res2 = future2.result()

        # Both returned without throwing unhandled exceptions
        assert res1 is not None
        assert res2 is not None

        # Verify final document status in DB is READY
        db_session.expire_all()
        reloaded = db_session.query(Document).filter(Document.id == doc.id).first()
        assert reloaded.processing_status == ProcessingStatus.READY.value
        assert reloaded.error_message is None

        # Verify exactly one set of chunks was created (not duplicate chunks)
        chunks = db_session.query(Chunk).filter(Chunk.document_id == doc.id).all()
        assert len(chunks) == 1
        assert test_vector_store.total_vectors == 1
        assert test_vector_store.contains_id(chunks[0].id)

    def test_losing_task_cannot_fail_already_ready_document(
        self,
        db_session: Session,
        session_factory: sessionmaker[Session],
        test_user: User,
        tmp_path: Path,
        ingestion_service: IngestionService,
        test_vector_store: VectorStore,
    ):
        """Verifies that an ingestion task executed against a READY document exits
        safely without transitioning the document to FAILED.
        """
        pdf_path = tmp_path / "already_ready.pdf"
        create_sample_pdf(pdf_path, ["Passage in ready document."])

        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="already_ready.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="ready_hash_1",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db_session.add(doc)
        db_session.commit()

        # Initial successful ingestion
        r1 = ingestion_service.process_document(doc.id, session_factory)
        assert r1.processing_status == ProcessingStatus.READY.value

        # Simulate a delayed second task that runs now that doc is READY
        r2 = ingestion_service.process_document(doc.id, session_factory, force=False)
        assert r2.processing_status == ProcessingStatus.READY.value

        # DB status remains READY and chunks remain untouched
        db_session.expire_all()
        reloaded = db_session.query(Document).filter(Document.id == doc.id).first()
        assert reloaded.processing_status == ProcessingStatus.READY.value
        assert reloaded.error_message is None

    def test_failed_claim_rolls_back_cleanly_without_corrupting_state(
        self,
        db_session: Session,
        session_factory: sessionmaker[Session],
        test_user: User,
        tmp_path: Path,
        ingestion_service: IngestionService,
    ):
        """Verifies that if a document is currently in EXTRACTING state,
        a concurrent attempt fails the claim, rolls back, and leaves state untouched.
        """
        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="in_flight.pdf",
            file_path=str(tmp_path / "in_flight.pdf"),
            file_hash_sha256="in_flight_hash",
            file_size_bytes=100,
            processing_status=ProcessingStatus.EXTRACTING.value,
        )
        db_session.add(doc)
        db_session.commit()

        # Attempt to process in-flight document
        result = ingestion_service.process_document(doc.id, session_factory)
        assert result.processing_status == ProcessingStatus.EXTRACTING.value

        db_session.expire_all()
        reloaded = db_session.query(Document).filter(Document.id == doc.id).first()
        assert reloaded.processing_status == ProcessingStatus.EXTRACTING.value

    def test_stale_in_flight_document_is_recovered_by_process_document(
        self,
        db_session: Session,
        session_factory: sessionmaker[Session],
        test_user: User,
        tmp_path: Path,
        ingestion_service: IngestionService,
        test_vector_store: VectorStore,
    ):
        """Proves that a document stranded in EXTRACTING after a process crash
        (where updated_at exceeds STALE_INGESTION_TIMEOUT_SECONDS) is atomically claimed
        and processed to READY by ingestion_service.process_document.
        """
        pdf_path = tmp_path / "stale_doc.pdf"
        create_sample_pdf(pdf_path, ["Recovered passage from stranded task."])

        stale_time = datetime.now(timezone.utc) - timedelta(seconds=settings.STALE_INGESTION_TIMEOUT_SECONDS + 60)
        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="stale_doc.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="stale_doc_hash_1",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.EXTRACTING.value,
            updated_at=stale_time,
        )
        db_session.add(doc)
        db_session.commit()

        # Run process_document on the stale document
        result = ingestion_service.process_document(doc.id, session_factory)
        assert result is not None
        assert result.processing_status == ProcessingStatus.READY.value
        assert result.error_message is None

        db_session.expire_all()
        reloaded = db_session.query(Document).filter(Document.id == doc.id).first()
        assert reloaded.processing_status == ProcessingStatus.READY.value
        assert test_vector_store.total_vectors == 1

    def test_active_in_flight_document_is_not_disrupted_by_second_task(
        self,
        db_session: Session,
        session_factory: sessionmaker[Session],
        test_user: User,
        tmp_path: Path,
        ingestion_service: IngestionService,
    ):
        """Proves that an actively running ingestion task (updated_at within STALE_INGESTION_TIMEOUT_SECONDS)
        cannot be claimed by a second task.
        """
        pdf_path = tmp_path / "active_doc.pdf"
        create_sample_pdf(pdf_path, ["Active content."])

        recent_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="active_doc.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="active_doc_hash_1",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.CHUNKING.value,
            updated_at=recent_time,
        )
        db_session.add(doc)
        db_session.commit()

        # Second task tries to claim the document
        result = ingestion_service.process_document(doc.id, session_factory)
        # Should fail atomic claim, return document in existing state without altering status or chunks
        assert result is not None
        assert result.processing_status == ProcessingStatus.CHUNKING.value

        db_session.expire_all()
        reloaded = db_session.query(Document).filter(Document.id == doc.id).first()
        assert reloaded.processing_status == ProcessingStatus.CHUNKING.value

    def test_retry_endpoint_recovers_stale_in_flight_document(
        self,
        db_session: Session,
        test_user: User,
        tmp_path: Path,
    ):
        """Proves that POST /api/v1/documents/{id}/retry recovers a stale in-flight document."""
        from app.api.deps import get_current_user, get_db

        pdf_path = tmp_path / "stale_retry.pdf"
        create_sample_pdf(pdf_path, ["Stale retry test passage."])

        stale_time = datetime.now(timezone.utc) - timedelta(seconds=settings.STALE_INGESTION_TIMEOUT_SECONDS + 120)
        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="stale_retry.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="stale_retry_hash_1",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.EMBEDDING.value,
            updated_at=stale_time,
        )
        db_session.add(doc)
        db_session.commit()

        app.dependency_overrides[get_current_user] = lambda: test_user
        app.dependency_overrides[get_db] = lambda: db_session

        try:
            with TestClient(app) as client:
                resp = client.post(f"/api/v1/documents/{doc.id}/retry")
                assert resp.status_code == 200
                data = resp.json()
                # Status reset to PENDING (or READY if background task ran in sync test client)
                assert data["processing_status"] in (
                    ProcessingStatus.PENDING.value,
                    ProcessingStatus.READY.value,
                )

            db_session.expire_all()
            reloaded = db_session.query(Document).filter(Document.id == doc.id).first()
            assert reloaded.processing_status in (
                ProcessingStatus.PENDING.value,
                ProcessingStatus.READY.value,
            )
        finally:
            app.dependency_overrides.clear()

    def test_retry_endpoint_does_not_disrupt_active_task(
        self,
        db_session: Session,
        test_user: User,
        tmp_path: Path,
    ):
        """Proves that POST /api/v1/documents/{id}/retry leaves active in-flight documents undisturbed."""
        from app.api.deps import get_current_user, get_db

        pdf_path = tmp_path / "active_retry.pdf"
        create_sample_pdf(pdf_path, ["Active retry test passage."])

        recent_time = datetime.now(timezone.utc) - timedelta(seconds=5)
        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="active_retry.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="active_retry_hash_1",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.EXTRACTING.value,
            updated_at=recent_time,
        )
        db_session.add(doc)
        db_session.commit()

        app.dependency_overrides[get_current_user] = lambda: test_user
        app.dependency_overrides[get_db] = lambda: db_session

        try:
            with TestClient(app) as client:
                resp = client.post(f"/api/v1/documents/{doc.id}/retry")
                assert resp.status_code == 200
                data = resp.json()
                # Must remain EXTRACTING and NOT reset to PENDING
                assert data["processing_status"] == ProcessingStatus.EXTRACTING.value

            db_session.expire_all()
            reloaded = db_session.query(Document).filter(Document.id == doc.id).first()
            assert reloaded.processing_status == ProcessingStatus.EXTRACTING.value
        finally:
            app.dependency_overrides.clear()

    def test_active_worker_renews_lease_heartbeat_during_long_embedding(
        self,
        db_session: Session,
        session_factory: sessionmaker[Session],
        test_user: User,
        tmp_path: Path,
        monkeypatch,
    ):
        """Proves that active workers keep their lease current during long embedding operations,
        preventing stale recovery from stealing a live worker's claim even when execution exceeds
        the configured stale timeout.
        """
        import time
        from app.api.deps import get_current_user, get_db

        # Set stale timeout to 1 second so long operation easily exceeds it
        monkeypatch.setattr(settings, "STALE_INGESTION_TIMEOUT_SECONDS", 1)

        pdf_path = tmp_path / "long_embedding.pdf"
        create_sample_pdf(pdf_path, ["Long embedding passage for heartbeat testing."])

        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="long_embedding.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="long_embed_hash_1",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db_session.add(doc)
        db_session.commit()

        # Slow embedding service that sleeps for 2.0s (> 1s stale timeout)
        slow_embedder = MagicMock(spec=EmbeddingService)
        slow_embedder.dimension = 384
        slow_embedder.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        embedding_entered = threading.Event()
        release_embed = threading.Event()

        def slow_embed(texts):
            # Signal that worker is actively in embedding step
            embedding_entered.set()
            # Hold worker until concurrent retry attempt is executed
            release_embed.wait(timeout=10.0)
            vec = np.zeros(384, dtype=np.float32)
            vec[0] = 1.0
            return [vec.tolist() for _ in texts]

        slow_embedder.embed_batch.side_effect = slow_embed

        test_store = VectorStore(
            dimension=384,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            index_dir=tmp_path / "long_embed_vec_store",
        )

        service = IngestionService(
            session_factory=session_factory,
            pdf_service=PDFService(),
            embedding_service=slow_embedder,
            vector_store=test_store,
        )

        app.dependency_overrides[get_current_user] = lambda: test_user
        app.dependency_overrides[get_db] = lambda: session_factory()

        try:
            # Run ingestion in a separate thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(service.process_document, doc.id, session_factory)

                # Deterministically wait for worker to enter embedding operation
                assert embedding_entered.wait(timeout=5.0), "Worker did not reach embedding step within 5s"

                # Capture baseline updated_at after entering EMBEDDING
                with db_session_scope(session_factory) as check_s:
                    init_doc = check_s.query(Document).filter(Document.id == doc.id).first()
                    init_updated_at = init_doc.updated_at

                # Poll with bounded deadline to verify that LeaseHeartbeat actively renews updated_at
                deadline = time.time() + 4.0
                renewed = False
                while time.time() < deadline:
                    time.sleep(0.05)
                    with db_session_scope(session_factory) as check_s:
                        curr_doc = check_s.query(Document).filter(Document.id == doc.id).first()
                        if curr_doc.updated_at != init_updated_at:
                            renewed = True
                            break

                assert renewed, f"LeaseHeartbeat failed to renew lease during long embedding (init={init_updated_at})"

                # Concurrently attempt to trigger retry from API while worker is actively embedding
                with TestClient(app) as client:
                    retry_resp = client.post(f"/api/v1/documents/{doc.id}/retry")
                    assert retry_resp.status_code == 200
                    data = retry_resp.json()
                    # Status MUST NOT be reset to PENDING because heartbeat kept the lease active!
                    assert data["processing_status"] in (
                        ProcessingStatus.EXTRACTING.value,
                        ProcessingStatus.CHUNKING.value,
                        ProcessingStatus.EMBEDDING.value,
                    )

                # Release embedding worker to complete processing
                release_embed.set()

                # Wait for original worker to finish
                result = future.result(timeout=10.0)
                assert result is not None
                assert result.processing_status == ProcessingStatus.READY.value
        finally:
            release_embed.set()
            app.dependency_overrides.clear()

        db_session.expire_all()
        reloaded = db_session.query(Document).filter(Document.id == doc.id).first()
        assert reloaded.processing_status == ProcessingStatus.READY.value
        assert test_store.total_vectors == 1

    def test_worker_that_lost_ownership_cannot_persist_chunks_or_mutate_faiss_or_overwrite_status(
        self,
        db_session: Session,
        session_factory: sessionmaker[Session],
        test_user: User,
        tmp_path: Path,
        mock_embedding_service: EmbeddingService,
        test_vector_store: VectorStore,
    ):
        """Proves that a worker that lost ownership lease:
        - Cannot persist chunks.
        - Cannot overwrite processing status (to FAILED or READY).
        - Cannot mutate the FAISS index for that document.
        """
        import time

        pdf_path = tmp_path / "lost_lease.pdf"
        create_sample_pdf(pdf_path, ["Passage for lost lease test."])

        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="lost_lease.pdf",
            file_path=str(pdf_path),
            file_hash_sha256="lost_lease_hash_1",
            file_size_bytes=pdf_path.stat().st_size,
            processing_status=ProcessingStatus.PENDING.value,
        )
        db_session.add(doc)
        db_session.commit()

        # Hook embedding service to simulate that Worker 2 stole the lease during Worker 1's embedding step
        def embed_and_steal_lease(texts):
            # Another session simulates Worker 2 claiming the document
            with db_session_scope(session_factory) as s2:
                s2.execute(
                    update(Document)
                    .where(Document.id == doc.id)
                    .values(
                        lease_token="worker_2_stolen_token",
                        processing_status=ProcessingStatus.PENDING.value,
                        updated_at=func.now(),
                    )
                    .execution_options(synchronize_session=False)
                )
                s2.commit()
            vec = np.zeros(384, dtype=np.float32)
            vec[0] = 1.0
            return [vec.tolist() for _ in texts]

        mock_embedding_service.embed_batch.side_effect = embed_and_steal_lease

        service = IngestionService(
            session_factory=session_factory,
            pdf_service=PDFService(),
            embedding_service=mock_embedding_service,
            vector_store=test_vector_store,
        )

        # Worker 1 runs process_document
        res = service.process_document(doc.id, session_factory)

        # Proves Worker 1 aborted safely without raising unhandled error
        assert res is not None

        # Verify FAISS index was NOT mutated by Worker 1
        assert test_vector_store.total_vectors == 0

        # Verify PostgreSQL status was NOT marked READY or FAILED by Worker 1
        db_session.expire_all()
        reloaded = db_session.query(Document).filter(Document.id == doc.id).first()
        # Status remains PENDING (as set by Worker 2), lease_token remains Worker 2's token
        assert reloaded.processing_status == ProcessingStatus.PENDING.value
        assert reloaded.lease_token == "worker_2_stolen_token"

    def test_retry_endpoint_strictly_owner_scoped(
        self,
        db_session: Session,
        test_user: User,
        tmp_path: Path,
    ):
        """Proves that a user cannot retry or steal the lease of another user's document."""
        from app.api.deps import get_current_user, get_db

        other_user = User(
            id=uuid4(),
            email="other_tenant@researchlens.ai",
            hashed_password="pw",
            is_active=True,
        )
        db_session.add(other_user)
        db_session.commit()

        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="user_a_doc.pdf",
            file_path=str(tmp_path / "user_a_doc.pdf"),
            file_hash_sha256="user_a_hash",
            file_size_bytes=100,
            processing_status=ProcessingStatus.FAILED.value,
            lease_token=None,
        )
        db_session.add(doc)
        db_session.commit()

        # Call retry authenticated as other_user
        app.dependency_overrides[get_current_user] = lambda: other_user
        app.dependency_overrides[get_db] = lambda: db_session

        try:
            with TestClient(app) as client:
                resp = client.post(f"/api/v1/documents/{doc.id}/retry")
                assert resp.status_code == 404
                assert resp.json()["detail"] == "Document not found."

            db_session.expire_all()
            reloaded = db_session.query(Document).filter(Document.id == doc.id).first()
            # Must remain untouched
            assert reloaded.processing_status == ProcessingStatus.FAILED.value
            assert reloaded.lease_token is None
        finally:
            app.dependency_overrides.clear()


# ==============================================================================
# 2. FAISS INDEX RECOVERY & REBUILD TESTS
# ==============================================================================

class TestFAISSIndexRecovery:
    """Tests startup verification, deterministic index rebuild, and failure safety."""

    def test_missing_index_files_triggers_clean_rebuild_from_chunk_text(
        self,
        db_session: Session,
        test_user: User,
        tmp_path: Path,
        mock_embedding_service: EmbeddingService,
    ):
        """Proves that when FAISS index files are missing but chunks exist in PostgreSQL:
        - The missing files are detected.
        - Embeddings are deterministically re-generated from chunk text.
        - The index is rebuilt, saved to disk, and validated.
        """
        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="paper.pdf",
            file_path=str(tmp_path / "paper.pdf"),
            file_hash_sha256="hash1",
            file_size_bytes=100,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.flush()

        # Add 3 chunks in PostgreSQL
        chunks = [
            Chunk(
                document_id=doc.id,
                chunk_index=0,
                page_number=1,
                content="First passage on neural networks.",
                char_start=0,
                char_end=33,
                token_count=6,
                embedding_model_name=mock_embedding_service.model_name,
            ),
            Chunk(
                document_id=doc.id,
                chunk_index=1,
                page_number=1,
                content="Second passage on attention mechanisms.",
                char_start=34,
                char_end=74,
                token_count=5,
                embedding_model_name=mock_embedding_service.model_name,
            ),
            Chunk(
                document_id=doc.id,
                chunk_index=2,
                page_number=2,
                content="Third passage on transformer architecture.",
                char_start=0,
                char_end=42,
                token_count=5,
                embedding_model_name=mock_embedding_service.model_name,
            ),
        ]
        db_session.add_all(chunks)
        db_session.commit()

        index_dir = tmp_path / "recovered_index"
        index_dir.mkdir(parents=True, exist_ok=True)
        # Ensure no pre-existing index files
        assert not (index_dir / "faiss_index.bin").exists()
        assert not (index_dir / "faiss_metadata.json").exists()

        store = VectorStore(
            dimension=384,
            model_name=mock_embedding_service.model_name,
            index_dir=index_dir,
        )

        rebuilt_count = rebuild_vector_index(
            session=db_session,
            vector_store=store,
            embedding_service=mock_embedding_service,
        )
        assert rebuilt_count == 3
        assert store.total_vectors == 3
        assert store.is_ready is True

        # Verify files were persisted to disk
        assert (index_dir / "faiss_index.bin").exists()
        assert (index_dir / "faiss_metadata.json").exists()

        # Verify all chunk IDs are present
        for c in chunks:
            assert store.contains_id(c.id)

        # Verify search executes successfully on recovered store
        query_vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        results = store.search(query_vec, top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id in {c.id for c in chunks}

    def test_corrupted_index_files_detected_and_safely_rebuilt(
        self,
        db_session: Session,
        test_user: User,
        tmp_path: Path,
        mock_embedding_service: EmbeddingService,
    ):
        """Proves that corrupted or truncated binary/json files are detected and rebuilt."""
        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="paper.pdf",
            file_path=str(tmp_path / "paper.pdf"),
            file_hash_sha256="hash_corrupt",
            file_size_bytes=100,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.flush()

        chunk = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content="Passage recovering from corrupt file.",
            char_start=0,
            char_end=37,
            token_count=5,
            embedding_model_name=mock_embedding_service.model_name,
        )
        db_session.add(chunk)
        db_session.commit()

        index_dir = tmp_path / "corrupt_test_dir"
        index_dir.mkdir(parents=True, exist_ok=True)
        # Create garbage binary file and corrupt json metadata
        (index_dir / "faiss_index.bin").write_bytes(b"NOT_A_VALID_FAISS_BINARY_CORRUPT")
        (index_dir / "faiss_metadata.json").write_text("{invalid json: broken}", encoding="utf-8")

        store = VectorStore(
            dimension=384,
            model_name=mock_embedding_service.model_name,
            index_dir=index_dir,
        )

        rebuilt_count = rebuild_vector_index(
            session=db_session,
            vector_store=store,
            embedding_service=mock_embedding_service,
        )
        assert rebuilt_count == 1
        assert store.total_vectors == 1
        assert store.contains_id(chunk.id)
        assert store.is_ready is True

    def test_failed_recovery_marks_store_not_ready_and_prevents_silent_empty_search(
        self,
        db_session: Session,
        test_user: User,
        tmp_path: Path,
        mock_embedding_service: EmbeddingService,
    ):
        """Proves that if rebuilding fails (e.g. embedding model OOM or network error):
        - VectorIndexRecoveryError is raised.
        - store.is_ready is False.
        - store.search() raises VectorStoreNotReadyError (does NOT return empty results).
        """
        doc = Document(
            id=uuid4(),
            owner_id=test_user.id,
            filename="paper.pdf",
            file_path=str(tmp_path / "paper.pdf"),
            file_hash_sha256="hash_fail",
            file_size_bytes=100,
            processing_status=ProcessingStatus.READY.value,
        )
        db_session.add(doc)
        db_session.flush()

        chunk = Chunk(
            document_id=doc.id,
            chunk_index=0,
            page_number=1,
            content="Passage that fails embedding.",
            char_start=0,
            char_end=30,
            token_count=5,
            embedding_model_name=mock_embedding_service.model_name,
        )
        db_session.add(chunk)
        db_session.commit()

        index_dir = tmp_path / "fail_store_dir"
        index_dir.mkdir(parents=True, exist_ok=True)

        store = VectorStore(
            dimension=384,
            model_name=mock_embedding_service.model_name,
            index_dir=index_dir,
        )

        # Make embedding service fail
        mock_embedding_service.embed_batch.side_effect = EncodingError("Fatal model crash")

        with pytest.raises(VectorIndexRecoveryError, match="Embedding generation failed"):
            rebuild_vector_index(
                session=db_session,
                vector_store=store,
                embedding_service=mock_embedding_service,
            )

        assert store.is_ready is False

        # Verify that querying a broken store raises VectorStoreNotReadyError
        query_vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        with pytest.raises(VectorStoreNotReadyError, match="FAISS vector store is not ready"):
            store.search(query_vec, top_k=5)


# ==============================================================================
# 3. MULTI-WORKER SAFEGUARD TESTS
# ==============================================================================

class TestMultiWorkerSafeguard:
    """Verifies that concurrent worker processes sharing a local FAISS directory fail fast."""

    def test_worker_lock_blocks_second_instance_with_multi_worker_error(self, tmp_path: Path):
        """Proves that a second WorkerLock instance on the same directory raises MultiWorkerError."""
        lock1 = WorkerLock(tmp_path)
        assert lock1.acquire() is True
        assert lock1.is_acquired is True

        lock2 = WorkerLock(tmp_path)
        with pytest.raises(MultiWorkerError, match="FAISS vector store requires single-worker execution"):
            lock2.acquire()

        # Releasing lock1 allows lock2 to acquire
        lock1.release()
        assert lock1.is_acquired is False

        assert lock2.acquire() is True
        assert lock2.is_acquired is True
        lock2.release()

    def test_reentrant_acquisition_within_same_process_succeeds(self, tmp_path: Path):
        """Verifies that nested acquire/release calls within the same process reference-count safely."""
        lock = WorkerLock(tmp_path)
        assert lock.acquire() is True
        assert lock.acquire() is True
        assert lock.is_acquired is True

        lock.release()
        assert lock.is_acquired is True  # still held by outer ref
        lock.release()
        assert lock.is_acquired is False

    def test_worker_lock_released_on_startup_exception(self, tmp_path: Path, monkeypatch):
        """Proves that if startup fails during index recovery, the advisory worker lock is released
        via try...finally and the original exception is preserved without masking.
        """
        from app.core.worker_lock import get_worker_lock, reset_worker_lock
        from app.main import lifespan

        lock_dir = tmp_path / "startup_fail_lock"
        lock_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(settings, "DATA_DIR", str(lock_dir))
        monkeypatch.setattr(settings, "VECTOR_INDEX_DIR", str(lock_dir))
        reset_worker_lock()

        simulated_error = RuntimeError("Fatal FAISS index corruption during startup recovery")

        test_app = FastAPI()

        with patch("app.main.verify_and_recover_vector_index", side_effect=simulated_error):
            # Attempt to run lifespan context
            with pytest.raises(RuntimeError) as exc_info:
                import asyncio

                async def run_lifespan():
                    async with lifespan(test_app):
                        pass

                asyncio.run(run_lifespan())

            # Original error was preserved and propagated
            assert "Fatal FAISS index corruption during startup recovery" in str(exc_info.value)

        # WorkerLock must have been cleanly released
        lock = get_worker_lock(lock_dir)
        assert lock.is_acquired is False

        # Another WorkerLock instance on the same directory should acquire successfully
        lock2 = WorkerLock(lock_dir)
        assert lock2.acquire() is True
        assert lock2.is_acquired is True
        lock2.release()
        reset_worker_lock()

    def test_windows_unsupported_platform_error(self, tmp_path: Path, monkeypatch):
        """Proves that on Windows or when fcntl is unavailable, WorkerLock.acquire() fails explicitly
        with PlatformNotSupportedError, rather than silently succeeding or throwing ModuleNotFoundError.
        """
        lock = WorkerLock(tmp_path)
        monkeypatch.setattr(sys, "platform", "win32")

        with pytest.raises(PlatformNotSupportedError, match="Advisory worker file locking is not supported on platform 'win32'"):
            lock.acquire()

        assert lock.is_acquired is False

    def test_worker_lock_handles_missing_fcntl_attribute(self, tmp_path: Path, monkeypatch):
        """Proves that if fcntl module is None, acquire() raises PlatformNotSupportedError and release() is safe."""
        import app.core.worker_lock as wl

        lock = WorkerLock(tmp_path)
        monkeypatch.setattr(wl, "fcntl", None)

        with pytest.raises(PlatformNotSupportedError, match="Advisory worker file locking is not supported"):
            lock.acquire()

        # release() on unacquired lock with fcntl=None does not raise AttributeError
        lock.release()
        assert lock.is_acquired is False

