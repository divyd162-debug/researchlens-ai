"""Integration tests for Document management API endpoints, security, and multi-tenant isolation."""
import io
import logging
from pathlib import Path
from unittest.mock import patch
from uuid import UUID, uuid4

import pymupdf
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.database import Base, get_db
from app.core.security import create_access_token
from app.main import app
from app.models.chunk import Chunk
from app.models.collection import Collection, CollectionDocument
from app.models.document import Document, ProcessingStatus
from app.models.user import User
from app.services.ingestion_service import IngestionLeaseLostError, LeaseHeartbeat
from app.services.vector_store import VectorStoreError, get_vector_store


def create_sample_pdf_bytes(pages_text: list[str]) -> bytes:
    """Helper utility to generate valid in-memory PDF bytes."""
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_encrypted_pdf_bytes(password: str = "secret") -> bytes:
    """Helper utility to generate password-encrypted in-memory PDF bytes."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Confidential content")
    pdf_bytes = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password)
    doc.close()
    return pdf_bytes


@pytest.fixture
def db_session():
    """Provides an isolated in-memory SQLite database session."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @sa.event.listens_for(engine, "connect")
    def _enable_fk(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def test_upload_dir(tmp_path: Path, monkeypatch):
    """Overrides settings.UPLOAD_DIR with an isolated temporary directory."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "UPLOAD_DIR", str(upload_dir))
    return upload_dir


@pytest.fixture
def client(db_session: Session, test_upload_dir: Path):
    """Provides a TestClient with get_db overridden to use db_session."""
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def user_a(db_session: Session) -> User:
    """Creates primary test user."""
    user = User(
        email="user_a@example.com",
        hashed_password="hashed_pw_a",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def user_b(db_session: Session) -> User:
    """Creates secondary test user for cross-tenant isolation testing."""
    user = User(
        email="user_b@example.com",
        hashed_password="hashed_pw_b",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers_a(user_a: User) -> dict:
    """Generates valid bearer headers for user_a."""
    token = create_access_token(subject=user_a.id)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers_b(user_b: User) -> dict:
    """Generates valid bearer headers for user_b."""
    token = create_access_token(subject=user_b.id)
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------

class TestUnauthenticatedDocuments:
    """Verifies that all document endpoints require authentication."""

    def test_upload_requires_auth(self, client: TestClient):
        response = client.post("/api/v1/documents", files={"file": ("test.pdf", b"%PDF-1.4\n...")})
        assert response.status_code == 401

    def test_list_requires_auth(self, client: TestClient):
        response = client.get("/api/v1/documents")
        assert response.status_code == 401

    def test_get_requires_auth(self, client: TestClient):
        response = client.get(f"/api/v1/documents/{uuid4()}")
        assert response.status_code == 401

    def test_delete_requires_auth(self, client: TestClient):
        response = client.delete(f"/api/v1/documents/{uuid4()}")
        assert response.status_code == 401


class TestDocumentUploadValidation:
    """Tests input validation, file constraints, and malformed files."""

    def test_upload_non_pdf_extension_rejected(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("malicious.exe", b"binary content")},
        )
        assert response.status_code == 400
        assert "Only PDF documents" in response.json()["detail"]

    def test_upload_zero_byte_file_rejected(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("empty.pdf", b"")},
        )
        assert response.status_code == 400
        assert "empty (0 bytes)" in response.json()["detail"]

    def test_upload_missing_magic_bytes_rejected(self, client: TestClient, auth_headers_a: dict):
        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("fake.pdf", b"This is plain text with no PDF header")},
        )
        assert response.status_code == 400
        assert "magic bytes" in response.json()["detail"]

    def test_upload_encrypted_pdf_rejected(self, client: TestClient, auth_headers_a: dict):
        enc_bytes = create_encrypted_pdf_bytes("secret123")
        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("protected.pdf", enc_bytes)},
        )
        assert response.status_code == 400
        assert "encrypted" in response.json()["detail"].lower()

    def test_upload_corrupted_pdf_rejected_with_sanitized_message(
        self, client: TestClient, auth_headers_a: dict, test_upload_dir: Path
    ):
        """Proves that corrupted PDF errors return stable, generic messages and never expose internal filesystem paths."""
        corrupted_bytes = b"%PDF-1.4\nCORRUPTED_BYTES_WITHOUT_VALID_XREF_OR_TRAILER"
        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("broken.pdf", corrupted_bytes)},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail == "Corrupted or invalid PDF file."

        # Verify absolute upload directory, temp filename, and system paths do not leak
        assert str(test_upload_dir) not in detail
        assert ".tmp" not in detail
        assert "/" not in detail
        assert "\\" not in detail

    def test_upload_exceeding_size_limit_rejected(
        self, client: TestClient, auth_headers_a: dict, monkeypatch, test_upload_dir: Path
    ):
        # Set a tiny size limit to test streaming size threshold
        monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 100)
        pdf_bytes = create_sample_pdf_bytes(["A" * 500])

        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("large.pdf", pdf_bytes)},
        )
        assert response.status_code == 413
        assert "exceeds maximum allowed limit" in response.json()["detail"]

        # Ensure no orphan temporary file is left in upload directory
        files_in_dir = list(test_upload_dir.iterdir())
        assert len(files_in_dir) == 0


class TestDocumentUploadSuccess:
    """Tests successful document uploads and metadata persistence."""

    def test_upload_valid_single_page_pdf(
        self, client: TestClient, auth_headers_a: dict, test_upload_dir: Path
    ):
        pdf_bytes = create_sample_pdf_bytes(["ResearchLens AI Overview"])
        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("overview.pdf", pdf_bytes, "application/pdf")},
            data={"title": "Custom Overview Title"},
        )
        assert response.status_code == 201
        data = response.json()

        assert data["filename"] == "overview.pdf"
        assert data["title"] == "Custom Overview Title"
        assert data["page_count"] == 1
        assert data["processing_status"] == ProcessingStatus.PENDING.value
        assert data["file_size_bytes"] == len(pdf_bytes)
        assert "file_path" not in data  # Never expose internal filesystem paths

        # Verify disk persistence
        doc_id = data["id"]
        stored_file = test_upload_dir / f"{doc_id}.pdf"
        assert stored_file.exists()
        assert stored_file.read_bytes() == pdf_bytes

    def test_upload_valid_multipage_pdf_without_title(
        self, client: TestClient, auth_headers_a: dict, test_upload_dir: Path
    ):
        pdf_bytes = create_sample_pdf_bytes(["Page 1 content", "Page 2 content", "Page 3 content"])
        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("multipage.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()

        assert data["filename"] == "multipage.pdf"
        assert data["title"] is None
        assert data["page_count"] == 3
        assert data["processing_status"] == ProcessingStatus.PENDING.value

    def test_filename_sanitization_and_path_traversal(
        self, client: TestClient, auth_headers_a: dict, test_upload_dir: Path
    ):
        pdf_bytes = create_sample_pdf_bytes(["Path traversal test"])
        # Attempt traversal attack in filename
        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("../../etc/passwd.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()

        # Filename should be stripped to basename only
        assert data["filename"] == "passwd.pdf"
        doc_id = data["id"]
        # File is safely written inside test_upload_dir as {uuid}.pdf
        assert (test_upload_dir / f"{doc_id}.pdf").exists()


class TestDuplicateUploadsAndUniqueness:
    """Tests duplicate document handling per owner, across tenants, and database uniqueness."""

    def test_duplicate_upload_same_user_rejected(self, client: TestClient, auth_headers_a: dict):
        pdf_bytes = create_sample_pdf_bytes(["Duplicate Test Content"])

        # First upload succeeds
        r1 = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("doc1.pdf", pdf_bytes, "application/pdf")},
        )
        assert r1.status_code == 201

        # Second upload with identical content returns 409 Conflict
        r2 = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("doc2.pdf", pdf_bytes, "application/pdf")},
        )
        assert r2.status_code == 409
        assert "identical content has already been uploaded" in r2.json()["detail"]

    def test_identical_upload_different_users_allowed(
        self, client: TestClient, auth_headers_a: dict, auth_headers_b: dict
    ):
        pdf_bytes = create_sample_pdf_bytes(["Multi-tenant Shared PDF"])

        # User A uploads
        r1 = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("user_a_doc.pdf", pdf_bytes, "application/pdf")},
        )
        assert r1.status_code == 201

        # User B uploads exact same file -> allowed due to tenant isolation
        r2 = client.post(
            "/api/v1/documents",
            headers=auth_headers_b,
            files={"file": ("user_b_doc.pdf", pdf_bytes, "application/pdf")},
        )
        assert r2.status_code == 201
        assert r1.json()["id"] != r2.json()["id"]

    def test_database_uniqueness_constraint_enforced_at_orm_level(
        self, db_session: Session, user_a: User
    ):
        """Verifies that the database schema strictly prevents two documents with identical (owner_id, file_hash_sha256).
        
        Architecture Note on Concurrency:
        SQLite in-memory test databases run in single-connection mode and cannot reliably simulate true multi-process
        PostgreSQL transactional races. However, this test proves that the database engine enforces the
        uq_document_owner_hash UniqueConstraint on commit, guaranteeing that concurrent inserts of identical
        owner/hash pairs will fail at the database level rather than creating duplicate records.
        A dedicated PostgreSQL integration test in CI should verify end-to-end multi-connection concurrency.
        """
        doc1 = Document(
            owner_id=user_a.id,
            filename="paper1.pdf",
            file_path="uploads/doc1.pdf",
            file_hash_sha256="same_sha256_hash",
            file_size_bytes=1000,
        )
        doc2 = Document(
            owner_id=user_a.id,
            filename="paper2.pdf",
            file_path="uploads/doc2.pdf",
            file_hash_sha256="same_sha256_hash",
            file_size_bytes=1000,
        )
        db_session.add(doc1)
        db_session.commit()

        db_session.add(doc2)
        with pytest.raises(IntegrityError) as exc_info:
            db_session.commit()
        db_session.rollback()

        err_msg = str(exc_info.value).lower()
        assert "uq_document_owner_hash" in err_msg or "unique constraint failed: documents.owner_id, documents.file_hash_sha256" in err_msg

    def test_upload_race_condition_uniqueness_integrity_error_handled(
        self, client: TestClient, auth_headers_a: dict, test_upload_dir: Path
    ):
        """Proves that a concurrent commit uniqueness collision triggers rollback, cleanup, and returns 409."""
        pdf_bytes = create_sample_pdf_bytes(["Concurrent race test content"])

        with patch.object(
            Session,
            "commit",
            side_effect=IntegrityError("UNIQUE constraint failed: documents.owner_id, documents.file_hash_sha256", None, None),
        ):
            response = client.post(
                "/api/v1/documents",
                headers=auth_headers_a,
                files={"file": ("race.pdf", pdf_bytes)},
            )
            assert response.status_code == 409
            assert "already been uploaded" in response.json()["detail"]

        # Ensure no orphan temporary or final files remain
        files_remaining = list(test_upload_dir.iterdir())
        assert len(files_remaining) == 0

    def test_unrelated_integrity_error_not_masked_as_duplicate(
        self, client: TestClient, auth_headers_a: dict, test_upload_dir: Path
    ):
        """Proves that unrelated database integrity failures (e.g. FK constraint) are not misclassified as duplicate uploads."""
        pdf_bytes = create_sample_pdf_bytes(["Unrelated integrity failure test"])

        with patch.object(
            Session,
            "commit",
            side_effect=IntegrityError("FOREIGN KEY constraint failed on foreign key reference", None, None),
        ):
            # Should not map to 409 Conflict
            with pytest.raises(IntegrityError):
                client.post(
                    "/api/v1/documents",
                    headers=auth_headers_a,
                    files={"file": ("fk_fail.pdf", pdf_bytes)},
                )

        # File is still cleaned up on rollback
        files_remaining = list(test_upload_dir.iterdir())
        assert len(files_remaining) == 0


class TestFilePromotionAndCleanupRobustness:
    """Tests file promotion failures and temporary file cleanup."""

    def test_file_promotion_rename_failure_cleans_up(
        self, client: TestClient, auth_headers_a: dict, test_upload_dir: Path
    ):
        """Proves that an OSError during temporary-to-final promotion cleans up temporary files and returns 500."""
        pdf_bytes = create_sample_pdf_bytes(["Promotion Failure Test"])

        with patch.object(Path, "rename", side_effect=OSError("Simulated cross-device or permission rename error")):
            response = client.post(
                "/api/v1/documents",
                headers=auth_headers_a,
                files={"file": ("fail_promo.pdf", pdf_bytes)},
            )
            assert response.status_code == 500
            assert "Failed to store uploaded document file" in response.json()["detail"]

        # Assert no temporary files remained
        files_in_dir = list(test_upload_dir.iterdir())
        assert len(files_in_dir) == 0

    def test_file_promotion_failure_with_cleanup_warning_logged(
        self, client: TestClient, auth_headers_a: dict, caplog
    ):
        """Proves that if cleanup itself fails during promotion error, the error is logged without masking the root cause."""
        pdf_bytes = create_sample_pdf_bytes(["Double Failure Test"])

        with patch.object(Path, "rename", side_effect=OSError("Disk write error")):
            with patch.object(Path, "unlink", side_effect=OSError("Disk locked")):
                with caplog.at_level(logging.WARNING):
                    response = client.post(
                        "/api/v1/documents",
                        headers=auth_headers_a,
                        files={"file": ("double_fail.pdf", pdf_bytes)},
                    )
                    assert response.status_code == 500
                    assert "Failed to store uploaded document file" in response.json()["detail"]
                    assert "Failed to clean up" in caplog.text


class TestDocumentRetrievalAndListing:
    """Tests get by ID and paginated listing with tenant scoping."""

    def test_get_document_success(self, client: TestClient, auth_headers_a: dict):
        pdf_bytes = create_sample_pdf_bytes(["Single Doc Test"])
        upload_resp = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("get_test.pdf", pdf_bytes)},
            data={"title": "Get Test"},
        )
        doc_id = upload_resp.json()["id"]

        get_resp = client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers_a)
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == doc_id
        assert get_resp.json()["title"] == "Get Test"

    def test_get_nonexistent_document_returns_404(self, client: TestClient, auth_headers_a: dict):
        response = client.get(f"/api/v1/documents/{uuid4()}", headers=auth_headers_a)
        assert response.status_code == 404
        assert response.json()["detail"] == "Document not found."

    def test_cross_user_isolation_returns_404(
        self, client: TestClient, auth_headers_a: dict, auth_headers_b: dict
    ):
        pdf_bytes = create_sample_pdf_bytes(["User A Private Document"])
        upload_resp = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("private_a.pdf", pdf_bytes)},
        )
        doc_a_id = upload_resp.json()["id"]

        # User B attempts to access User A's document -> must be 404, not 403
        get_resp = client.get(f"/api/v1/documents/{doc_a_id}", headers=auth_headers_b)
        assert get_resp.status_code == 404
        assert get_resp.json()["detail"] == "Document not found."

    def test_list_documents_pagination_and_isolation(
        self, client: TestClient, auth_headers_a: dict, auth_headers_b: dict
    ):
        # Create 3 documents for User A
        for i in range(3):
            pdf_bytes = create_sample_pdf_bytes([f"User A doc {i}"])
            client.post(
                "/api/v1/documents",
                headers=auth_headers_a,
                files={"file": (f"doc_a_{i}.pdf", pdf_bytes)},
            )

        # Create 1 document for User B
        pdf_bytes_b = create_sample_pdf_bytes(["User B doc"])
        client.post(
            "/api/v1/documents",
            headers=auth_headers_b,
            files={"file": ("doc_b.pdf", pdf_bytes_b)},
        )

        # User A list: should return exactly 3 items
        resp_a = client.get("/api/v1/documents", headers=auth_headers_a)
        assert resp_a.status_code == 200
        data_a = resp_a.json()
        assert data_a["total"] == 3
        assert len(data_a["items"]) == 3

        # User B list: should return exactly 1 item
        resp_b = client.get("/api/v1/documents", headers=auth_headers_b)
        assert resp_b.status_code == 200
        data_b = resp_b.json()
        assert data_b["total"] == 1
        assert len(data_b["items"]) == 1

        # Test limit and offset on User A
        paged_resp = client.get("/api/v1/documents?limit=2&offset=1", headers=auth_headers_a)
        assert paged_resp.status_code == 200
        paged_data = paged_resp.json()
        assert paged_data["total"] == 3
        assert len(paged_data["items"]) == 2
        assert paged_data["limit"] == 2
        assert paged_data["offset"] == 1


class TestDocumentDeletionAndCleanup:
    """Tests document deletion, unlink observability, and failure cleanup behaviors."""

    def test_delete_document_success(
        self, client: TestClient, auth_headers_a: dict, test_upload_dir: Path
    ):
        pdf_bytes = create_sample_pdf_bytes(["Document to be deleted"])
        upload_resp = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("delete_me.pdf", pdf_bytes)},
        )
        doc_id = upload_resp.json()["id"]
        stored_file = test_upload_dir / f"{doc_id}.pdf"
        assert stored_file.exists()

        # Delete document
        del_resp = client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers_a)
        assert del_resp.status_code == 200
        assert del_resp.json()["message"] == "Document deleted successfully"

        # Verify disk file is unlinked
        assert not stored_file.exists()

        # Verify subsequent retrieval returns 404
        get_resp = client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers_a)
        assert get_resp.status_code == 404

    def test_delete_nonexistent_or_other_users_document_returns_404(
        self, client: TestClient, auth_headers_a: dict, auth_headers_b: dict, test_upload_dir: Path
    ):
        # Nonexistent document
        del_missing = client.delete(f"/api/v1/documents/{uuid4()}", headers=auth_headers_a)
        assert del_missing.status_code == 404

        # Other user's document
        pdf_bytes = create_sample_pdf_bytes(["User A precious document"])
        upload_resp = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("protected.pdf", pdf_bytes)},
        )
        doc_a_id = upload_resp.json()["id"]

        del_unauthorized = client.delete(f"/api/v1/documents/{doc_a_id}", headers=auth_headers_b)
        assert del_unauthorized.status_code == 404

        # Verify User A's file and record are still intact
        assert (test_upload_dir / f"{doc_a_id}.pdf").exists()
        get_resp = client.get(f"/api/v1/documents/{doc_a_id}", headers=auth_headers_a)
        assert get_resp.status_code == 200

    def test_delete_unlink_failure_logs_recovery_error_and_preserves_db_delete(
        self, client: TestClient, auth_headers_a: dict, test_upload_dir: Path, caplog
    ):
        """Proves that if unlinking fails upon deletion, the database record remains deleted and the failure is logged for recovery."""
        pdf_bytes = create_sample_pdf_bytes(["Unlink failure logging test"])
        upload_resp = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("unlink_fail.pdf", pdf_bytes)},
        )
        doc_id = upload_resp.json()["id"]

        with patch.object(Path, "unlink", side_effect=PermissionError("Permission denied during file unlink")):
            with caplog.at_level(logging.ERROR):
                del_resp = client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers_a)
                assert del_resp.status_code == 200
                assert del_resp.json()["message"] == "Document deleted successfully"

                # Verify failure was logged at ERROR level with orphan file recovery info
                assert "Orphan file recovery required" in caplog.text
                assert str(doc_id) in caplog.text

        # Verify database record was deleted (subsequent retrieval returns 404)
        get_resp = client.get(f"/api/v1/documents/{doc_id}", headers=auth_headers_a)
        assert get_resp.status_code == 404

    def test_delete_ready_document_purges_faiss_vectors_and_repeated_delete_returns_404(
        self, client: TestClient, auth_headers_a: dict, test_upload_dir: Path, db_session: Session
    ):
        """Proves that deleting a READY document removes its FAISS vectors, unlinks the file, and repeated deletion is safe (404)."""
        pdf_bytes = create_sample_pdf_bytes(["Sample text for ready document deletion"])
        upload_resp = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("ready_delete.pdf", pdf_bytes)},
        )
        doc_id = upload_resp.json()["id"]
        doc_uuid = UUID(doc_id)

        # Upload in TestClient runs ingestion, so document is READY with chunks and vectors
        doc = db_session.get(Document, doc_uuid)
        assert doc.processing_status == ProcessingStatus.READY.value
        chunk_ids = [c.id for c in doc.chunks]
        assert len(chunk_ids) > 0

        store = get_vector_store()
        for cid in chunk_ids:
            assert store.contains_id(cid) is True

        # First delete succeeds
        del_resp = client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers_a)
        assert del_resp.status_code == 200
        assert del_resp.json()["message"] == "Document deleted successfully"

        # Verify vectors are purged from FAISS
        for cid in chunk_ids:
            assert store.contains_id(cid) is False

        # Verify file is unlinked
        stored_file = test_upload_dir / f"{doc_id}.pdf"
        assert not stored_file.exists()

        # Repeated delete request returns 404 (safe and idempotent)
        del_repeat = client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers_a)
        assert del_repeat.status_code == 404
        assert del_repeat.json()["detail"] == "Document not found."

    def test_delete_document_removes_collection_associations_and_preserves_collection(
        self, client: TestClient, auth_headers_a: dict, db_session: Session
    ):
        """Proves deleting a document removes collection_documents rows while preserving the collection itself."""
        # Create a collection
        col_resp = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": "Neuroscience Papers", "description": "Brain research collection"},
        )
        assert col_resp.status_code == 201
        col_id = col_resp.json()["id"]
        col_uuid = UUID(col_id)

        # Upload document
        pdf_bytes = create_sample_pdf_bytes(["Neuroscience research content"])
        upload_resp = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("brain_paper.pdf", pdf_bytes)},
        )
        doc_id = upload_resp.json()["id"]
        doc_uuid = UUID(doc_id)

        # Associate document with collection
        cd = CollectionDocument(collection_id=col_uuid, document_id=doc_uuid)
        db_session.add(cd)
        db_session.commit()

        # Verify association exists
        assert db_session.get(CollectionDocument, (col_uuid, doc_uuid)) is not None

        # Delete document
        del_resp = client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers_a)
        assert del_resp.status_code == 200

        # Expire session identity map to read post-cascade state from database
        db_session.expire_all()

        # Collection still exists
        col = db_session.get(Collection, col_uuid)
        assert col is not None
        assert col.name == "Neuroscience Papers"

        # Association is deleted via cascade
        assert db_session.get(CollectionDocument, (col_uuid, doc_uuid)) is None

    def test_delete_document_vector_cleanup_failure_logs_error_and_preserves_db_delete(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, caplog
    ):
        """Proves that if FAISS vector store removal fails, the failure is logged at ERROR level and the DB delete still completes."""
        pdf_bytes = create_sample_pdf_bytes(["Vector cleanup failure test"])
        upload_resp = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("vec_fail.pdf", pdf_bytes)},
        )
        doc_id = upload_resp.json()["id"]
        doc_uuid = UUID(doc_id)

        doc = db_session.get(Document, doc_uuid)
        assert len(doc.chunks) > 0

        store = get_vector_store()
        with patch.object(store, "remove_vectors", side_effect=VectorStoreError("Simulated FAISS vector deletion error")):
            with caplog.at_level(logging.ERROR):
                del_resp = client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers_a)
                assert del_resp.status_code == 200
                assert del_resp.json()["message"] == "Document deleted successfully"

                # Verify failure was logged at ERROR level with orphan vector cleanup info
                assert "Orphan vector cleanup required" in caplog.text
                assert str(doc_id) in caplog.text

        # Verify DB delete succeeded
        assert db_session.get(Document, doc_uuid) is None

    def test_delete_document_fences_in_flight_ingestion_worker(
        self, client: TestClient, auth_headers_a: dict, db_session: Session
    ):
        """Proves that deleting a document while a background worker holds an active lease fences the worker."""
        pdf_bytes = create_sample_pdf_bytes(["Worker race test"])
        upload_resp = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("worker_race.pdf", pdf_bytes)},
        )
        doc_id = upload_resp.json()["id"]
        doc_uuid = UUID(doc_id)

        # Simulate in-flight worker holding an active lease
        worker_lease = "active-worker-lease-token-12345"
        doc = db_session.get(Document, doc_uuid)
        doc.processing_status = ProcessingStatus.EXTRACTING.value
        doc.lease_token = worker_lease
        db_session.commit()

        # User deletes document while worker is "in flight"
        del_resp = client.delete(f"/api/v1/documents/{doc_id}", headers=auth_headers_a)
        assert del_resp.status_code == 200

        # Simulate worker checking its heartbeat / lease
        heartbeat = LeaseHeartbeat(
            session_factory=lambda: db_session,
            document_id=doc_uuid,
            lease_token=worker_lease,
        )
        with pytest.raises(IngestionLeaseLostError):
            heartbeat.verify_current(db_session)


class TestRequestSizeLimitMiddleware:
    """Tests application/ASGI-boundary request size limit enforcement and multipart upload sizing."""

    def test_oversized_content_length_rejected_before_processing(
        self, client: TestClient, auth_headers_a: dict
    ):
        """Proves that a request with Content-Length exceeding MAX_REQUEST_BODY_SIZE_BYTES is rejected with 413 immediately."""
        # 30 MB declared Content-Length (limit is 25 MB + 256 KB)
        headers = {**auth_headers_a, "Content-Length": str(30 * 1024 * 1024)}
        response = client.post(
            "/api/v1/documents",
            headers=headers,
            content=b"dummy small body",
        )
        assert response.status_code == 413
        assert "exceeds maximum allowed limit" in response.json()["detail"]

    def test_oversized_payload_without_content_length_rejected(
        self, client: TestClient, auth_headers_a: dict, monkeypatch
    ):
        """Proves that chunked streaming requests exceeding MAX_REQUEST_BODY_SIZE_BYTES without Content-Length

        are metered at the ASGI boundary and rejected with 413.
        """
        monkeypatch.setattr(settings, "MAX_REQUEST_BODY_SIZE_BYTES", 500)

        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        headers = {
            **auth_headers_a,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }

        def gen():
            yield f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"test.pdf\"\r\nContent-Type: application/pdf\r\n\r\n".encode()
            for _ in range(10):
                yield b"X" * 100
            yield f"\r\n--{boundary}--\r\n".encode()

        response = client.post("/api/v1/documents", headers=headers, content=gen())
        assert response.status_code == 413
        assert "Request payload exceeds maximum allowed limit of 500 bytes." in response.json()["detail"]

    def test_valid_pdf_comfortably_below_limit(
        self, client: TestClient, auth_headers_a: dict, test_upload_dir: Path
    ):
        """Proves that a valid PDF comfortably below MAX_UPLOAD_SIZE_BYTES uploads successfully."""
        pdf_bytes = create_sample_pdf_bytes(["A document comfortably below size limits."])
        assert len(pdf_bytes) < settings.MAX_UPLOAD_SIZE_BYTES

        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("comfortable.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "comfortable.pdf"
        assert data["file_size_bytes"] == len(pdf_bytes)

    def test_valid_pdf_close_to_file_size_limit_succeeds(
        self, client: TestClient, auth_headers_a: dict, monkeypatch, test_upload_dir: Path
    ):
        """Proves that a PDF whose file size is within MAX_UPLOAD_SIZE_BYTES succeeds

        even when multipart framing pushes the raw HTTP request body size above MAX_UPLOAD_SIZE_BYTES.
        Verifies that MAX_REQUEST_BODY_SIZE_BYTES = MAX_UPLOAD_SIZE_BYTES + MULTIPART_OVERHEAD_BYTES
        properly accommodates multipart overhead without prematurely rejecting valid files.
        """
        file_limit = 5000
        # Configure file limit and request body limit with overhead buffer
        monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", file_limit)
        monkeypatch.setattr(settings, "MAX_REQUEST_BODY_SIZE_BYTES", file_limit + settings.MULTIPART_OVERHEAD_BYTES)

        # Generate a valid PDF of 4,800 bytes (within the 5,000 byte limit)
        base_pdf = create_sample_pdf_bytes(["Near limit test document"])
        padding = b"\n%" + (b"0" * (4800 - len(base_pdf) - 2))
        near_limit_pdf = base_pdf + padding
        assert len(near_limit_pdf) == 4800
        assert len(near_limit_pdf) <= file_limit

        # Multipart form boundaries, headers, and fields add ~300 bytes of wire overhead,
        # pushing total request body size to ~5,100 bytes (exceeding file_limit, but within request_body_limit)
        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("near_limit.pdf", near_limit_pdf, "application/pdf")},
            data={"title": "Near Limit PDF Document"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "near_limit.pdf"
        assert data["file_size_bytes"] == 4800

    def test_pdf_exceeding_file_size_limit_rejected(
        self, client: TestClient, auth_headers_a: dict, monkeypatch, test_upload_dir: Path
    ):
        """Proves that a PDF exceeding MAX_UPLOAD_SIZE_BYTES is rejected strictly with 413 by DocumentService,

        even when the total request body is within MAX_REQUEST_BODY_SIZE_BYTES.
        """
        file_limit = 5000
        monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", file_limit)
        monkeypatch.setattr(settings, "MAX_REQUEST_BODY_SIZE_BYTES", file_limit + 10000)

        # Generate a valid PDF of 5,500 bytes (> 5,000 bytes)
        base_pdf = create_sample_pdf_bytes(["Oversized PDF content"])
        padding = b"\n%" + (b"0" * (5500 - len(base_pdf) - 2))
        oversized_pdf = base_pdf + padding
        assert len(oversized_pdf) == 5500
        assert len(oversized_pdf) > file_limit

        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("too_large.pdf", oversized_pdf, "application/pdf")},
        )
        assert response.status_code == 413
        assert "File size exceeds maximum allowed limit of 5000 bytes." in response.json()["detail"]

    def test_pdf_at_or_below_page_limit_succeeds(
        self, client: TestClient, auth_headers_a: dict, test_upload_dir: Path
    ):
        """Proves that a PDF with exactly MAX_DOCUMENT_PAGES (default 100) is accepted."""
        pdf_100 = create_sample_pdf_bytes([f"Page {i}" for i in range(100)])
        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("exact_100.pdf", pdf_100, "application/pdf")},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["page_count"] == 100

    def test_pdf_exceeding_page_limit_rejected_and_cleans_up(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, test_upload_dir: Path
    ):
        """Proves that a PDF exceeding MAX_DOCUMENT_PAGES is rejected with 400 and cleans up temp files."""
        pdf_101 = create_sample_pdf_bytes([f"Page {i}" for i in range(101)])
        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("too_many_pages.pdf", pdf_101, "application/pdf")},
        )
        assert response.status_code == 400
        assert "Document exceeds maximum allowed limit of 100 pages (got 101)." in response.json()["detail"]

        # Ensure no orphaned temp files or uploaded PDFs remain in upload directory
        assert list(test_upload_dir.glob("*.tmp")) == []
        assert list(test_upload_dir.glob("*.pdf")) == []

        # Ensure no document row was written to database
        assert db_session.query(Document).count() == 0

    def test_pdf_page_limit_configurable_via_settings(
        self, client: TestClient, auth_headers_a: dict, monkeypatch, test_upload_dir: Path
    ):
        """Proves that MAX_DOCUMENT_PAGES limit is dynamically configurable."""
        monkeypatch.setattr(settings, "MAX_DOCUMENT_PAGES", 5)

        pdf_6 = create_sample_pdf_bytes([f"Page {i}" for i in range(6)])
        response = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("six_pages.pdf", pdf_6, "application/pdf")},
        )
        assert response.status_code == 400
        assert "Document exceeds maximum allowed limit of 5 pages (got 6)." in response.json()["detail"]

        pdf_5 = create_sample_pdf_bytes([f"Page {i}" for i in range(5)])
        response_ok = client.post(
            "/api/v1/documents",
            headers=auth_headers_a,
            files={"file": ("five_pages.pdf", pdf_5, "application/pdf")},
        )
        assert response_ok.status_code == 201
        assert response_ok.json()["page_count"] == 5

    def test_document_settings_validation(self, monkeypatch):
        """Verifies that validate_document_settings rejects zero or negative configuration."""
        settings.validate_document_settings()  # Should succeed with defaults

        monkeypatch.setattr(settings, "MAX_DOCUMENT_PAGES", 0)
        with pytest.raises(ValueError, match="MAX_DOCUMENT_PAGES must be a positive integer"):
            settings.validate_document_settings()

        monkeypatch.setattr(settings, "MAX_DOCUMENT_PAGES", -5)
        with pytest.raises(ValueError, match="MAX_DOCUMENT_PAGES must be a positive integer"):
            settings.validate_document_settings()

        monkeypatch.setattr(settings, "MAX_DOCUMENT_PAGES", 100)
        monkeypatch.setattr(settings, "MAX_UPLOAD_SIZE_BYTES", 0)
        with pytest.raises(ValueError, match="MAX_UPLOAD_SIZE_BYTES must be a positive integer"):
            settings.validate_document_settings()


class TestStaleTemporaryFileCleanup:
    """Test suite for cleanup_stale_temp_files utility."""

    def test_cleanup_stale_tmp_files_removed(self, tmp_path: Path):
        """Verifies that .tmp files older than max_age_seconds are safely removed."""
        import os
        import time
        from app.services.document_service import cleanup_stale_temp_files

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()

        stale_file = upload_dir / "stale_upload.tmp"
        stale_file.write_bytes(b"%PDF-sample")
        # Set mtime to 2 hours ago
        stale_mtime = time.time() - 7200
        os.utime(stale_file, (stale_mtime, stale_mtime))

        purged = cleanup_stale_temp_files(upload_dir, max_age_seconds=3600)
        assert purged == 1
        assert not stale_file.exists()

    def test_cleanup_recent_tmp_files_preserved(self, tmp_path: Path):
        """Verifies that .tmp files newer than max_age_seconds are preserved."""
        import os
        import time
        from app.services.document_service import cleanup_stale_temp_files

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()

        recent_file = upload_dir / "recent_upload.tmp"
        recent_file.write_bytes(b"%PDF-sample")
        # Set mtime to 60 seconds ago (< 1 hour)
        recent_mtime = time.time() - 60
        os.utime(recent_file, (recent_mtime, recent_mtime))

        purged = cleanup_stale_temp_files(upload_dir, max_age_seconds=3600)
        assert purged == 0
        assert recent_file.exists()

    def test_cleanup_preserves_pdfs_gitkeep_and_subdirectories(self, tmp_path: Path):
        """Verifies that .pdf files, .gitkeep, and directories are never deleted, even if old."""
        import os
        import time
        from app.services.document_service import cleanup_stale_temp_files

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()

        old_pdf = upload_dir / "research_doc.pdf"
        old_pdf.write_bytes(b"%PDF-valid-research-paper")
        old_gitkeep = upload_dir / ".gitkeep"
        old_gitkeep.write_bytes(b"keep")
        sub_dir = upload_dir / "subdir"
        sub_dir.mkdir()

        # Set all mtimes to 30 days ago
        ancient_mtime = time.time() - (86400 * 30)
        os.utime(old_pdf, (ancient_mtime, ancient_mtime))
        os.utime(old_gitkeep, (ancient_mtime, ancient_mtime))

        purged = cleanup_stale_temp_files(upload_dir, max_age_seconds=3600)
        assert purged == 0
        assert old_pdf.exists()
        assert old_gitkeep.exists()
        assert sub_dir.exists()

    def test_cleanup_ignores_symlinks(self, tmp_path: Path):
        """Verifies that symlinks are strictly skipped and never deleted or followed."""
        import os
        import time
        from app.services.document_service import cleanup_stale_temp_files

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        external_dir = tmp_path / "external"
        external_dir.mkdir()

        target_file = external_dir / "external_target.tmp"
        target_file.write_bytes(b"external")
        old_mtime = time.time() - 7200
        os.utime(target_file, (old_mtime, old_mtime))

        symlink_file = upload_dir / "symlinked.tmp"
        try:
            symlink_file.symlink_to(target_file)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform/filesystem.")

        purged = cleanup_stale_temp_files(upload_dir, max_age_seconds=3600)
        assert purged == 0
        assert symlink_file.is_symlink()
        assert target_file.exists()

    def test_cleanup_handles_filesystem_permission_errors_gracefully(self, tmp_path: Path, monkeypatch):
        """Verifies that unexpected filesystem errors do not raise or crash application startup."""
        from app.services.document_service import cleanup_stale_temp_files

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()

        def mock_iterdir(_self):
            raise PermissionError("Access denied to upload directory")

        monkeypatch.setattr(Path, "iterdir", mock_iterdir)

        # Should log a warning and return 0 without raising
        purged = cleanup_stale_temp_files(upload_dir, max_age_seconds=3600)
        assert purged == 0

    def test_cleanup_nonexistent_or_invalid_directory_returns_zero(self, tmp_path: Path):
        """Verifies that nonexistent directory or regular file path safely returns 0."""
        from app.services.document_service import cleanup_stale_temp_files

        # Nonexistent path
        nonexistent = tmp_path / "does_not_exist"
        assert cleanup_stale_temp_files(nonexistent) == 0

        # Regular file instead of directory
        reg_file = tmp_path / "file.txt"
        reg_file.write_text("hello")
        assert cleanup_stale_temp_files(reg_file) == 0

