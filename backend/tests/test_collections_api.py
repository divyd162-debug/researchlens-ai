"""Integration tests for Collections API endpoints and multi-tenant isolation."""
from unittest.mock import patch
from uuid import uuid4

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.core.security import create_access_token
from app.main import app
from app.models.collection import Collection, CollectionDocument
from app.models.document import Document
from app.models.user import User


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
def client(db_session: Session):
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


class TestUnauthenticatedCollections:
    """Verifies that all collection endpoints require authentication."""

    def test_create_collection_unauthenticated(self, client: TestClient):
        res = client.post("/api/v1/collections", json={"name": "Test"})
        assert res.status_code == 401

    def test_list_collections_unauthenticated(self, client: TestClient):
        res = client.get("/api/v1/collections")
        assert res.status_code == 401

    def test_get_collection_unauthenticated(self, client: TestClient):
        res = client.get(f"/api/v1/collections/{uuid4()}")
        assert res.status_code == 401

    def test_update_collection_unauthenticated(self, client: TestClient):
        res = client.patch(f"/api/v1/collections/{uuid4()}", json={"name": "New"})
        assert res.status_code == 401

    def test_delete_collection_unauthenticated(self, client: TestClient):
        res = client.delete(f"/api/v1/collections/{uuid4()}")
        assert res.status_code == 401


class TestCreateCollection:
    """Tests for POST /api/v1/collections."""

    def test_create_collection_success(self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User):
        res = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={
                "name": "  Quantum Error Correction  ",
                "description": "Papers on surface codes and topological memory.",
            },
        )
        assert res.status_code == 201
        data = res.json()

        assert "id" in data
        assert data["name"] == "Quantum Error Correction"  # Normalized / trimmed
        assert data["description"] == "Papers on surface codes and topological memory."
        assert data["document_count"] == 0
        assert "created_at" in data
        assert "updated_at" in data

        # Security check: owner_id must NOT be leaked
        assert "owner_id" not in data

        # Check DB state
        from uuid import UUID
        db_col = db_session.query(Collection).filter(Collection.id == UUID(data["id"])).first()
        assert db_col is not None
        assert db_col.owner_id == user_a.id
        assert db_col.name == "Quantum Error Correction"


    def test_create_duplicate_name_same_user_returns_409(self, client: TestClient, auth_headers_a: dict):
        res1 = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": "Deep Learning"},
        )
        assert res1.status_code == 201

        # Attempt to create collection with same name
        res2 = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": "  Deep Learning  "},
        )
        assert res2.status_code == 409
        assert "already exists" in res2.json()["detail"].lower()

    def test_same_name_different_user_succeeds(self, client: TestClient, auth_headers_a: dict, auth_headers_b: dict):
        res_a = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": "Shared Name Collection"},
        )
        assert res_a.status_code == 201

        # User B creating with same name must succeed
        res_b = client.post(
            "/api/v1/collections",
            headers=auth_headers_b,
            json={"name": "Shared Name Collection"},
        )
        assert res_b.status_code == 201
        assert res_a.json()["id"] != res_b.json()["id"]

    @pytest.mark.parametrize(
        "invalid_name",
        [
            "",
            "   ",
            "x" * 256,
        ],
    )
    def test_create_invalid_name_rejected(self, client: TestClient, auth_headers_a: dict, invalid_name: str):
        res = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": invalid_name},
        )
        assert res.status_code in (400, 422)

    def test_create_collection_race_condition_rollback(self, client: TestClient, auth_headers_a: dict, db_session: Session):
        """Simulates an IntegrityError race during commit to ensure clean 409 and rollback."""
        with patch.object(db_session, "commit", side_effect=IntegrityError("duplicate", {}, None)):
            res = client.post(
                "/api/v1/collections",
                headers=auth_headers_a,
                json={"name": "Race Collection"},
            )
            assert res.status_code == 409
            assert "already exists" in res.json()["detail"].lower()


class TestListCollections:
    """Tests for GET /api/v1/collections."""

    def test_list_empty_collections(self, client: TestClient, auth_headers_a: dict):
        res = client.get("/api/v1/collections", headers=auth_headers_a)
        assert res.status_code == 200
        data = res.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_list_collections_tenant_isolation(
        self, client: TestClient, auth_headers_a: dict, auth_headers_b: dict
    ):
        # User A creates 2 collections
        client.post("/api/v1/collections", headers=auth_headers_a, json={"name": "A1"})
        client.post("/api/v1/collections", headers=auth_headers_a, json={"name": "A2"})

        # User B creates 1 collection
        client.post("/api/v1/collections", headers=auth_headers_b, json={"name": "B1"})

        # User A only sees their 2 collections
        res_a = client.get("/api/v1/collections", headers=auth_headers_a)
        assert res_a.status_code == 200
        data_a = res_a.json()
        assert data_a["total"] == 2
        names_a = [item["name"] for item in data_a["items"]]
        assert sorted(names_a) == ["A1", "A2"]

        # User B only sees their 1 collection
        res_b = client.get("/api/v1/collections", headers=auth_headers_b)
        assert res_b.status_code == 200
        data_b = res_b.json()
        assert data_b["total"] == 1
        assert data_b["items"][0]["name"] == "B1"

    def test_list_collections_pagination(self, client: TestClient, auth_headers_a: dict):
        for i in range(5):
            client.post("/api/v1/collections", headers=auth_headers_a, json={"name": f"Col {i}"})

        # Fetch limit=2, offset=0
        res = client.get("/api/v1/collections?limit=2&offset=0", headers=auth_headers_a)
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 5
        assert len(data["items"]) == 2

        # Fetch limit=2, offset=4
        res2 = client.get("/api/v1/collections?limit=2&offset=4", headers=auth_headers_a)
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["total"] == 5
        assert len(data2["items"]) == 1


class TestGetCollection:
    """Tests for GET /api/v1/collections/{id}."""

    def test_get_collection_success(self, client: TestClient, auth_headers_a: dict):
        create_res = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": "My Papers", "description": "Notes"},
        )
        col_id = create_res.json()["id"]

        res = client.get(f"/api/v1/collections/{col_id}", headers=auth_headers_a)
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == col_id
        assert data["name"] == "My Papers"
        assert data["description"] == "Notes"
        assert data["document_count"] == 0
        assert "owner_id" not in data

    def test_get_nonexistent_collection_returns_404(self, client: TestClient, auth_headers_a: dict):
        res = client.get(f"/api/v1/collections/{uuid4()}", headers=auth_headers_a)
        assert res.status_code == 404
        assert res.json()["detail"] == "Collection not found."

    def test_get_other_user_collection_returns_404(
        self, client: TestClient, auth_headers_a: dict, auth_headers_b: dict
    ):
        # User A creates a collection
        create_res = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": "Secret A"},
        )
        col_id = create_res.json()["id"]

        # User B attempts to access it -> Must return 404, NOT 403 or 200
        res_b = client.get(f"/api/v1/collections/{col_id}", headers=auth_headers_b)
        assert res_b.status_code == 404
        assert res_b.json()["detail"] == "Collection not found."


class TestUpdateCollection:
    """Tests for PATCH /api/v1/collections/{id}."""

    def test_update_collection_name_and_description(self, client: TestClient, auth_headers_a: dict):
        create_res = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": "Old Name", "description": "Old Desc"},
        )
        col_id = create_res.json()["id"]

        update_res = client.patch(
            f"/api/v1/collections/{col_id}",
            headers=auth_headers_a,
            json={"name": "New Name", "description": "New Desc"},
        )
        assert update_res.status_code == 200
        data = update_res.json()
        assert data["name"] == "New Name"
        assert data["description"] == "New Desc"

    def test_update_collection_duplicate_name_returns_409(self, client: TestClient, auth_headers_a: dict):
        client.post("/api/v1/collections", headers=auth_headers_a, json={"name": "Existing"})
        c2 = client.post("/api/v1/collections", headers=auth_headers_a, json={"name": "To Rename"}).json()

        update_res = client.patch(
            f"/api/v1/collections/{c2['id']}",
            headers=auth_headers_a,
            json={"name": "Existing"},
        )
        assert update_res.status_code == 409
        assert "already exists" in update_res.json()["detail"].lower()

    def test_update_other_user_collection_returns_404(
        self, client: TestClient, auth_headers_a: dict, auth_headers_b: dict
    ):
        c_a = client.post("/api/v1/collections", headers=auth_headers_a, json={"name": "A's Collection"}).json()

        # User B attempts to update User A's collection
        res = client.patch(
            f"/api/v1/collections/{c_a['id']}",
            headers=auth_headers_b,
            json={"name": "Hacked Name"},
        )
        assert res.status_code == 404
        assert res.json()["detail"] == "Collection not found."

        # Verify User A's collection was not modified
        verify_res = client.get(f"/api/v1/collections/{c_a['id']}", headers=auth_headers_a)
        assert verify_res.json()["name"] == "A's Collection"

    def test_update_nonexistent_collection_returns_404(self, client: TestClient, auth_headers_a: dict):
        res = client.patch(
            f"/api/v1/collections/{uuid4()}",
            headers=auth_headers_a,
            json={"name": "Does Not Matter"},
        )
        assert res.status_code == 404

    def test_update_explicit_name_null_rejected(self, client: TestClient, auth_headers_a: dict):
        col = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": "Keep This Name"},
        ).json()

        # Explicit name: null must be rejected with 400 or 422
        res = client.patch(
            f"/api/v1/collections/{col['id']}",
            headers=auth_headers_a,
            json={"name": None},
        )
        assert res.status_code in (400, 422)

        # Verify collection name remained intact
        verify = client.get(f"/api/v1/collections/{col['id']}", headers=auth_headers_a).json()
        assert verify["name"] == "Keep This Name"

    def test_update_empty_patch_noop_preserves_timestamp(self, client: TestClient, auth_headers_a: dict):
        col = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": "Noop Target", "description": "Original Description"},
        ).json()
        orig_updated_at = col["updated_at"]

        # Empty PATCH is an idempotent no-op returning 200 without mutating timestamp
        res = client.patch(
            f"/api/v1/collections/{col['id']}",
            headers=auth_headers_a,
            json={},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["name"] == "Noop Target"
        assert data["description"] == "Original Description"
        assert data["updated_at"] == orig_updated_at

    def test_update_unchanged_values_preserves_timestamp(self, client: TestClient, auth_headers_a: dict):
        col = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": "Identical Values", "description": "Same Desc"},
        ).json()
        orig_updated_at = col["updated_at"]

        # Sending identical values avoids unnecessary database write and preserves updated_at
        res = client.patch(
            f"/api/v1/collections/{col['id']}",
            headers=auth_headers_a,
            json={"name": "Identical Values", "description": "Same Desc"},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["updated_at"] == orig_updated_at

    def test_update_collection_race_condition_rollback(
        self, client: TestClient, auth_headers_a: dict, db_session: Session
    ):
        """Simulates an IntegrityError race during update commit to verify 409 and clean rollback."""
        col = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": "Pre-race Name"},
        ).json()

        with patch.object(db_session, "commit", side_effect=IntegrityError("duplicate", {}, None)):
            res = client.patch(
                f"/api/v1/collections/{col['id']}",
                headers=auth_headers_a,
                json={"name": "Race Renamed"},
            )
            assert res.status_code == 409
            assert "already exists" in res.json()["detail"].lower()



class TestDeleteCollection:
    """Tests for DELETE /api/v1/collections/{id}."""

    def test_delete_collection_success(
        self, client: TestClient, auth_headers_a: dict, db_session: Session
    ):
        col = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": "To Delete"},
        ).json()

        del_res = client.delete(f"/api/v1/collections/{col['id']}", headers=auth_headers_a)
        assert del_res.status_code == 200
        assert del_res.json()["message"] == "Collection deleted successfully"

        # Verify not found after deletion
        get_res = client.get(f"/api/v1/collections/{col['id']}", headers=auth_headers_a)
        assert get_res.status_code == 404

        # Verify database record is gone
        from uuid import UUID
        db_col = db_session.query(Collection).filter(Collection.id == UUID(col["id"])).first()
        assert db_col is None

    def test_delete_other_user_collection_returns_404(
        self, client: TestClient, auth_headers_a: dict, auth_headers_b: dict, db_session: Session
    ):
        from uuid import UUID
        col_a = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": "User A Keeps This"},
        ).json()

        # User B attempts to delete User A's collection
        del_res = client.delete(f"/api/v1/collections/{col_a['id']}", headers=auth_headers_b)
        assert del_res.status_code == 404
        assert del_res.json()["detail"] == "Collection not found."

        # Verify collection still exists for User A
        db_col = db_session.query(Collection).filter(Collection.id == UUID(col_a["id"])).first()
        assert db_col is not None

    def test_delete_nonexistent_collection_returns_404(self, client: TestClient, auth_headers_a: dict):
        res = client.delete(f"/api/v1/collections/{uuid4()}", headers=auth_headers_a)
        assert res.status_code == 404

    def test_delete_collection_leaves_documents_intact(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        """Verifies deleting a collection deletes associations but preserves member documents."""
        from uuid import UUID
        # Create collection
        col_data = client.post(
            "/api/v1/collections",
            headers=auth_headers_a,
            json={"name": "Parent Collection"},
        ).json()
        col_uuid = UUID(col_data["id"])

        # Create Document in DB
        doc = Document(
            owner_id=user_a.id,
            filename="test_paper.pdf",
            file_hash_sha256="abc123hash",
            file_size_bytes=1024,
            file_path="uploads/test_paper.pdf",
        )
        db_session.add(doc)
        db_session.commit()

        # Associate document with collection
        assoc = CollectionDocument(collection_id=col_uuid, document_id=doc.id)
        db_session.add(assoc)
        db_session.commit()

        # Check document_count before delete
        get_res = client.get(f"/api/v1/collections/{col_data['id']}", headers=auth_headers_a)
        assert get_res.json()["document_count"] == 1

        # Delete collection
        del_res = client.delete(f"/api/v1/collections/{col_data['id']}", headers=auth_headers_a)
        assert del_res.status_code == 200

        # Association row is purged
        assoc_in_db = (
            db_session.query(CollectionDocument)
            .filter(CollectionDocument.collection_id == col_uuid)
            .first()
        )
        assert assoc_in_db is None

        # Document record remains intact
        doc_in_db = db_session.query(Document).filter(Document.id == doc.id).first()
        assert doc_in_db is not None
        assert doc_in_db.filename == "test_paper.pdf"


class TestCollectionsOpenAPISpecification:
    """Verifies that OpenAPI exposes collection routes and security requirements."""

    def test_openapi_schema_contains_collection_routes(self, client: TestClient):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        paths = schema.get("paths", {})

        assert "/api/v1/collections" in paths
        assert "post" in paths["/api/v1/collections"]
        assert "get" in paths["/api/v1/collections"]

        assert "/api/v1/collections/{id}" in paths
        assert "get" in paths["/api/v1/collections/{id}"]
        assert "patch" in paths["/api/v1/collections/{id}"]
        assert "delete" in paths["/api/v1/collections/{id}"]

        assert "/api/v1/collections/{id}/documents" in paths
        assert "post" in paths["/api/v1/collections/{id}/documents"]
        assert "get" in paths["/api/v1/collections/{id}/documents"]

        assert "/api/v1/collections/{id}/documents/{document_id}" in paths
        assert "delete" in paths["/api/v1/collections/{id}/documents/{document_id}"]

        # Check security on protected collection route
        get_col = paths["/api/v1/collections/{id}"]["get"]
        assert "security" in get_col
        assert any("HTTPBearer" in req for req in get_col["security"])


class TestAddDocumentsToCollection:
    """Verifies adding documents to a collection with tenant isolation, duplicate handling, and atomicity."""

    def test_add_single_document_by_document_id(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        # Create collection
        col = Collection(owner_id=user_a.id, name="Neuroscience Papers")
        db_session.add(col)
        # Create document
        doc = Document(
            owner_id=user_a.id,
            filename="paper1.pdf",
            file_hash_sha256="hash1",
            file_size_bytes=1000,
            file_path="uploads/paper1.pdf",
            processing_status="READY",
        )
        db_session.add(doc)
        db_session.commit()

        # Add using single document_id field
        res = client.post(
            f"/api/v1/collections/{col.id}/documents",
            headers=auth_headers_a,
            json={"document_id": str(doc.id)},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["id"] == str(col.id)
        assert data["document_count"] == 1

        # Check association in DB
        assoc = db_session.query(CollectionDocument).filter_by(collection_id=col.id, document_id=doc.id).first()
        assert assoc is not None

    def test_add_single_document_by_document_ids_list(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        col = Collection(owner_id=user_a.id, name="Biology Papers")
        doc = Document(
            owner_id=user_a.id,
            filename="bio.pdf",
            file_hash_sha256="hashbio",
            file_size_bytes=2000,
            file_path="uploads/bio.pdf",
            processing_status="READY",
        )
        db_session.add_all([col, doc])
        db_session.commit()

        res = client.post(
            f"/api/v1/collections/{col.id}/documents",
            headers=auth_headers_a,
            json={"document_ids": [str(doc.id)]},
        )
        assert res.status_code == 200
        assert res.json()["document_count"] == 1

    def test_add_multiple_documents_increments_count(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        col = Collection(owner_id=user_a.id, name="Multi Doc Col")
        docs = [
            Document(
                owner_id=user_a.id,
                filename=f"doc_{i}.pdf",
                file_hash_sha256=f"hash_{i}",
                file_size_bytes=100 * i,
                file_path=f"uploads/doc_{i}.pdf",
                processing_status="READY",
            )
            for i in range(1, 4)
        ]
        db_session.add(col)
        db_session.add_all(docs)
        db_session.commit()

        res = client.post(
            f"/api/v1/collections/{col.id}/documents",
            headers=auth_headers_a,
            json={"document_ids": [str(d.id) for d in docs]},
        )
        assert res.status_code == 200
        assert res.json()["document_count"] == 3

    def test_add_duplicate_document_is_idempotent(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        col = Collection(owner_id=user_a.id, name="Duplicate Test Col")
        doc1 = Document(
            owner_id=user_a.id,
            filename="d1.pdf",
            file_hash_sha256="hd1",
            file_size_bytes=500,
            file_path="uploads/d1.pdf",
            processing_status="READY",
        )
        doc2 = Document(
            owner_id=user_a.id,
            filename="d2.pdf",
            file_hash_sha256="hd2",
            file_size_bytes=600,
            file_path="uploads/d2.pdf",
            processing_status="READY",
        )
        db_session.add_all([col, doc1, doc2])
        db_session.commit()

        # Add doc1 first
        client.post(
            f"/api/v1/collections/{col.id}/documents",
            headers=auth_headers_a,
            json={"document_ids": [str(doc1.id)]},
        )

        # Add doc1 again along with doc2, plus doc1 repeated in payload
        res = client.post(
            f"/api/v1/collections/{col.id}/documents",
            headers=auth_headers_a,
            json={"document_ids": [str(doc1.id), str(doc2.id), str(doc1.id)]},
        )
        assert res.status_code == 200
        assert res.json()["document_count"] == 2

        # Verify exact row count in DB
        assoc_count = db_session.query(CollectionDocument).filter_by(collection_id=col.id).count()
        assert assoc_count == 2

    def test_add_document_to_nonexistent_collection_returns_404(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        doc = Document(
            owner_id=user_a.id,
            filename="d.pdf",
            file_hash_sha256="hd",
            file_size_bytes=500,
            file_path="uploads/d.pdf",
        )
        db_session.add(doc)
        db_session.commit()

        res = client.post(
            f"/api/v1/collections/{uuid4()}/documents",
            headers=auth_headers_a,
            json={"document_ids": [str(doc.id)]},
        )
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

    def test_add_document_to_other_user_collection_returns_404(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User, user_b: User
    ):
        # Collection owned by User B
        col_b = Collection(owner_id=user_b.id, name="User B Col")
        # Document owned by User A
        doc_a = Document(
            owner_id=user_a.id,
            filename="doc_a.pdf",
            file_hash_sha256="hda",
            file_size_bytes=500,
            file_path="uploads/doc_a.pdf",
        )
        db_session.add_all([col_b, doc_a])
        db_session.commit()

        # User A tries to add doc_a to User B's collection
        res = client.post(
            f"/api/v1/collections/{col_b.id}/documents",
            headers=auth_headers_a,
            json={"document_ids": [str(doc_a.id)]},
        )
        assert res.status_code == 404

    def test_add_nonexistent_document_returns_404(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        col = Collection(owner_id=user_a.id, name="Col Test")
        db_session.add(col)
        db_session.commit()

        res = client.post(
            f"/api/v1/collections/{col.id}/documents",
            headers=auth_headers_a,
            json={"document_ids": [str(uuid4())]},
        )
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

    def test_add_other_user_document_returns_404(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User, user_b: User
    ):
        col_a = Collection(owner_id=user_a.id, name="Col A")
        doc_b = Document(
            owner_id=user_b.id,
            filename="doc_b.pdf",
            file_hash_sha256="hdb",
            file_size_bytes=500,
            file_path="uploads/doc_b.pdf",
        )
        db_session.add_all([col_a, doc_b])
        db_session.commit()

        # User A tries to add User B's document to User A's collection
        res = client.post(
            f"/api/v1/collections/{col_a.id}/documents",
            headers=auth_headers_a,
            json={"document_ids": [str(doc_b.id)]},
        )
        assert res.status_code == 404
        assert "not found" in res.json()["detail"].lower()

        # Check no association created
        assoc = db_session.query(CollectionDocument).filter_by(collection_id=col_a.id).first()
        assert assoc is None

    def test_atomicity_partial_invalid_batch_adds_nothing(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User, user_b: User
    ):
        col_a = Collection(owner_id=user_a.id, name="Atomic Col")
        doc_valid = Document(
            owner_id=user_a.id,
            filename="valid.pdf",
            file_hash_sha256="hvalid",
            file_size_bytes=500,
            file_path="uploads/valid.pdf",
        )
        doc_other = Document(
            owner_id=user_b.id,
            filename="other.pdf",
            file_hash_sha256="hother",
            file_size_bytes=500,
            file_path="uploads/other.pdf",
        )
        db_session.add_all([col_a, doc_valid, doc_other])
        db_session.commit()

        # Send batch with 1 valid doc and 1 invalid doc (belongs to User B)
        res = client.post(
            f"/api/v1/collections/{col_a.id}/documents",
            headers=auth_headers_a,
            json={"document_ids": [str(doc_valid.id), str(doc_other.id)]},
        )
        assert res.status_code == 404

        # Crucial: verify that doc_valid was NOT added
        assoc = db_session.query(CollectionDocument).filter_by(collection_id=col_a.id).all()
        assert len(assoc) == 0

    def test_add_documents_in_various_processing_states(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        col = Collection(owner_id=user_a.id, name="States Col")
        doc_pending = Document(
            owner_id=user_a.id,
            filename="pending.pdf",
            file_hash_sha256="hpending",
            file_size_bytes=100,
            file_path="uploads/pending.pdf",
            processing_status="PENDING",
        )
        doc_extracting = Document(
            owner_id=user_a.id,
            filename="extracting.pdf",
            file_hash_sha256="hextract",
            file_size_bytes=200,
            file_path="uploads/ext.pdf",
            processing_status="EXTRACTING",
        )
        doc_ready = Document(
            owner_id=user_a.id,
            filename="ready.pdf",
            file_hash_sha256="hready",
            file_size_bytes=300,
            file_path="uploads/ready.pdf",
            processing_status="READY",
        )
        db_session.add_all([col, doc_pending, doc_extracting, doc_ready])
        db_session.commit()

        res = client.post(
            f"/api/v1/collections/{col.id}/documents",
            headers=auth_headers_a,
            json={"document_ids": [str(doc_pending.id), str(doc_extracting.id), str(doc_ready.id)]},
        )
        assert res.status_code == 200
        assert res.json()["document_count"] == 3


class TestRemoveDocumentFromCollection:
    """Verifies removing a document from a collection preserves the underlying document."""

    def test_remove_document_success_decrements_count(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        col = Collection(owner_id=user_a.id, name="Col To Remove")
        doc1 = Document(
            owner_id=user_a.id,
            filename="r1.pdf",
            file_hash_sha256="hr1",
            file_size_bytes=100,
            file_path="uploads/r1.pdf",
        )
        doc2 = Document(
            owner_id=user_a.id,
            filename="r2.pdf",
            file_hash_sha256="hr2",
            file_size_bytes=200,
            file_path="uploads/r2.pdf",
        )
        db_session.add_all([col, doc1, doc2])
        db_session.commit()

        assoc1 = CollectionDocument(collection_id=col.id, document_id=doc1.id)
        assoc2 = CollectionDocument(collection_id=col.id, document_id=doc2.id)
        db_session.add_all([assoc1, assoc2])
        db_session.commit()

        # Remove doc1
        res = client.delete(
            f"/api/v1/collections/{col.id}/documents/{doc1.id}",
            headers=auth_headers_a,
        )
        assert res.status_code == 200
        assert res.json()["document_count"] == 1

        # Check assoc1 is gone, assoc2 remains
        remaining = db_session.query(CollectionDocument).filter_by(collection_id=col.id).all()
        assert len(remaining) == 1
        assert remaining[0].document_id == doc2.id

    def test_remove_document_preserves_underlying_document(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        col = Collection(owner_id=user_a.id, name="Preserve Col")
        doc = Document(
            owner_id=user_a.id,
            filename="preserve.pdf",
            file_hash_sha256="hpres",
            file_size_bytes=100,
            file_path="uploads/pres.pdf",
        )
        db_session.add_all([col, doc])
        db_session.commit()

        db_session.add(CollectionDocument(collection_id=col.id, document_id=doc.id))
        db_session.commit()

        res = client.delete(
            f"/api/v1/collections/{col.id}/documents/{doc.id}",
            headers=auth_headers_a,
        )
        assert res.status_code == 200
        assert res.json()["document_count"] == 0

        # Verify underlying document record in DB is completely intact
        doc_in_db = db_session.query(Document).filter_by(id=doc.id).first()
        assert doc_in_db is not None
        assert doc_in_db.filename == "preserve.pdf"

    def test_remove_non_member_document_returns_404(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        col = Collection(owner_id=user_a.id, name="Empty Col")
        doc = Document(
            owner_id=user_a.id,
            filename="not_in_col.pdf",
            file_hash_sha256="hnotin",
            file_size_bytes=100,
            file_path="uploads/notin.pdf",
        )
        db_session.add_all([col, doc])
        db_session.commit()

        res = client.delete(
            f"/api/v1/collections/{col.id}/documents/{doc.id}",
            headers=auth_headers_a,
        )
        assert res.status_code == 404
        assert "not a member" in res.json()["detail"].lower()

    def test_remove_document_other_user_collection_returns_404(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User, user_b: User
    ):
        col_b = Collection(owner_id=user_b.id, name="User B Col")
        doc_b = Document(
            owner_id=user_b.id,
            filename="b.pdf",
            file_hash_sha256="hb",
            file_size_bytes=100,
            file_path="uploads/b.pdf",
        )
        db_session.add_all([col_b, doc_b])
        db_session.commit()
        db_session.add(CollectionDocument(collection_id=col_b.id, document_id=doc_b.id))
        db_session.commit()

        # User A attempts to delete from User B's collection
        res = client.delete(
            f"/api/v1/collections/{col_b.id}/documents/{doc_b.id}",
            headers=auth_headers_a,
        )
        assert res.status_code == 404


class TestListCollectionDocuments:
    """Verifies listing and pagination for member documents in a collection."""

    def test_list_documents_empty_collection(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        col = Collection(owner_id=user_a.id, name="Empty Col")
        db_session.add(col)
        db_session.commit()

        res = client.get(f"/api/v1/collections/{col.id}/documents", headers=auth_headers_a)
        assert res.status_code == 200
        data = res.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["limit"] == 50
        assert data["offset"] == 0

    def test_list_documents_returns_member_details_with_status(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        col = Collection(owner_id=user_a.id, name="Members Col")
        doc = Document(
            owner_id=user_a.id,
            filename="quantum_paper.pdf",
            title="Quantum Foundations",
            file_hash_sha256="hqf",
            file_size_bytes=4096,
            page_count=12,
            file_path="uploads/quantum_paper.pdf",
            processing_status="READY",
        )
        db_session.add_all([col, doc])
        db_session.commit()

        db_session.add(CollectionDocument(collection_id=col.id, document_id=doc.id))
        db_session.commit()

        res = client.get(f"/api/v1/collections/{col.id}/documents", headers=auth_headers_a)
        assert res.status_code == 200
        data = res.json()
        assert data["total"] == 1
        item = data["items"][0]
        assert item["id"] == str(doc.id)
        assert item["title"] == "Quantum Foundations"
        assert item["filename"] == "quantum_paper.pdf"
        assert item["page_count"] == 12
        assert item["file_size_bytes"] == 4096
        assert item["processing_status"] == "READY"

    def test_list_documents_pagination(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        col = Collection(owner_id=user_a.id, name="Paginated Col")
        docs = [
            Document(
                owner_id=user_a.id,
                filename=f"doc_{i}.pdf",
                file_hash_sha256=f"hash_pag_{i}",
                file_size_bytes=1000,
                file_path=f"uploads/doc_{i}.pdf",
            )
            for i in range(5)
        ]
        db_session.add(col)
        db_session.add_all(docs)
        db_session.commit()

        for d in docs:
            db_session.add(CollectionDocument(collection_id=col.id, document_id=d.id))
        db_session.commit()

        # Page 1: limit 2, offset 0
        p1 = client.get(f"/api/v1/collections/{col.id}/documents?limit=2&offset=0", headers=auth_headers_a)
        assert p1.status_code == 200
        d1 = p1.json()
        assert d1["total"] == 5
        assert len(d1["items"]) == 2

        # Page 2: limit 2, offset 2
        p2 = client.get(f"/api/v1/collections/{col.id}/documents?limit=2&offset=2", headers=auth_headers_a)
        assert p2.status_code == 200
        d2 = p2.json()
        assert d2["total"] == 5
        assert len(d2["items"]) == 2

        # Verify page 1 items differ from page 2 items
        ids_p1 = {item["id"] for item in d1["items"]}
        ids_p2 = {item["id"] for item in d2["items"]}
        assert ids_p1.isdisjoint(ids_p2)

    def test_list_documents_other_user_collection_returns_404(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_b: User
    ):
        col_b = Collection(owner_id=user_b.id, name="Col B")
        db_session.add(col_b)
        db_session.commit()

        res = client.get(f"/api/v1/collections/{col_b.id}/documents", headers=auth_headers_a)
        assert res.status_code == 404


class DummyCollectionTestEmbeddingService:
    def __init__(self, dimension: int = 4):
        self._dimension = dimension
        self.model_name = "dummy-embedder"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str):
        import numpy as np
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class TestCollectionScopedSearchAndChatAfterMembership:
    """Verifies collection-scoped search and chat before and after membership changes."""

    def test_collection_scoped_search_after_document_addition_and_removal(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        import numpy as np
        from app.api.v1.search import get_retrieval_service
        from app.models.chunk import Chunk
        from app.services.retrieval_service import RetrievalService
        from app.services.vector_store import VectorStore

        # 1. Setup isolated vector store and retrieval service
        vs = VectorStore(dimension=4, model_name="dummy-embedder")
        embed_svc = DummyCollectionTestEmbeddingService(dimension=4)
        retrieval_svc = RetrievalService(vector_store=vs, embedding_service=embed_svc)
        app.dependency_overrides[get_retrieval_service] = lambda: retrieval_svc

        try:
            # 2. Create document and chunk
            doc = Document(
                owner_id=user_a.id,
                filename="quantum_paper.pdf",
                title="Quantum Computing",
                file_hash_sha256="qchash123",
                file_size_bytes=1024,
                processing_status="READY",
                file_path="uploads/qc.pdf",
            )
            db_session.add(doc)
            db_session.commit()

            vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            chunk = Chunk(
                id=9991,
                document_id=doc.id,
                chunk_index=0,
                content="Quantum entanglement allows instantaneous state correlations.",
                page_number=1,
                char_start=0,
                char_end=60,
                token_count=10,
            )
            db_session.add(chunk)
            db_session.commit()
            vs.add_vectors(
                chunk_ids=[9991],
                vectors=vec.reshape(1, -1).astype(np.float32),
            )

            # 3. Create collection
            col_res = client.post(
                "/api/v1/collections",
                headers=auth_headers_a,
                json={"name": "Quantum Research"},
            )
            assert col_res.status_code == 201
            col_id = col_res.json()["id"]

            # 4. Search COLLECTION before adding doc -> empty results
            s_before = client.post(
                "/api/v1/search",
                headers=auth_headers_a,
                json={"query": "quantum entanglement", "scope_type": "COLLECTION", "scope_id": col_id},
            )
            assert s_before.status_code == 200
            assert s_before.json()["total_results"] == 0

            # 5. Add document to collection via API
            add_res = client.post(
                f"/api/v1/collections/{col_id}/documents",
                headers=auth_headers_a,
                json={"document_id": str(doc.id)},
            )
            assert add_res.status_code == 200
            assert add_res.json()["document_count"] == 1

            # 6. Search COLLECTION after adding doc -> finds the chunk
            s_after = client.post(
                "/api/v1/search",
                headers=auth_headers_a,
                json={"query": "quantum entanglement", "scope_type": "COLLECTION", "scope_id": col_id},
            )
            assert s_after.status_code == 200
            assert s_after.json()["total_results"] == 1
            assert s_after.json()["results"][0]["chunk_id"] == 9991

            # 7. Remove document from collection via API
            del_res = client.delete(
                f"/api/v1/collections/{col_id}/documents/{doc.id}",
                headers=auth_headers_a,
            )
            assert del_res.status_code == 200
            assert del_res.json()["document_count"] == 0

            # 8. Search COLLECTION after removing doc -> 0 results again
            s_removed = client.post(
                "/api/v1/search",
                headers=auth_headers_a,
                json={"query": "quantum entanglement", "scope_type": "COLLECTION", "scope_id": col_id},
            )
            assert s_removed.status_code == 200
            assert s_removed.json()["total_results"] == 0

            # 9. Verify GLOBAL search still finds the document
            s_global = client.post(
                "/api/v1/search",
                headers=auth_headers_a,
                json={"query": "quantum entanglement", "scope_type": "GLOBAL"},
            )
            assert s_global.status_code == 200
            assert s_global.json()["total_results"] == 1
        finally:
            app.dependency_overrides.pop(get_retrieval_service, None)

    def test_collection_scoped_chat_after_membership_changes(
        self, client: TestClient, auth_headers_a: dict, db_session: Session, user_a: User
    ):
        # 1. Create document and collection
        doc = Document(
            owner_id=user_a.id,
            filename="chat_paper.pdf",
            file_hash_sha256="chathash123",
            file_size_bytes=1024,
            processing_status="READY",
            file_path="uploads/chat_paper.pdf",
        )
        col = Collection(owner_id=user_a.id, name="Chat Col")
        db_session.add_all([doc, col])
        db_session.commit()

        # 2. Add document to collection
        add_res = client.post(
            f"/api/v1/collections/{col.id}/documents",
            headers=auth_headers_a,
            json={"document_ids": [str(doc.id)]},
        )
        assert add_res.status_code == 200

        # 3. Create chat session scoped to COLLECTION
        chat_res = client.post(
            "/api/v1/chat/sessions",
            headers=auth_headers_a,
            json={
                "title": "Quantum Discussion",
                "scope_type": "COLLECTION",
                "target_collection_id": str(col.id),
            },
        )
        assert chat_res.status_code == 201
        session_id = chat_res.json()["id"]

        # 4. Remove document from collection
        del_res = client.delete(
            f"/api/v1/collections/{col.id}/documents/{doc.id}",
            headers=auth_headers_a,
        )
        assert del_res.status_code == 200

        # 5. Existing chat session remains valid and accessible
        get_chat = client.get(f"/api/v1/chat/sessions/{session_id}", headers=auth_headers_a)
        assert get_chat.status_code == 200
        assert get_chat.json()["id"] == session_id
        assert get_chat.json()["target_collection_id"] == str(col.id)
