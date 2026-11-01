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

        # Check security on protected collection route
        get_col = paths["/api/v1/collections/{id}"]["get"]
        assert "security" in get_col
        assert any("HTTPBearer" in req for req in get_col["security"])


