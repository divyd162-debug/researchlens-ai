"""Service layer for Collection management and multi-tenant isolation."""
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.chat import ChatSession
from app.models.collection import Collection, CollectionDocument
from app.models.document import Document
from app.schemas.collection import (
    AddDocumentsToCollectionRequest,
    CollectionCreateRequest,
    CollectionListResponse,
    CollectionResponse,
    CollectionUpdateRequest,
)
from app.schemas.document import DocumentListResponse, DocumentResponse


class CollectionServiceError(Exception):
    """Base exception for collection service failures."""
    pass


class CollectionNotFoundError(CollectionServiceError):
    """Raised when a collection does not exist or does not belong to the user."""
    pass


class DuplicateCollectionNameError(CollectionServiceError):
    """Raised when a collection name already exists for the owner."""
    pass


class InvalidCollectionNameError(CollectionServiceError):
    """Raised when a collection name fails validation rules."""
    pass


class DocumentNotFoundError(CollectionServiceError):
    """Raised when a document does not exist or does not belong to the user."""
    pass


class AssociationNotFoundError(CollectionServiceError):
    """Raised when a document is not associated with the collection."""
    pass


class CollectionService:
    """Provides multi-tenant isolated collection management operations."""

    def create_collection(
        self,
        db: Session,
        owner_id: UUID,
        request: Union[CollectionCreateRequest, Dict[str, Any]],
    ) -> CollectionResponse:
        """Creates a new collection owned by owner_id.

        Args:
            db: Active database session.
            owner_id: Authenticated user ID (tenant owner).
            request: CollectionCreateRequest or payload dictionary.

        Returns:
            CollectionResponse for the created collection.

        Raises:
            InvalidCollectionNameError: If collection name is invalid.
            DuplicateCollectionNameError: If a collection with this name already exists for the user.
        """
        if isinstance(request, dict):
            try:
                req = CollectionCreateRequest.model_validate(request)
            except ValidationError as exc:
                for err in exc.errors():
                    loc = err.get("loc", ())
                    if "name" in loc:
                        raise InvalidCollectionNameError(f"Invalid collection name: {err.get('msg')}") from exc
                raise InvalidCollectionNameError(f"Validation failed: {exc}") from exc
        elif isinstance(request, CollectionCreateRequest):
            req = request
        else:
            raise ValueError(f"Expected CollectionCreateRequest or dict, got {type(request).__name__}")

        # Check existing collection name for this owner
        existing = (
            db.query(Collection)
            .filter(Collection.owner_id == owner_id, Collection.name == req.name)
            .first()
        )
        if existing is not None:
            raise DuplicateCollectionNameError(f"A collection named '{req.name}' already exists.")

        collection = Collection(
            owner_id=owner_id,
            name=req.name,
            description=req.description,
        )
        db.add(collection)

        try:
            db.commit()
            db.refresh(collection)
        except IntegrityError as exc:
            db.rollback()
            raise DuplicateCollectionNameError(f"A collection named '{req.name}' already exists.") from exc
        except Exception:
            db.rollback()
            raise

        return CollectionResponse(
            id=collection.id,
            name=collection.name,
            description=collection.description,
            document_count=0,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
        )

    def list_collections(
        self,
        db: Session,
        owner_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> CollectionListResponse:
        """Lists collections owned by owner_id with pagination.

        Args:
            db: Active database session.
            owner_id: Authenticated user ID.
            limit: Maximum items to return (1-100).
            offset: Items to skip.

        Returns:
            CollectionListResponse containing page items, total count, limit, and offset.
        """
        base_query = db.query(Collection).filter(Collection.owner_id == owner_id)
        total = base_query.count()

        collections = (
            base_query.order_by(Collection.created_at.desc(), Collection.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )


        # Batch count associated documents for all collections on this page
        col_ids = [c.id for c in collections]
        doc_counts: Dict[UUID, int] = {}
        if col_ids:
            counts = (
                db.query(CollectionDocument.collection_id, func.count(CollectionDocument.document_id))
                .filter(CollectionDocument.collection_id.in_(col_ids))
                .group_by(CollectionDocument.collection_id)
                .all()
            )
            doc_counts = {cid: cnt for cid, cnt in counts}

        items = [
            CollectionResponse(
                id=c.id,
                name=c.name,
                description=c.description,
                document_count=doc_counts.get(c.id, 0),
                created_at=c.created_at,
                updated_at=c.updated_at,
            )
            for c in collections
        ]

        return CollectionListResponse(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )

    def get_collection(
        self,
        db: Session,
        owner_id: UUID,
        collection_id: UUID,
    ) -> CollectionResponse:
        """Retrieves a single collection by ID, strictly scoped to owner_id.

        Args:
            db: Active database session.
            owner_id: Authenticated user ID.
            collection_id: Collection identifier.

        Returns:
            CollectionResponse if found and owned by owner_id.

        Raises:
            CollectionNotFoundError: If collection does not exist or belongs to another user.
        """
        collection = (
            db.query(Collection)
            .filter(Collection.owner_id == owner_id, Collection.id == collection_id)
            .first()
        )
        if collection is None:
            raise CollectionNotFoundError("Collection not found.")

        doc_count = (
            db.query(func.count(CollectionDocument.document_id))
            .filter(CollectionDocument.collection_id == collection.id)
            .scalar()
            or 0
        )

        return CollectionResponse(
            id=collection.id,
            name=collection.name,
            description=collection.description,
            document_count=doc_count,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
        )

    def update_collection(
        self,
        db: Session,
        owner_id: UUID,
        collection_id: UUID,
        request: Union[CollectionUpdateRequest, Dict[str, Any]],
    ) -> CollectionResponse:
        """Updates an existing collection's name and/or description.

        Args:
            db: Active database session.
            owner_id: Authenticated user ID.
            collection_id: Collection identifier.
            request: CollectionUpdateRequest or payload dictionary.

        Returns:
            Updated CollectionResponse.

        Raises:
            CollectionNotFoundError: If collection does not exist or belongs to another user.
            DuplicateCollectionNameError: If the new name is already taken by the user.
            InvalidCollectionNameError: If the new name is invalid.
        """
        if isinstance(request, dict):
            try:
                req = CollectionUpdateRequest.model_validate(request)
            except ValidationError as exc:
                for err in exc.errors():
                    loc = err.get("loc", ())
                    if "name" in loc:
                        raise InvalidCollectionNameError(f"Invalid collection name: {err.get('msg')}") from exc
                raise InvalidCollectionNameError(f"Validation failed: {exc}") from exc
        elif isinstance(request, CollectionUpdateRequest):
            req = request
        else:
            raise ValueError(f"Expected CollectionUpdateRequest or dict, got {type(request).__name__}")

        collection = (
            db.query(Collection)
            .filter(Collection.owner_id == owner_id, Collection.id == collection_id)
            .first()
        )
        if collection is None:
            raise CollectionNotFoundError("Collection not found.")

        # Track whether any attribute actually changed to avoid unnecessary commits
        # and unprompted updated_at timestamp updates on empty {} or unchanged payloads.
        modified = False

        # Update name if provided and different
        if req.name is not None and req.name != collection.name:
            existing = (
                db.query(Collection)
                .filter(
                    Collection.owner_id == owner_id,
                    Collection.name == req.name,
                    Collection.id != collection_id,
                )
                .first()
            )
            if existing is not None:
                raise DuplicateCollectionNameError(f"A collection named '{req.name}' already exists.")
            collection.name = req.name
            modified = True

        # Update description if provided in request and different
        if "description" in req.model_fields_set and req.description != collection.description:
            collection.description = req.description
            modified = True

        if modified:
            try:
                db.commit()
                db.refresh(collection)
            except IntegrityError as exc:
                db.rollback()
                raise DuplicateCollectionNameError(f"A collection named '{collection.name}' already exists.") from exc
            except Exception:
                db.rollback()
                raise


        doc_count = (
            db.query(func.count(CollectionDocument.document_id))
            .filter(CollectionDocument.collection_id == collection.id)
            .scalar()
            or 0
        )

        return CollectionResponse(
            id=collection.id,
            name=collection.name,
            description=collection.description,
            document_count=doc_count,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
        )

    def delete_collection(
        self,
        db: Session,
        owner_id: UUID,
        collection_id: UUID,
    ) -> None:
        """Deletes a collection owned by owner_id.

        Cascade delete purges collection_documents associations, leaving underlying
        Document records intact.

        Args:
            db: Active database session.
            owner_id: Authenticated user ID.
            collection_id: Collection identifier.

        Raises:
            CollectionNotFoundError: If collection does not exist or belongs to another user.
        """
        collection = (
            db.query(Collection)
            .filter(Collection.owner_id == owner_id, Collection.id == collection_id)
            .first()
        )
        if collection is None:
            raise CollectionNotFoundError("Collection not found.")

        # Safely archive any active chat sessions referencing this collection before deletion
        # to satisfy ck_chat_session_scope_targets check constraint.
        active_sessions = (
            db.query(ChatSession)
            .filter(
                ChatSession.target_collection_id == collection.id,
                ChatSession.is_archived.is_(False),
            )
            .all()
        )
        for session in active_sessions:
            session.archive()

        db.delete(collection)
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

    def add_documents_to_collection(
        self,
        db: Session,
        owner_id: UUID,
        collection_id: UUID,
        request: Union[AddDocumentsToCollectionRequest, Dict[str, Any]],
    ) -> CollectionResponse:
        """Associates one or more documents owned by owner_id with collection_id.

        Atomicity & Validation:
        - Collection must exist and be owned by owner_id.
        - Every requested document must exist and be owned by owner_id.
        - If any document is not found or not owned by the caller, the operation aborts
          cleanly with DocumentNotFoundError without making partial associations.
        - Duplicate memberships are safely and idempotently handled.
        - Documents in any processing state (e.g. PENDING, PROCESSING, READY) can be associated
          for organization; retrieval strictly queries only chunks from READY documents.

        Returns:
            CollectionResponse with updated document_count.
        """
        if isinstance(request, dict):
            req = AddDocumentsToCollectionRequest.model_validate(request)
        elif isinstance(request, AddDocumentsToCollectionRequest):
            req = request
        else:
            raise ValueError(f"Expected AddDocumentsToCollectionRequest or dict, got {type(request).__name__}")

        collection = (
            db.query(Collection)
            .filter(Collection.id == collection_id, Collection.owner_id == owner_id)
            .first()
        )
        if collection is None:
            raise CollectionNotFoundError("Collection not found.")

        unique_doc_ids = list(dict.fromkeys(req.document_ids))
        if not unique_doc_ids:
            raise ValueError("At least one document ID must be provided.")

        # Verify all documents exist and belong to the authenticated owner
        found_docs = (
            db.query(Document)
            .filter(Document.id.in_(unique_doc_ids), Document.owner_id == owner_id)
            .all()
        )
        found_doc_ids = {d.id for d in found_docs}
        if len(found_doc_ids) != len(unique_doc_ids):
            raise DocumentNotFoundError("One or more documents not found.")

        # Find existing associations to prevent duplicate primary key insertion
        existing_assocs = (
            db.query(CollectionDocument.document_id)
            .filter(
                CollectionDocument.collection_id == collection.id,
                CollectionDocument.document_id.in_(unique_doc_ids),
            )
            .all()
        )
        already_associated_ids = {row[0] for row in existing_assocs}
        new_ids_to_add = [did for did in unique_doc_ids if did not in already_associated_ids]

        if new_ids_to_add:
            for did in new_ids_to_add:
                db.add(CollectionDocument(collection_id=collection.id, document_id=did))
            try:
                db.commit()
            except IntegrityError:
                db.rollback()
            except Exception:
                db.rollback()
                raise

        doc_count = (
            db.query(func.count(CollectionDocument.document_id))
            .filter(CollectionDocument.collection_id == collection.id)
            .scalar()
            or 0
        )
        db.refresh(collection)

        return CollectionResponse(
            id=collection.id,
            name=collection.name,
            description=collection.description,
            document_count=doc_count,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
        )

    def remove_document_from_collection(
        self,
        db: Session,
        owner_id: UUID,
        collection_id: UUID,
        document_id: UUID,
    ) -> CollectionResponse:
        """Removes a document association from a collection without deleting the underlying document.

        Verification:
        - Collection must exist and be owned by owner_id.
        - Document must exist and be owned by owner_id.
        - Association between collection and document must exist.

        Returns:
            CollectionResponse with updated document_count.
        """
        collection = (
            db.query(Collection)
            .filter(Collection.id == collection_id, Collection.owner_id == owner_id)
            .first()
        )
        if collection is None:
            raise CollectionNotFoundError("Collection not found.")

        document = (
            db.query(Document)
            .filter(Document.id == document_id, Document.owner_id == owner_id)
            .first()
        )
        if document is None:
            raise DocumentNotFoundError("Document not found.")

        assoc = (
            db.query(CollectionDocument)
            .filter(
                CollectionDocument.collection_id == collection.id,
                CollectionDocument.document_id == document.id,
            )
            .first()
        )
        if assoc is None:
            raise AssociationNotFoundError("Document is not a member of this collection.")

        db.delete(assoc)
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise

        doc_count = (
            db.query(func.count(CollectionDocument.document_id))
            .filter(CollectionDocument.collection_id == collection.id)
            .scalar()
            or 0
        )
        db.refresh(collection)

        return CollectionResponse(
            id=collection.id,
            name=collection.name,
            description=collection.description,
            document_count=doc_count,
            created_at=collection.created_at,
            updated_at=collection.updated_at,
        )

    def list_collection_documents(
        self,
        db: Session,
        owner_id: UUID,
        collection_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> DocumentListResponse:
        """Lists member documents of a collection owned by owner_id with pagination.

        Verification:
        - Collection must exist and be owned by owner_id.

        Returns:
            DocumentListResponse with items, total count, limit, and offset.
        """
        collection = (
            db.query(Collection)
            .filter(Collection.id == collection_id, Collection.owner_id == owner_id)
            .first()
        )
        if collection is None:
            raise CollectionNotFoundError("Collection not found.")

        base_query = (
            db.query(Document)
            .join(CollectionDocument, CollectionDocument.document_id == Document.id)
            .filter(
                CollectionDocument.collection_id == collection.id,
                Document.owner_id == owner_id,
            )
        )
        total = base_query.count()
        documents = (
            base_query.order_by(CollectionDocument.added_at.desc(), Document.id.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        items = [DocumentResponse.model_validate(doc) for doc in documents]
        return DocumentListResponse(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )


# Singleton instance
collection_service = CollectionService()
