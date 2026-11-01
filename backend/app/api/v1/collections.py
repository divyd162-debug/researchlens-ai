"""API router for Collection management endpoints."""
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.models.user import User
from app.schemas.collection import (
    CollectionCreateRequest,
    CollectionListResponse,
    CollectionResponse,
    CollectionUpdateRequest,
    MessageResponse,
)
from app.services.collection_service import (
    CollectionNotFoundError,
    DuplicateCollectionNameError,
    InvalidCollectionNameError,
    collection_service,
)

router = APIRouter(prefix="/collections", tags=["Collections"])


@router.post(
    "",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new collection",
    description="Creates a research collection for the authenticated user with a unique name.",
)
def create_collection(
    request: CollectionCreateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CollectionResponse:
    """Creates a new collection owned by the current authenticated user."""
    try:
        return collection_service.create_collection(
            db=db,
            owner_id=current_user.id,
            request=request,
        )
    except DuplicateCollectionNameError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except InvalidCollectionNameError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.get(
    "",
    response_model=CollectionListResponse,
    status_code=status.HTTP_200_OK,
    summary="List collections",
    description="Returns a paginated list of collections owned by the authenticated caller.",
)
def list_collections(
    limit: int = Query(default=50, ge=1, le=100, description="Page limit"),
    offset: int = Query(default=0, ge=0, description="Page offset"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CollectionListResponse:
    """Lists collections owned exclusively by the authenticated caller."""
    return collection_service.list_collections(
        db=db,
        owner_id=current_user.id,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{id}",
    response_model=CollectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get collection by ID",
    description="Retrieves metadata and document count for a collection owned by the caller.",
)
def get_collection(
    id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CollectionResponse:
    """Retrieves a single collection by ID, strictly checking ownership."""
    try:
        return collection_service.get_collection(
            db=db,
            owner_id=current_user.id,
            collection_id=id,
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.patch(
    "/{id}",
    response_model=CollectionResponse,
    status_code=status.HTTP_200_OK,
    summary="Update collection",
    description="Updates name or description of an owned collection.",
)
def update_collection(
    id: UUID,
    request: CollectionUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CollectionResponse:
    """Updates a collection owned by the current authenticated user."""
    try:
        return collection_service.update_collection(
            db=db,
            owner_id=current_user.id,
            collection_id=id,
            request=request,
        )
    except CollectionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DuplicateCollectionNameError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except InvalidCollectionNameError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@router.delete(
    "/{id}",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete collection",
    description="Deletes a collection owned by the caller. Member documents remain intact.",
)
def delete_collection(
    id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MessageResponse:
    """Deletes a collection owned by the authenticated caller."""
    try:
        collection_service.delete_collection(
            db=db,
            owner_id=current_user.id,
            collection_id=id,
        )
        return MessageResponse(message="Collection deleted successfully")
    except CollectionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
