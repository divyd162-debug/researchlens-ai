"""API v1 router definitions."""
from fastapi import APIRouter
from app.api.v1.auth import router as auth_router
from app.api.v1.chat import router as chat_router
from app.api.v1.collections import router as collections_router
from app.api.v1.documents import router as documents_router
from app.api.v1.query import router as query_router
from app.api.v1.search import router as search_router

api_v1_router = APIRouter()
api_v1_router.include_router(auth_router)
api_v1_router.include_router(chat_router)
api_v1_router.include_router(collections_router)
api_v1_router.include_router(documents_router)
api_v1_router.include_router(query_router)
api_v1_router.include_router(search_router)

__all__ = ["api_v1_router", "chat_router", "query_router", "search_router"]


