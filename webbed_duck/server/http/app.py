"""Minimal FastAPI application exposing cache metadata summaries."""

from __future__ import annotations

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request

from ..cache import CacheStorage, peek_metadata
from .models import CacheMetadataResponse


def create_cache_router(storage: CacheStorage) -> APIRouter:
    """Build a router that exposes cache metadata endpoints."""

    router = APIRouter()

    def get_storage() -> CacheStorage:
        return storage

    @router.get(
        "/cache/{digest}",
        name="cache-metadata",
        response_model=CacheMetadataResponse,
        summary="Inspect cached entry metadata without loading Parquet pages",
    )
    def read_cache_metadata(
        digest: str, request: Request, store: CacheStorage = Depends(get_storage)
    ) -> CacheMetadataResponse:
        summary = peek_metadata(store, digest)
        if summary is None:
            raise HTTPException(status_code=404, detail="Cache entry not found")
        template = _build_page_template(request, summary.digest)
        return CacheMetadataResponse.from_summary(summary, page_url_template=template)

    return router


def create_cache_app(storage: CacheStorage) -> FastAPI:
    """Create a standalone FastAPI application exposing cache metadata routes."""

    app = FastAPI()
    app.include_router(create_cache_router(storage))
    return app


def _build_page_template(request: Request, digest: str) -> str:
    base_url = str(request.url_for("cache-metadata", digest=digest)).rstrip("/")
    return f"{base_url}/pages/{{page}}?format={{format}}"


__all__ = ["create_cache_app", "create_cache_router"]
