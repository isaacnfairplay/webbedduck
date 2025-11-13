"""Pydantic response models for the cache HTTP API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..cache import CacheMetadataSummary, ResponseEnvelope


class PageTemplate(BaseModel):
    """Template descriptor for per-page downloads."""

    url: str = Field(..., description="URL template with {page} and {format} placeholders.")


class CacheMetadataResponse(BaseModel):
    """Serializable metadata summary mirroring the cache response envelope."""

    digest: str
    route_slug: str
    parameters: dict[str, Any]
    constants: dict[str, Any]
    row_count: int
    page_size: int
    page_count: int
    created_at: datetime
    expires_at: datetime
    from_cache: bool
    from_superset: bool
    formats: tuple[str, ...]
    requested_invariants: dict[str, tuple[str, ...]]
    cached_invariants: dict[str, tuple[str, ...]]
    page_template: PageTemplate

    @classmethod
    def from_summary(
        cls, summary: CacheMetadataSummary, *, page_url_template: str
    ) -> "CacheMetadataResponse":
        return cls(
            digest=summary.digest,
            route_slug=summary.route_slug,
            parameters=dict(summary.parameters),
            constants=dict(summary.constants),
            row_count=summary.row_count,
            page_size=summary.page_size,
            page_count=summary.page_count,
            created_at=summary.created_at,
            expires_at=summary.expires_at,
            from_cache=summary.from_cache,
            from_superset=summary.from_superset,
            formats=summary.formats,
            requested_invariants={
                name: tuple(tokens) for name, tokens in summary.requested_invariants.items()
            },
            cached_invariants={
                name: tuple(tokens) for name, tokens in summary.cached_invariants.items()
            },
            page_template=PageTemplate(url=page_url_template),
        )


class RouteExecutionRequest(BaseModel):
    """Request payload for executing a SQL route."""

    parameters: dict[str, object] = Field(default_factory=dict)
    constants: dict[str, object] = Field(default_factory=dict)


class RouteExecutionResponse(BaseModel):
    """JSON serialisable view over cache metadata returned for a route."""

    digest: str
    route_slug: str
    parameters: dict[str, object]
    constants: dict[str, object]
    row_count: int
    page_size: int
    page_count: int
    created_at: datetime | None
    expires_at: datetime | None
    from_cache: bool
    from_superset: bool
    formats: tuple[str, ...]
    requested_invariants: dict[str, tuple[str, ...]]
    cached_invariants: dict[str, tuple[str, ...]]
    page_template: PageTemplate

    @classmethod
    def from_envelope(
        cls,
        envelope: ResponseEnvelope,
        *,
        slug: str,
        digest: str,
        parameters: dict[str, object],
        constants: dict[str, object],
        page_url_template: str,
    ) -> "RouteExecutionResponse":
        return cls(
            digest=digest,
            route_slug=slug,
            parameters=parameters,
            constants=constants,
            row_count=envelope.row_count,
            page_size=envelope.page_size,
            page_count=envelope.page_count,
            created_at=envelope.created_at,
            expires_at=envelope.expires_at,
            from_cache=envelope.from_cache,
            from_superset=envelope.from_superset,
            formats=envelope.formats,
            requested_invariants={
                name: tuple(tokens)
                for name, tokens in envelope.requested_invariants.items()
            },
            cached_invariants={
                name: tuple(tokens)
                for name, tokens in envelope.cached_invariants.items()
            },
            page_template=PageTemplate(url=page_url_template),
        )


__all__ = [
    "CacheMetadataResponse",
    "PageTemplate",
    "RouteExecutionRequest",
    "RouteExecutionResponse",
]
