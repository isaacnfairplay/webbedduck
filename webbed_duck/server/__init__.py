"""Server utilities for DuckDB-backed caching."""

from .cache import (
    Cache,
    CacheConfig,
    CacheKey,
    CacheMetadataSummary,
    CacheResult,
    DataHandle,
    ResponseEnvelope,
    peek_metadata,
)
from .template_metadata import (
    RouteDescription,
    build_route_registry,
    collect_template_metadata,
)

__all__ = [
    "Cache",
    "CacheConfig",
    "CacheKey",
    "CacheMetadataSummary",
    "CacheResult",
    "DataHandle",
    "ResponseEnvelope",
    "peek_metadata",
    "RouteDescription",
    "build_route_registry",
    "collect_template_metadata",
]
