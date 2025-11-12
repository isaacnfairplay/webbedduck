"""Server utilities for DuckDB-backed caching."""

from .cache import Cache, CacheConfig, CacheKey, CacheResult, DataHandle, ResponseEnvelope
from .template_metadata import (
    RouteDescription,
    build_route_registry,
    collect_template_metadata,
)

__all__ = [
    "Cache",
    "CacheConfig",
    "CacheKey",
    "CacheResult",
    "DataHandle",
    "ResponseEnvelope",
    "RouteDescription",
    "build_route_registry",
    "collect_template_metadata",
]
