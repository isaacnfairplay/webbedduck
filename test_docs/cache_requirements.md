# Cache requirements

The cache behaviour is validated by `tests/server/test_cache.py` and is summarised below.

## Cache key construction

* Cache keys combine the route slug, request parameters, and invariant constants.
* Constants that are explicitly `None` are recorded using the `__null__` sentinel to keep `None` distinct from an omitted constant.
* Keys are normalised by sorting parameters and constants before hashing, so structurally equivalent requests share the same digest.

## Parquet page layout

* Cached result sets are written beneath a deterministic directory derived from the cache key digest.
* Each cache materialisation is paged into sequential Parquet files named `page-<index>.parquet`.
* The configured `page_size` bounds the number of rows per Parquet page; extra rows create additional page files.

## Invariant filter semantics

* Invariant constants act as row filters when reading from the cache.
* Filters accept `__null__` to request `NULL` values in the resulting table.
* Subset requests reuse cached pages and apply the invariant filters at read time via PyArrow compute expressions.

## Configuration overrides

* The cache honours `ttl_seconds`, `page_size`, and `storage_root` values supplied via `CacheConfig`.
* TTL expiry is computed relative to the injected clock; stale entries are repopulated via the DuckDB runner.
* Cache directories are created under the configured storage root, allowing callers to redirect cache data for testing or production.

## Dependency-injected IO

* A monotonic clock can be injected to make TTL behaviour deterministic under test.
* DuckDB runners are supplied per call and are only invoked on cache misses.
* Parquet IO uses dependency injection so that read and write helpers can be replaced for testing environments.
