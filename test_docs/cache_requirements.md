# Cache requirements derived from automated tests

The behaviors in this document are asserted by the cache test suite and are
therefore considered normative.

## Cache keys

- `test_cache_key_digest_is_deterministic_and_null_safe` fixes the
  representation of a cache key. Route slugs are produced by replacing `/` with
  `-`, parameters are normalised into deterministic tuples, and constants use
  the `__null__` sentinel whenever a caller supplies `None`. The digest is
  stable across input ordering changes.

## Parquet page layout

- `test_fetch_or_populate_writes_pages_enforces_ttl_and_filters_subsets`
  demonstrates that cached DuckDB query results are written beneath the route
  slug directory and sharded into sequential `page-0000.parquet` style files
  that respect the configured page size. A `metadata.json` file is emitted next
  to the pages and stores the page size together with the original key.

## Invariant filters and the `__null__` sentinel

- The same test validates that cache entries recorded with `__null__` constants
  behave as invariant-satisfying supersets. Requests that tighten those
  constants (for example, setting `region` to a concrete value while leaving
  `active` as `__null__`) are fulfilled without rerunning DuckDB; the Parquet
  pages are filtered in-memory and only matching rows are returned.

## Configuration overrides

- The cache obeys the supplied `CacheConfig` during `fetch_or_populate`. The
  tests explicitly set a 30-second TTL, a page size of 2, and a temporary
  storage root directory to verify entry expiry, sharding, and isolation.

## Dependency injection expectations

- The DuckDB runner is passed into `fetch_or_populate` and is only invoked on
  cache misses or expired entries, which allows tests to stub the runner and
  assert call counts. Filesystem access is likewise driven by the configured
  storage root so that fixtures such as `tmp_path` can isolate IO.

