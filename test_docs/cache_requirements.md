# Cache requirements derived from tests

The cache integration tests in `tests/server/test_cache.py` outline the expected behaviour of the DuckDB-backed cache layer:

## Cache key construction
- Keys are built from a route slug plus normalized parameter and constant dictionaries (`test_cache_key_deterministic_and_null_sentinel`).
- Parameter/constant pairs are sorted alphabetically, converted to strings, and use the `__null__` sentinel whenever a value is missing or explicitly `None`.
- The normalized structure feeds a deterministic digest so that equivalent inputs in different orders produce the same cache directory name.

## Parquet page layout
- Cache entries materialize under a directory named with the key digest (`test_fetch_or_populate_persists_pages_and_enforces_ttl`).
- Result tables are chunked into fixed-size Parquet pages (`page-*.parquet`), respecting the configured `page_size`.
- A JSON metadata file accompanies the pages and records row counts and other bookkeeping fields.

## Invariant filtering semantics
- Requests specify invariant constants that are re-applied after reading cached pages (`test_invariant_filters_and_null_semantics`).
- Filters support the `__null__` sentinel, returning only rows whose column is null when requested.
- Combining invariants narrows the cached data set while leaving other columns intact, ensuring the cache can satisfy subset-style requests without rerunning DuckDB.

## Configuration overrides
- `CacheConfig` accepts overrides for storage root, TTL, and page size, all of which are respected during fetch/populate cycles (`test_fetch_or_populate_persists_pages_and_enforces_ttl`).
- TTL enforcement relies on a dependency-injected clock so tests can advance time deterministically.

## Dependency-injected IO
- The DuckDB runner and clock used for TTL checking are both injected, letting tests stub them and assert cache hits bypass compute work (`test_fetch_or_populate_persists_pages_and_enforces_ttl`, `test_invariant_filters_and_null_semantics`).
- Storage is rooted at the configured path, making it trivial to sandbox cache IO in temporary directories during tests.

## Result metadata contract
- `Cache.fetch_or_populate` returns an immutable `ResponseEnvelope` wrapper that preserves Arrow ergonomics (`to_pylist`, column access) while exposing cache metadata and format-agnostic access (`test_fetch_or_populate_persists_pages_and_enforces_ttl`).
- Callers can observe whether data came from disk or a fresh run via `from_cache` and `from_superset`, along with the serving entry digest and filtered row counts (`test_multi_value_invariant_superset_reuse_and_metadata`).
- Requested invariant tokens and the backing cache entry's invariant set are surfaced as read-only mappings so routing layers can reason about superset reuse without touching on-disk JSON (`test_invariant_filters_and_null_semantics`, `test_case_insensitive_invariant_tokens`, `test_numeric_invariant_tokens_apply_column_type`).
