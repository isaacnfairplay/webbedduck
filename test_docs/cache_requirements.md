# Cache requirements derived from tests/server/test_cache.py

- **Cache keys** are built from the route slug plus normalized parameter and constant mappings. `None` values are normalized to the `"__null__"` sentinel so identical logical requests share the same digest. *(See `test_cache_key_includes_null_sentinel`.)*
- **Page storage** writes each cached result beneath `<storage_root>/<cache_digest>/` as `page-*.parquet` files sized by the configured `page_size`. *(See `test_fetch_or_populate_round_trip` asserting page counts.)*
- **Invariant filters** are applied after loading cached pages, allowing subset requests such as parity-specific slices without re-running DuckDB. The filter mapping supports the `"__null__"` sentinel for nullable predicates. *(See the even/odd assertions in `test_fetch_or_populate_round_trip`.)*
- **Configuration overrides** are provided via `CacheConfig`, which accepts explicit TTLs, page sizes, and storage roots. TTL expiry forces a cache miss when exceeded. *(See the clock manipulation in `test_fetch_or_populate_round_trip`.)*
- **Dependency injection**: the cache accepts injected clocks and DuckDB runners so tests can stub execution and observe cache-hit behavior without hitting DuckDB on repeated reads. *(See the `runner` closure and `FakeClock` helper in `test_fetch_or_populate_round_trip`.)*
