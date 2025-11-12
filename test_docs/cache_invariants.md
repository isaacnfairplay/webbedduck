# Cache invariant expectations

The invariant-aware cache combines deterministic cache key construction with
runtime predicate pruning to avoid unnecessary reruns of expensive DuckDB
queries. Tests under `tests/server/test_cache.py` assert the following
behaviour:

## Normalising invariant tokens
- Invariant inputs honour the `__null__` sentinel. When callers pass either
  `None` or `"__null__"`, the invariant layer interprets the token as a real
  `NULL` value and ensures only rows with database nulls are returned.
- Multi-value invariants split incoming strings with the configured separator
  (default `|`). Tokens are trimmed, deduplicated, lowercased when requested,
  and sorted to provide deterministic cache keys.
- Case-insensitive filters lower tokens as well as column values before
  comparison, ensuring values like `VIP`, `vip`, or `Vip` hit the same shard.

## Persisted metadata
- Cache entries now materialise Parquet pages without applying invariant
  filters. The JSON metadata therefore records empty `invariants` maps and the
  unfiltered `row_count` so callers know the stored dataset is a broad superset
  of any particular invariant request.
- Stored `ttl_seconds` values still refresh on cache misses. Because the same
  entry services every invariant combination, subsequent requests reuse the
  existing metadata until the TTL expires.

## Shared base entries
- Cache key digests intentionally exclude invariant constants. Two requests
  that differ only by invariant tokens hash to the same directory, guaranteeing
  they share the same stored Parquet pages.
- Each request receives a filtered view of that base table at read time. The
  response envelope surfaces the requested invariant tokens, while the cached
  invariant map stays empty to reflect that the on-disk pages were persisted
  without filtering.
- Because the base entry already contains every invariant value, the DuckDB
  runner executes only once per parameter + non-invariant constant combination.

These requirements ensure invariant filters behave predictably while avoiding
the complexity of superset detection or shard recombination.
