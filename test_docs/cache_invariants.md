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
- Each cache entry stores invariant metadata alongside Parquet pages. The JSON
  metadata records the normalised invariant tokens so TTL refreshes and cache
  reuse can reason about shard coverage.
- Stored `ttl_seconds` and `row_count` values update on every cache miss. Tests
  validate that superset hits do not trigger a refresh, while new shards write
  fresh metadata entries.

## Reusing and recombining shards
- Cache keys encode invariant parameters using the same normalised token order
  found in metadata. Two requests that differ only by token case hash to the
  same digest, whereas incompatible token sets route to distinct directories.
- When a cached entry contains a superset of invariant tokens, follow-up
  requests for subsets reuse the stored pages. The cache filters the superset
  down in-memory, so the DuckDB runner is not invoked.
- Requests that include tokens outside a cached superset result in misses. New
  shards are materialised under their own digest and persisted with the
  matching invariant metadata, ready for future recombination.

These requirements ensure that route-level invariants behave predictably and
that the cache can confidently combine, prune, or refresh shards without
violating query semantics.
