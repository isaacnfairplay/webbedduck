# Branch Changelog

## Refactor cache response contract
- Replace the Arrow-centric `CacheResult` payload with a `ResponseEnvelope` and `DataHandle` that advertise available formats and page counts.
- Persist configured page sizes alongside cached metadata so clients can request specific pages across formats.
- Extend the cache tests to exercise CSV and JSON exports from the new handle and assert stored metadata includes page sizing.
- Export the new types from `webbed_duck.server` and document the updated behaviour in the cache requirements guide.
- Tidy the templating range guard validation to reduce branching while preserving its error messages.

## Follow-up: adapter polish and paging contract
- Restore zero-based page selection for Arrow-friendly ergonomics and tighten paging validation inside the data handle.
- Introduce dedicated format adapters to encapsulate Arrow, Parquet, CSV, and JSON encoding concerns and expose their names via the response envelope.
- Document the 0-indexed paging contract in the cache requirements guide and expand cache tests to cover the revised paging semantics.
