# Branch Changelog

## Refactor cache response contract
- Replace the Arrow-centric `CacheResult` payload with a `ResponseEnvelope` and `DataHandle` that advertise available formats and page counts.
- Persist configured page sizes alongside cached metadata so clients can request specific pages across formats.
- Extend the cache tests to exercise CSV and JSON exports from the new handle and assert stored metadata includes page sizing.
- Export the new types from `webbed_duck.server` and document the updated behaviour in the cache requirements guide.
- Tidy the templating range guard validation to reduce branching while preserving its error messages.
