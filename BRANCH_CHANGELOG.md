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

## Follow-up: streaming exports and Arrow ergonomics
- Serve Parquet requests by returning the cached page files directly and stream CSV/JSON/JSONL encodings batch-by-batch, adding `jsonl` to the advertised formats.
- Encourage `ResponseEnvelope.as_arrow(page=...)` for PyArrow access, keep `ResponseEnvelope.open(...)` as a documented alias, and update docs/tests to rely on `envelope.data.open(...)` for file-backed formats.
- Clarify `DataHandle.open` return types in docstrings and extend cache tests to assert JSONL output plus on-disk Parquet reuse.
- Extract range guard preflight validation into a helper to reduce cyclomatic complexity while preserving existing error messaging.

## Complexity-focused streaming refactors
- Extract a shared `_TextAdapter` to manage CSV, JSON, and JSONL text streaming with one context manager.
- Drive JSON and JSONL exports through a single `_iter_json_payloads` generator wired via partial-based dispatch.
- Restructure `_IterTextIO.read` to use deterministic buffering loops, lowering cyclomatic complexity while preserving behaviour.
- Teach the complexity workflow to refresh comments in place and annotate reports with applied refactor patterns.

## Complexity reduction: guard helpers and metadata decoding
- Introduce `_apply_guard_mapping` to centralise guard retrieval, trimming branch-heavy validators for length, range, datetime, and compare guards.
- Streamline choice and regex guards with concise truthiness checks while preserving existing error messages.
- Share Parquet metadata coercion through `_decode_entry_metadata` and tighten cache iteration helpers to cut branches and lines of code.
- Regenerate complexity reports and metrics to capture the reduced cyclomatic totals and maintainability gains.

