# Branch Changelog

## Follow-up: HTTP route execution complexity trims
- Extract `_resolve_route_request`, `_ensure_supported_format`, and chunk-specific helpers inside `webbed_duck/server/http/app.py` so parameter binding, cache key creation, and streaming payloads each have focused, low-branch responsibilities.
- Rework `_stream_handle` to reuse the new helpers and cut duplicate read loops while keeping Arrow, CSV/JSON, and Parquet behaviour unchanged.
- Capture the complexity report outputs for the HTTP stack to document how the reorganised helpers keep cyclomatic scores for the FastAPI surface in check.

## Complexity reduction: directive guard dispatch
- Replace the branching-heavy `_directives_for_spec` guard handling with a data-driven registry so each guard declares a concise builder.
- Add focused helpers for choices, regex, mapping-based, and compare guards to cut duplicated filtering logic while preserving option semantics.
- Document the selection approach in `lessons_learned/refactor_selection.md` so future refactors start from hotspot data instead of intuition.

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

## Complexity reduction: guard registry polish and SLOC cuts
- Replace ad-hoc guard iteration with `_SIMPLE_GUARDS`/`_MAPPING_GUARDS` dispatching, eliminating bespoke wrappers while keeping error messages intact.
- Inline boolean coercion, datetime boundary formatting, and range preflight checks to drop redundant helpers and cut non-space SLOC.
- Hoist a reusable `_TemplateParameterNamespace` class so template consumption tracking no longer redefines per-call classes.
- Refresh complexity reports after the SLOC-focused refactor to document the net reductions.
## Complexity reduction: coercion dispatch and format helpers
- Centralise parameter coercion behind `_TYPE_COERCERS`, cutting branching in `_coerce_value` while sharing boolean parsing across contexts.
- Simplify datetime guard checks by reusing the original boundary helper and tightening formatting branches for clearer error messaging.
- Introduce `_require_format_definition` so date, timestamp, and number formatters reuse the same guard logic instead of duplicating checks.
- Refresh complexity reports to capture the lower cyclomatic totals, reduced non-space SLOC, and improved maintainability index.

## Complexity reduction: cache fetch orchestration
- Extract `_resolve_cached_entry`, `_materialise_entry`, and `_populate_entry` helpers so `Cache.fetch_or_populate` focuses on orchestration with fewer branches.
- Preserve invariant and constant filtering while reducing duplicated storage reads for cached and superset paths.
- Capture cache-focused refactoring guidance in the `lessons_learned/` notebooks and surface the reminder inside `AGENTS.md`.

## Complexity reduction: filter pipeline and superset resolver
- Move invariant and constant filtering into a dedicated `CacheFilterPipeline`, trimming branches inside the main cache facade while keeping Arrow filtering logic together.
- Extract superset-matching responsibilities into a `SupersetResolver` that pre-normalises requested keys and compares non-invariant constants consistently.
- Continue producing complexity reports after each structural change to document hotspot score improvements and validate the metrics pipeline.


## Plain-language cache API summary
- Add `test_docs/cache_metadata_api_overview.md` to explain the cache architecture, filter behaviour, and metadata-first API proposal in approachable language for non-implementers.
- Highlight how invariant and non-invariant filters, shared base entries, and page-by-page fetching translate into a summary-first network design with explicit freshness controls.
- Expand the explainer with extra background on what the cache stores, how response envelopes work, and why the metadata-first API lines up with existing helpers.

## Drop invariant superset reuse in cache
- Remove the superset resolver and always serve responses from a single broad cache entry keyed by parameters and non-invariant constants.
- Update cache key digests to ignore invariant tokens, store Parquet pages without invariant filtering, and apply invariants only when building the response envelope.
- Refresh cache tests and docs to reflect the shared-base-entry model and the fact that `from_superset` now remains `False` while cached invariant metadata stays empty.

## Inline template metadata extraction
- Add `webbed_duck.server.template_metadata` to parse inline `webbed_duck:` directives and `ctx.ui` helper calls into immutable validator descriptors.
- Extend the templating renderer with a metadata callback and `ctx.ui` proxy so templates can register validators without influencing rendered SQL.
- Provide a route registry helper that scans SQL templates, attaches cache invariant descriptors, and surface the combined description via new `webbed_duck.server` exports and tests.

## Follow-up: validation-derived template metadata
- Remove the renderer metadata callback and instead reuse existing guard and whitelist logic when assembling template metadata, dropping the temporary `ctx.ui` helpers.
- Teach `collect_template_metadata` to merge inline directives with parameter whitelists and guard definitions sourced from validation manifests so UI descriptors mirror enforcement rules.
- Allow the route registry builder to accept per-slug validation contexts and update tests to assert the combined invariants, validation-derived directives, and inline annotations.

## Follow-up: documented route metadata examples
- Add an example sales route template under `examples/route_templates/` to anchor inline metadata directives in documentation.
- Capture the route registry output in `test_docs/cache_route_metadata_examples.md` and assert it stays in sync through a regression test.
- Extend the template metadata test module with helpers that render registry output for docs so consumers can see how comments, whitelists, and validation manifests interact.

## Follow-up: inline directive parsing polish
- Allow `webbed_duck:` directives to appear inline alongside SQL clauses so validators sit next to the parameters they protect.
- Update server metadata tests to cover inline comment parsing and keep route registry expectations anchored to the new style.
- Refresh the catalog sales example template to showcase inline comments and keep the documentation snapshot current.

## Follow-up: call-based inline directives
- Replace SQL comment annotations with `{{ webbed_duck.*(...) }}` inline directives so metadata stays adjacent to the guarded expressions without hiding in comments.
- Parse inline directive calls via the metadata collector, treating the first two positional arguments as `target` and `name` while capturing keyword options for descriptors.
- Teach the template renderer to treat inline directive calls as no-op expressions during rendering, update docs/tests/examples to the call syntax, and keep validation-derived metadata intact.

## Follow-up: inline DSL reference documentation
- Add a dedicated reference template under `examples/route_templates/reference/` that demonstrates inline validator calls alongside real whitelist and range guards.
- Generate `test_docs/template_metadata_dsl_reference.md` from the test suite so the documented DSL, sample SQL, and collected metadata stay in sync.
- Extend the template metadata tests with helpers that render the DSL reference, covering inline directives, validation-derived annotations, and aggregated route registry output.

## Cache concurrency guard and regression test
- Protect cache population with per-digest mutexes so directory creation, Parquet writes, and metadata persistence happen once per key even under concurrent requests.
- Re-check for freshly written entries while holding the mutex to avoid rerunning queries when a peer request already populated the cache.
- Add a regression test that spawns two threads targeting the same key and asserts that both handles are valid while only one set of Parquet pages hits disk.

## Cache metadata peek API and HTTP surface
- Introduce a `CacheMetadataSummary` dataclass plus a `peek_metadata` helper that reads only `metadata.json`, ensuring freshness checks match the response envelope while keeping Parquet untouched.
- Add FastAPI response models and a `GET /cache/{digest}` route that exposes metadata summaries along with download URL templates for page-level exports.
- Cover the helper and HTTP route with regression tests that assert the metadata flags mirror cached responses and that Parquet reads are skipped during metadata peeks.

## FastAPI route executor and streaming endpoints
- Add a `create_route_app` helper that wires the cache, request context store, DuckDB executor, and compiled route registry into a FastAPI application.
- Surface `POST /routes/{slug}` to validate parameters via `_templating.binding`, fetch or populate the cache, and return the response envelope metadata alongside cache page templates.
- Expand the cache router with `/cache/{digest}/pages/{page}` streaming plus new Pydantic models so the metadata and route APIs share the same download contract.
- Cover the HTTP stack with regression tests that spin up `TestClient`, submit parameter payloads end-to-end, and assert cache hits/misses as well as streamed JSON payloads line up with expectations.
