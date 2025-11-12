# Cache metadata API overview (plain-language)

This note restates the cache discussion with extra background so we can agree on the design without needing to dig through the code first.

## Quick background: what lives on disk

- Every cached result has its own folder. Inside you will always find:
  - A small JSON file. It records the total row count (for the broad, unfiltered dataset), how many rows we stored per page, timestamps, and an empty `invariants` map to remind us that the stored Parquet pages were written without invariant filters.
  - One Parquet file per page of results. Page 0 holds the first chunk of rows, page 1 the next chunk, and so on. CSV/JSON/JSONL are produced on demand from these Parquet pages.
- When application code calls the cache today it gets back a **response envelope**. Think of it as a cover sheet attached to the cached data. The envelope tells the caller:
  - Did we reuse something that already existed (`from_cache`)?
  - (Legacy field) A `from_superset` flag that now always reads `False` because every request shares the same broad cache entry.
  - Which output formats are available and how many pages are stored?
  - How to open a specific page via a helper called `DataHandle`.

## Filters in practice

- **Invariant filters** now run *after* we read the cached Parquet pages. We always write the broad dataset without invariant filtering, then apply the requested tokens in memory before returning rows to the caller.
- **Non-invariant filters** are cheaper. They run when we read a page. We load the cached data first and only then apply the non-invariant filter in memory.
- Because invariant values no longer affect the cache digest, the first request for a given parameter set creates the on-disk pages and every subsequent invariant combination reuses that exact entry.

## Why a metadata-first API fits the current model

1. **The JSON metadata already answers summary questions.** Row counts, page sizing, timestamps, and an empty invariant map live there today. The only missing piece is a helper that reads just that JSON file so we can answer a "summary only" request without touching Parquet data.
2. **Freshness signals are already captured.** The envelope flags tell us whether the result was served straight from disk or rebuilt. The legacy `from_superset` flag is now always `False`, so the metadata response can surface the same truth without any extra logic.
3. **Page-by-page downloads already exist in code.** After reading the summary, a caller can ask for page 0, page 1, etc. The `DataHandle` helper already does this internally; an HTTP endpoint would simply expose the same behaviour: "here is the cache digest, here are the formats, here is how to ask for page N".

## Recommended next steps (kept intentionally plain English)

1. **Add a lightweight `peek_metadata` helper.** It should load only the metadata JSON file and return the digest, timestamps, page counts, the (empty) invariant map, and the freshness flags. No Parquet files are opened during this step, so metadata calls stay quick.
2. **Mirror the envelope in the network contract.** The metadata response should mirror what the envelope already exposes: page count, formats, requested invariant tokens, and a template URL for fetching a page. Follow-up requests such as `GET /cache/<digest>/pages/<n>?format=parquet` can reuse the existing `DataHandle` logic to stream the cached Parquet (or convert it to CSV/JSON/JSONL) without recomputing anything.
3. **Document the freshness choices clearly.** Spell out two cases in the summary response: "freshly rebuilt" and "served from cache". Note that `from_superset` is a legacy field that will always be `False` while we rely on a shared base entry.

Sticking with this split keeps the current invariants-and-pages design intact while giving UI and API consumers a friendly, metadata-first handshake before they pull down individual pages.
