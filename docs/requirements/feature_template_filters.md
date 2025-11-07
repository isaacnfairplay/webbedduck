# Feature: Template Filter Pipeline for Request Constants

## Overview
Introduce a filter pipeline for `{{ ctx.* }}` templates so declarative filters from route definitions can be expressed without writing Python modifiers. The feature focuses on mirroring the guard-friendly filters from `webbed_duck/core/interpolation.py`, adding safe lower/upper casing, identifier enforcement, SQL literal rendering, and JSON serialisation.

## Acceptance Criteria
- Templates may append bare filter names after `|` (for example `| upper`) in addition to the existing function-style modifiers.
- Supported filters: `lower`, `upper`, `identifier`, `literal`, and `json`.
- `lower`/`upper` accept only non-null text values; using them on non-text data raises `TemplateApplicationError` with a message indicating that text is required.
- `identifier` ensures the value is a valid SQL identifier (`[A-Za-z_][A-Za-z0-9_]*`); invalid values raise `TemplateApplicationError` that mentions the identifier requirement.
- `literal` emits SQL-safe literals matching the behaviour in `webbed_duck/core/interpolation.py`:
  - Strings are single-quoted with embedded quotes doubled.
  - Numbers emit their string form without quotes.
  - Booleans emit `TRUE`/`FALSE`.
  - `None` emits `NULL`.
  - Lists/tuples are rendered as comma-separated lists wrapped in parentheses.
- `json` renders the JSON string via `json.dumps`.
- Filters compose with existing modifiers and keep compatibility with `TemplateRenderer` serialization (`stringify`).

## Non-Goals
- Guard modes such as `choices`, `path`, or `role` remain unimplemented.
- No new plugin loading or runtime request context wiring is introduced.
- Filter registration remains internal; exposing a public registry is out-of-scope.

## Error Handling
- Missing or unsupported filters raise `TemplateApplicationError` with a specific message (`Unknown filter 'name'`).
- Invalid value types raise `TemplateApplicationError` with messages matching the acceptance criteria so tests can assert on them.

## Examples
- `{{ ctx.constants.str.report_name | upper }}` → `DAILY METRICS`
- `{{ ctx.constants.str.report_name | identifier }}` raises because the value contains a space.
- `{{ ctx.constants.misc.optional | literal }}` → `NULL`
- `{{ ctx.constants.misc.choices | literal }}` with `choices = ["alpha", "beta"]` → `('alpha', 'beta')`
- `{{ ctx.constants.str.report_name | json }}` → `"Daily Metrics"`
