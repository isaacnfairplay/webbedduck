# Template String Modifiers

## Summary
Expose additional string-focused modifiers in `TemplateRenderer` so template authors can safely transform request constants without shelling out to bespoke Python code.

## Functional Requirements
- Introduce five modifiers that can be chained like existing helpers:
  - `lower()` — returns the string value lowercased. Non-string values raise `TemplateApplicationError`.
  - `upper()` — returns the string value uppercased. Non-string values raise `TemplateApplicationError`.
  - `identifier()` — validates the value as a SQL identifier (letters, numbers, underscores; must not start with a digit) and emits a canonical snake_case version. Rejects invalid values with an explicit error.
  - `literal()` — renders the value as a SQL-safe literal (`NULL`, `TRUE`/`FALSE`, numeric passthrough, quoted strings with single-quote escaping, list/tuple expansion `(a, b, c)`).
  - `json()` — serialises the value via `json.dumps` using default settings.
- Modifiers participate in the existing dispatch pipeline and are accessible from templates via `| modifier()` syntax.
- Error messages must surface the offending modifier so template authors can diagnose misuses quickly.

## Non-Functional Requirements
- Preserve existing modifiers and behaviour; new modifiers should not change prior output formatting.
- Keep cyclomatic complexity impact within the documented budget (ΔCC ≤ +2) by reusing helper functions and short guard clauses.
- Maintain type hints and error consistency with existing templating code.

## Acceptance Criteria
- New unit tests cover happy-path transformations and error handling (invalid identifier, non-string case operations, JSON output, literal serialisation for strings, numbers, and `None`).
- `TemplateApplicationError` is raised with a clear message when identifier validation fails or case modifiers see incompatible data.
- Chaining still works with existing modifiers (manual smoke tests via unit coverage sufficient).

## Out of Scope
- No support for user-registered modifiers or plugin discovery.
- No new template expression syntax beyond the introduced modifier names.
