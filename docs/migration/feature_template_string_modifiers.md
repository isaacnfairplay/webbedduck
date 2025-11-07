# Template String Modifiers

## Configuration
- No new configuration keys are required. Existing request contexts automatically gain access to the new modifiers once the package is upgraded.
- Templates may call the modifiers via `{{ ctx.constants.* | lower() }}`, `upper()`, `identifier()`, `literal()`, or `json()` with no additional wiring.

## Defaults & Opt-In/Out
- Modifiers are available by default and have no flags to disable. Legacy templates continue to work because previously unsupported modifier names still raise `TemplateApplicationError`.
- To revert to pre-feature behaviour on a single template, remove the modifier call from the Jinja-style expression.

## Backward Compatibility
- Existing modifiers retain their behaviour; only new names were added to the dispatch table.
- `identifier()` only succeeds when the sanitised value is a valid SQL identifier. Templates that previously expected raw strings should keep using the underlying constant without the modifier.

## Rollback Plan
1. Remove uses of the new modifiers from templates or replace them with inline formatting before downgrading.
2. Reinstall the previous release of `webbed_duck`.
3. No data migrations are required because the feature is purely functional; clearing template caches is unnecessary.
