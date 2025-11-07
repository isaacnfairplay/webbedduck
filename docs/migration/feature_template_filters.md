# Migration: Template Filter Pipeline

## Summary
Adds built-in filters (`lower`, `upper`, `identifier`, `literal`, `json`) to the `{{ ctx.* }}` templating pipeline so route authors can mirror the protections present in `webbed_duck`. Filters apply after existing modifiers and emit serialised strings via the normal renderer.

## Configuration
- No new configuration flags are required.
- Filters are available automatically once the package is upgraded.

## Backward Compatibility
- Templates that previously used bare identifiers after `|` now resolve as filters. Because earlier releases raised `TemplateApplicationError` for bare segments, behaviour only changes from failure → success for supported filter names.
- Function-style modifiers continue to work unchanged.

## Rollback Plan
- To disable the feature, remove the filter handling code paths introduced in `webbed_duck/_templating/renderer.py` and delete the new tests/documentation.
- Existing templates that rely on the new filters will revert to raising `TemplateApplicationError` if the feature is rolled back.
