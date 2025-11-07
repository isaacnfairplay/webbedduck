# Migration: Safe relative path guard

## Summary
The parameter validation layer now recognises a `path` guard that enforces relative, backslash-free values by default.

## Configuration
- Guard syntax: declare `guards = {"path": {}}` on a parameter manifest entry.
- Optional flags inside the mapping:
  - `allow_parent` (default `false`): permit `..` segments.
  - `allow_absolute` (default `false`): permit absolute paths such as `/var/data` or `C:\data`.
  - `allow_backslash` (default `false`): permit `\\` characters.

No environment variables or global configuration toggles are required.

## Backward compatibility
- Existing manifests without a `path` guard are unaffected.
- Parameters already using `guards` continue to validate as before; the new guard participates in the aggregated error reporting.
- Optional parameters remain optional—the guard only runs when a value is supplied.

## Rollback plan
1. Remove `guards = {"path": ...}` entries from affected parameter manifests.
2. If necessary, revert the code change by checking out the previous revision of `webbed_duck/_templating/binding.py` and re-running the test suite.
3. Downstream callers do not require configuration changes after rollback.
