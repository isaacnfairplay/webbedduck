# Feature: Safe relative path guard for parameters

## Overview
Upstream `webbed_duck` supports a dedicated guard that validates template-only parameters used as filesystem fragments, ensuring they remain relative and sandboxed (`webbed_duck/core/interpolation.py`). This project currently lacks an equivalent, so templated strings that represent file paths could traverse outside approved directories.

## Requirements
- When a parameter declares `guards = {"path": {...}}`, the validator must ensure provided values:
  - Default to rejecting absolute paths (e.g. `/etc/passwd`, `C:\secrets`).
  - Default to rejecting parent directory traversals containing `..` segments.
  - Default to rejecting backslash characters to avoid cross-platform escape issues.
- The guard accepts an optional configuration mapping with boolean flags:
  - `allow_parent`: when `True`, parent directory segments are allowed.
  - `allow_absolute`: when `True`, absolute paths are allowed.
  - `allow_backslash`: when `True`, backslash characters are allowed.
- Invalid guard configuration (non-mapping, or any flag not a boolean) must raise `ParameterBindingError` with an actionable message.
- Guard enforcement integrates with `ValidationContext.resolve` so callers receive aggregated error messages consistent with existing guard behaviour.
- Parameters that are optional and unset should bypass the guard without error.

## Acceptance criteria
- Resolving a manifest with the path guard applied to a relative child path succeeds and returns the original value.
- Providing a value containing parent traversal, an absolute path, or backslashes raises `ParameterBindingError` with clear hints about the violation.
- Setting `allow_parent = True` allows parent traversal while other defaults remain enforced.
- Invalid guard configuration (e.g. integer, string, or non-boolean flags) raises `ParameterBindingError` mentioning the configuration requirement.
- Leaving an optional parameter unset does not trigger validation errors.

## Out of scope
- Normalising or collapsing path separators beyond the specified checks.
- Resolving symlinks or verifying on-disk existence.
- Integrating request-specific security context or role-based authorisation (tracked separately).
