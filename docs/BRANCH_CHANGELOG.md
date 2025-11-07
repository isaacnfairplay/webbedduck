# Branch Changelog

## Template Modifiers Expansion
- Added lower/upper/identifier/literal/json modifiers to the templating engine for parity with the public `webbed_duck` compare guards.
- Documented requirements, migration guidance, and feature comparison rationale for the new modifiers.
- Extended tests to cover the new behaviour and guardrail failure modes.
- Captured complexity baselines (before/after) under `reports/complexity/` with an auto-generated summary.
- Reduced `_validate_compare_guard` cyclomatic complexity by introducing helper utilities and early-return flow.
