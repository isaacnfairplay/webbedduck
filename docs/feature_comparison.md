# Feature Comparison

This document contrasts capabilities between the upstream [`webbed_duck`](https://github.com/isaacnfairplay/webbed_duck) repository and this successor project. Sources in the upstream tree are referenced with their paths for traceability.

## Matrix

| Upstream feature | Upstream reference | Status in `webbedduck` | Decision | Rationale |
| --- | --- | --- | --- | --- |
| Declarative TOML + SQL route compiler producing FastAPI endpoints | `webbed_duck/core/compiler.py`, `webbed_duck/server/__init__.py` | Missing | Skip | Rebuilding the full compiler/runtime stack would exceed the current project's scope and complexity budget; this successor focuses on templating utilities for now. |
| Multi-format HTTP/Arrow/Parquet/HTML outputs | `webbed_duck/server/ui/pages.py`, `webbed_duck/server/__init__.py` | Missing | Skip | Requires large surface area (web stack, asset pipeline) not yet present here; defer until the lighter templating layer is stable. |
| Template-only parameter guard for safe relative paths | `webbed_duck/core/interpolation.py` (path guard logic) | Missing | Mirror | Guarding path-like parameters adds meaningful safety for downstream SQL/template rendering with modest complexity. |
| Template-only parameter guard requiring request roles | `webbed_duck/core/interpolation.py` (role guard) | Missing | Skip | Depends on FastAPI request objects and auth context that this project does not yet expose. |
| Template filters for JSON/literal/identifier coercion | `webbed_duck/core/interpolation.py` (filter loop) | Missing | Skip | Would require introducing a SQL templating pipeline; postponing until route compilation exists. |

## Ranked shortlist

| Feature | Value Add (1-5) | Ease (1-5) | Risk | Complexity Budget (ΔCC, ΔSLOC) | Affected Modules |
| --- | --- | --- | --- | --- | --- |
| Safe relative path guard for parameters | 4 | 4 | Low | ΔCC ≤ +2, ΔSLOC ≤ +60 | `webbed_duck/_templating/binding.py`, `tests/test_feature_safe_path_guard.py`, docs under `docs/requirements/` and `docs/migration/` |

The shortlisted feature aligns with upstream safety guarantees while requiring only incremental changes to the existing parameter validation layer.
