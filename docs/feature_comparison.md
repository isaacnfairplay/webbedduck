# Feature Comparison

This document summarises the high-value capabilities available in the public `webbed_duck` project and evaluates whether they exist in this repository (`webbedduck`). Only net-new items are captured.

## Matrix

| webbed_duck feature | Status in webbedduck | Decision | Rationale |
| --- | --- | --- | --- |
| Template parameter guards and string filters declared via SQL template slots (`webbed_duck/core/interpolation.py`) | Missing | Mirror (subset) | Provides reusable lower/upper/json-style filters and identifier/literal guards for template values; high leverage for constant templating without introducing the full SQL compiler surface. |
| Plugin-driven preprocess pipeline for parameter mutation before execution (`webbed_duck/server/preprocess.py`) | Missing | Skip (for now) | Would require porting plugin loader, callable resolution, and FastAPI request plumbing; exceeds current complexity budget and depends on runtime subsystems not yet present in this fork. |
| Declarative TOML+SQL route compiler with parameter typing and guard metadata (`webbed_duck/core/compiler.py`, `webbed_duck/core/routes.py`) | Missing | Skip | Reproducing the compiler, TOML schema, and runtime integration is a multi-module effort that would dwarf existing scope; defer until core templating/runtime pieces are ready. |
| Incremental cache slice serving with offset/limit selection and invariant-aware reuse (`webbed_duck/server/cache.py`) | Different | Skip (monitor) | Our cache already supports invariants and superset reuse but lacks offset/limit pagination helpers; implementing readers, pagination metadata, and invariant indexes would significantly raise complexity relative to immediate value. |

## Prioritised Shortlist

| Feature | Value Add (1-5) | Ease (1-5) | Risk | Complexity Budget (ΔCC, ΔSLOC) | Affected Modules |
| --- | --- | --- | --- | --- | --- |
| Template string modifiers (lower/upper/json/identifier/literal) | 4 | 4 | Low | ΔCC ≤ +2, ΔSLOC ≤ +120 | `webbed_duck/_templating/renderer.py`, `webbed_duck/_templating/formatters.py`, tests |

The shortlist reflects features that maximise template expressiveness per unit of code while fitting within the present architecture.
