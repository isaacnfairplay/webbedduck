# Feature Comparison: webbed_duck vs. webbedduck

## Source Feature Matrix

| Feature | Status in webbedduck | Decision | Rationale |
| --- | --- | --- | --- |
| Template-only parameter filters (`lower`, `upper`, `identifier`, `literal`, `json`) with enforced SQL policies | Missing | Mirror (subset) | `webbed_duck/core/interpolation.py` enforces declarative filters and safe literal policies for template parameters to avoid injection vectors; webbedduck currently only supports function-style modifiers on constants. |
| Guard modes for template parameters (`choices`, `path`, `role`) | Missing | Skip (defer) | `webbed_duck/core/interpolation.py` validates guards such as path traversal and allowed choices before interpolation. Implementing role-aware guards in webbedduck would require a broader request model than we ship today. |
| Deterministic cache with invariant filter reuse | Missing | Skip | `webbed_duck/server/cache.py` maintains Parquet-backed caches with invariant-aware superset matching. webbedduck has no caching runtime yet, and introducing storage/runtime coupling would exceed the current complexity budget. |
| Preprocessor pipeline for parameter rewriting via plugin loader | Missing | Skip | `webbed_duck/server/preprocess.py` and `webbed_duck/plugins/loader.py` load declared preprocessors and run them before execution. webbedduck lacks plugin loading infrastructure, so mirroring this would require multiple new subsystems. |
| Multi-format streaming outputs (HTML views, JSON, CSV, Parquet, Arrow RPC) | Missing | Skip | `webbed_duck/server/app.py` dispatches to numerous streaming response builders. webbedduck is not yet an HTTP server, so this surface is intentionally out-of-scope for now. |
| Observability diagnostics with route execution timelines | Missing | Skip | `webbed_duck/server/diagnostics.py` collects execution spans and failures for UI display. webbedduck does not yet expose an execution runtime, so the feature would not deliver value today. |
| Context string whitelist enforcement for constants | Have | Skip | Both repositories enforce whitelisted constant lookups (webbed_duck via request context mapping; webbedduck through `StringNamespace` in `webbed_duck/_templating/state.py`), so no action is needed. |

## Implementation Shortlist

| Feature | Value Add (1-5) | Ease (1-5) | Risk | Complexity Budget (ΔCC, ΔSLOC) | Affected Modules |
| --- | --- | --- | --- | --- | --- |
| Template parameter filter pipeline with safe literal/identifier handling | 4 | 3 | Low | ΔCC ≤ +2, ΔSLOC ≤ +120 | `webbed_duck/_templating/renderer.py`, new filter helpers, examples/tests/docs |
