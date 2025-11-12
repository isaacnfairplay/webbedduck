# Feature gap analysis between legacy `webbed_duck` and this successor

## Overview
- The [legacy repository](https://github.com/isaacnfairplay/webbed_duck) ships a fully featured DuckDB data server with a CLI, FastAPI runtime, plugin ecosystem, cache orchestration, and UI assets.
- This repository currently focuses on foundational building blocks (templating, parameter binding, and a Parquet-backed cache) that will support a redesigned implementation.
- Goal: replace the legacy codebase without inheriting unnecessary coupling. We should reuse proven concepts while simplifying the runtime and embracing a streaming-first design (e.g., moving from Arrow tables in memory to Parquet artifacts in flight).

## Current functionality snapshot (this repo)
- **Templating pipeline** – `webbed_duck.constants.apply_constants` and the `_templating` package provide request context preparation, string whitelists, formatters, and template modifiers that render `{{ ctx.* }}` placeholders.【F:webbed_duck/constants.py†L1-L37】【F:webbed_duck/_templating/renderer.py†L16-L113】
- **Parameter binding helpers** – `ParameterContext` supports coercion, defaults, template visibility, and guard checks (choices, regex, length, range, datetime windows, comparisons).【F:webbed_duck/_templating/binding.py†L23-L338】
- **Request context orchestration** – `RequestContextStore` normalizes constants, merges formatter overrides, and enforces whitelists for string constants.【F:webbed_duck/_templating/state.py†L1-L131】
- **Parquet-backed cache** – `webbed_duck.server.cache.Cache` materializes Arrow tables into paged Parquet files, stores them without invariant filtering, and reapplies invariants per request.【F:webbed_duck/server/cache.py†L1-L310】【F:webbed_duck/server/cache.py†L311-L640】
- **Engineering tooling** – `tools/complexity_report.py` already extracts Radon metrics, Halstead measures, and clone detection to help keep future contributions maintainable.【F:tools/complexity_report.py†L1-L123】

## Legacy functionality highlights
- CLI commands (`compile`, `serve`, `perf`) with optional watch mode, hot reload, and strict config parsing.
- Declarative route authoring via paired TOML + SQL sidecars, with compiler outputs stored under `routes_build/`.
- Runtime features: FastAPI application, request/session management, overlays, share links, authentication stubs, diagnostics endpoints.
- Extensive format support (HTML tables/cards/feeds, JSON variants, CSV, Parquet, Arrow streaming, Chart.js embeds).
- Plugin loaders for charts, static assets, email notifications, and third-party integrations.
- Route dependency graph (`[[uses]]` blocks) with execution modes (`relation`, `parquet_path`) and cache control.
- Incremental job helper for scheduled catch-up runs with checkpointing.
- Comprehensive CI gates (ruff, mypy strict modes, pytest incl. snapshot & benchmark suites, bandit, vulture, radon).
- Front-end assets (Chart.js vendoring, modular UI templates, vitest coverage).

## Feature comparison and recommendations

### Template & parameter handling
| Feature | Legacy repo | This repo | Usefulness | Implementation complexity | Footguns / integration notes |
| --- | --- | --- | --- | --- | --- |
| Request context templating | Mature renderer with filters + context store | Implemented with request store, modifiers, whitelists | Essential – enables reuse of metadata + SQL templating | Low (already present; only needs API hardening) | Ensure contexts remain immutable between requests to avoid leakage |
| Parameter validation & guards | Rich `[params]` manifest with type coercion, guards | `ParameterContext` covers similar guards and provenance | High – protects DuckDB from unsafe inputs | Medium – align manifest schema with new pipeline | Guard order matters; mismatched dependency resolution can throw confusing errors |
| Template-only parameter exposure | Provided via `allow_template` flags in manifests | `for_template()` tracks consumption, prevents reuse | High – necessary for safe identifier interpolation | Low – feature exists; expose in higher-level API | Must surface errors when template uses non-whitelisted params |

### Caching & data access
| Feature | Legacy repo | This repo | Usefulness | Implementation complexity | Footguns / integration notes |
| Parquet cache storage | Paged Parquet artifacts with invariant-aware reuse | Same primitives via `Cache` and `CacheStorage` | High – keeps hot slices on disk, supports sharing | Medium – we need to wire it into execution stack | Need deterministic ordering + schema validation before reuse |
| Invariant filters with shared base entry | Supports case-insensitive tokens, filters applied on read | Implemented via `InvariantFilter` tokens and digest rules that ignore invariants | High – avoids rerunning DuckDB for invariant tweaks | Medium – filter definitions still live in route metadata | Stored pages include all invariant values, so clients must filter on read |
| Arrow streaming outputs | Legacy exposes Arrow/IPC endpoints | Currently expect in-memory Arrow tables from runners | Medium – Arrow readers keep compatibility with clients | Medium – need streaming pipeline; also plan Parquet handoff | Arrow tables inflate memory footprint; adopt Parquet streaming earlier in stack |
| Direct Parquet responses | HTTP `?format=parquet` streaming | Cache already writes Parquet; no HTTP surface yet | High – aligns with desire to avoid large Arrow payloads | Medium – once HTTP layer exists, reuse cache pages | Ensure file handles close cleanly; chunked transfer support needed |

### Route authoring & compilation
| Feature | Legacy repo | This repo | Usefulness | Implementation complexity | Footguns / integration notes |
| TOML + SQL route compiler | Mature CLI/compiler populating `routes_build` | Not yet implemented | High – single source of truth, typed params | High – requires parser, manifest schema, code generation | Need deterministic module names; partial compilation can desync source/build |
| Route dependency graph (`[[uses]]`) | Allows composable joins, parquet-path execution | Missing | High – promotes reuse & modular routes | High – demands execution planner + dependency resolver | Cycles / inconsistent cache modes can deadlock runs |
| Config system (`config.toml`) | Layered defaults for server, cache, plugins | Missing | Medium – centralizes environment config | Medium – implement dataclass/TOML loader | Misconfig detection + helpful errors essential to avoid silent fallbacks |

### Runtime & delivery
| Feature | Legacy repo | This repo | Usefulness | Implementation complexity | Footguns / integration notes |
| FastAPI/ASGI server | Full HTTP runtime with endpoints & middleware | No web server yet | High – required for parity | High – need request lifecycle, DI, streaming responses | Coupling cache + templating + parameter binding carefully to avoid blocking IO |
| Watch mode / hot reload | File polling and recompilation | Missing | Medium – developer ergonomics | High – depends on compiler + dependency tracking | File watching across network filesystems tricky; need debouncing |
| Authentication & share workflows | Pseudo-auth, share links, overlays | Missing | Medium – needed for production security | High – requires session store, token handling | Security-critical; poor defaults create footguns |
| Diagnostics & metadata endpoints | `/routes`, `/diagnostics`, overlays | Missing | Medium – aids observability | Medium – once server exists | Must redact secrets; ensure expensive diagnostics are cached |

### Output formats & UI
| Feature | Legacy repo | This repo | Usefulness | Implementation complexity | Footguns / integration notes |
| HTML table/card/feed renderers | Server-side templating + vendored assets | Missing | Medium – provides user-friendly default UI | Medium – can reuse static assets strategy later | Need accessibility + theming; keep logic decoupled from data core |
| Chart.js integration | Plugin + vendored asset fallback | Missing | Low/Medium – nice-to-have but not critical early | Medium – requires plugin loader + asset delivery | CDN fallback vs bundled asset must handle air-gapped installs |
| CSV streaming | HTTP `?format=csv` outputs | Missing | Medium – essential for analysts | Medium – convert Arrow/Parquet to stream | Watch out for memory blowups; use incremental writes |

### Operations, tooling, and jobs
| Feature | Legacy repo | This repo | Usefulness | Implementation complexity | Footguns / integration notes |
| CLI (`webbed-duck`) | Compile/serve/perf commands | Missing | High – entrypoint for operators | Medium – start with minimal serve/compile wrappers | Keep CLI thin; avoid re-embedding business logic |
| Incremental runner | Checkpointed catch-up jobs | Missing | Medium – supports scheduled refresh | Medium/High – depends on route executor abstraction | Checkpoint corruption or schema drift can poison reruns |
| Plugin loading framework | Charts, email, assets | Missing | Medium – future extensibility | High – requires stable extension API | Version skew between plugins and core causes runtime errors |
| Test harness breadth | Pytest suites incl. HTTP, snapshot, property tests | Focused tests around templating + cache | High – ensures regressions caught | Medium – expand coverage alongside new features | Need strategy to mock DuckDB + filesystem without flakiness |

### Front-end & ecosystem
| Feature | Legacy repo | This repo | Usefulness | Implementation complexity | Footguns / integration notes |
| Bundled UI assets + vitest | Chart.js, CSS, TypeScript packages | None | Medium – supports interactive dashboards | High – involves build tooling | Asset pipeline can bloat repo; prefer CDN opt-in |
| Documentation site | Docs folder, security policy | Minimal README only | High – necessary for onboarding | Low/Medium – extend docs with MkDocs or Sphinx | Keep docs versioned with code to avoid drift |

## Recommended sequencing (minimal disruption)
1. **Stabilize the core primitives** – Finalize templating and cache APIs, introduce formal data contracts so higher layers can depend on them with confidence. Start shifting long-running responses to Parquet streaming instead of large Arrow tables to honour the memory goals.
2. **Introduce a slim compiler + manifest loader** – Parse TOML + SQL into `RouteDefinition` objects that wrap parameter specs, cache settings, and SQL bodies. Reuse existing parameter guards and cache config to avoid duplicating validation logic.
3. **Layer a minimal ASGI service** – Expose a small FastAPI (or Starlette) application that wires request parsing → parameter binding → executor → cache. Begin with JSON/Parquet responses before expanding to HTML or CSV.
4. **Iterate on developer ergonomics** – Add a thin CLI, file watching, and targeted tests as the surface grows. Delay plugins and advanced UI until the execution core is proven stable.

## Design principles to carry forward
- **Composable boundaries** – Keep templating, parameter binding, execution, and caching as separate modules so alternative frontends (CLI, batch jobs, HTTP) can reuse them without circular imports.
- **Streaming-first mindset** – Prefer interfaces that pass Parquet file handles or readers instead of fully materialized Arrow tables. The cache already writes Parquet pages; extend that pattern so downstream consumers can stream results without rehydrating into memory.
- **Explicit configuration** – Adopt structured config objects (dataclasses or `pydantic` models) with clear validation errors instead of free-form dictionaries.
- **Observability hooks** – Provide well-defined extension points for logging, metrics, and tracing so operators can inspect query latency and cache behaviour.
- **Progressive enhancement** – Deliver a reliable JSON/Parquet core before layering UI, share links, or plugins, reducing the risk of regressions while we replace the legacy codebase.

