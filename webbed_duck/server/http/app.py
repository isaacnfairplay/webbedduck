"""HTTP application wiring for the DuckDB cache and route executor."""

from __future__ import annotations

import copy
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import pyarrow as pa
import pyarrow.ipc as paipc
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response, StreamingResponse

from ..._templating import RequestContextStore, TemplateRenderer
from ..._templating.binding import ParameterBindingError, ValidationContext
from ..cache import (
    Cache,
    CacheConfig,
    CacheKey,
    CacheStorage,
    DataHandle,
    peek_metadata,
)
from ..template_metadata import RouteDescription, build_route_registry
from .models import (
    CacheMetadataResponse,
    RouteExecutionRequest,
    RouteExecutionResponse,
)


@dataclass(frozen=True)
class RouteRuntimeState:
    """Objects shared across HTTP handlers."""

    cache: Cache
    cache_config: CacheConfig
    registry: Mapping[str, RouteDescription]
    validations: Mapping[str, ValidationContext]
    request_context: RequestContextStore


@dataclass(frozen=True)
class ResolvedRouteRequest:
    """Cache key inputs resolved for a route execution."""

    slug: str
    description: RouteDescription
    parameters: dict[str, Any]
    constants: dict[str, Any]
    cache_key: CacheKey


class DuckDBExecutor:
    """Very small helper that executes SQL against a DuckDB database."""

    def __init__(self, database: str | Path = ":memory:") -> None:
        import duckdb  # local import to avoid mandatory dependency during docs builds

        self._duckdb = duckdb
        self._database = str(database)

    def __call__(self, sql: str) -> pa.Table:
        connection = self._duckdb.connect(database=self._database)
        try:
            return connection.sql(sql).arrow().read_all()
        finally:
            connection.close()


class RouteQueryRunner:
    """Callable compatible with :class:`Cache` that renders and runs SQL."""

    def __init__(
        self,
        *,
        registry: Mapping[str, RouteDescription],
        validations: Mapping[str, ValidationContext],
        request_context_store: RequestContextStore,
        executor: DuckDBExecutor,
    ) -> None:
        self._registry = registry
        self._validations = validations
        self._request_context_store = request_context_store
        self._executor = executor

    def __call__(
        self, route_slug: str, parameters: Mapping[str, Any], constants: Mapping[str, Any]
    ) -> pa.Table:
        description = self._registry.get(route_slug)
        if description is None:
            raise RuntimeError(f"Unknown route '{route_slug}'")
        validation = self._validations.get(route_slug, _EMPTY_VALIDATION_CONTEXT)
        parameter_context = validation.resolve(parameters)
        renderer_context = _build_renderer_context(
            self._request_context_store, parameter_context
        )
        renderer = TemplateRenderer(renderer_context)
        template_text = description.template_path.read_text(encoding="utf-8")
        sql = renderer.render(template_text)
        return self._executor(sql)


def create_route_app(
    *,
    cache_config: CacheConfig,
    template_root: Path,
    request_context: Mapping[str, Any] | None = None,
    validations: Mapping[str, Mapping[str, Any] | ValidationContext] | None = None,
    duckdb_database: str | Path = ":memory:",
) -> FastAPI:
    """Create a FastAPI app that executes SQL templates via the cache."""

    registry = build_route_registry(
        template_root,
        cache_config=cache_config,
        request_context=request_context,
        validations=validations,
    )
    store = RequestContextStore()
    store.set(request_context or {})
    validation_map = {
        slug: description.validation or _EMPTY_VALIDATION_CONTEXT
        for slug, description in registry.items()
    }
    storage = CacheStorage(cache_config.storage_root)
    executor = DuckDBExecutor(duckdb_database)
    runner = RouteQueryRunner(
        registry=registry,
        validations=validation_map,
        request_context_store=store,
        executor=executor,
    )
    cache = Cache(config=cache_config, run_query=runner, storage=storage)

    app = FastAPI()
    app.state.route_state = RouteRuntimeState(
        cache=cache,
        cache_config=cache_config,
        registry=registry,
        validations=validation_map,
        request_context=store,
    )
    app.include_router(create_cache_router(storage))
    app.include_router(_ROUTE_ROUTER)
    return app


def create_cache_router(storage: CacheStorage) -> APIRouter:
    """Build a router that exposes cache metadata and streaming endpoints."""

    router = APIRouter()

    def get_storage() -> CacheStorage:
        return storage

    @router.get(
        "/cache/{digest}",
        name="cache-metadata",
        response_model=CacheMetadataResponse,
        summary="Inspect cached entry metadata without loading Parquet pages",
    )
    def read_cache_metadata(
        digest: str, request: Request, store: CacheStorage = Depends(get_storage)
    ) -> CacheMetadataResponse:
        summary = peek_metadata(store, digest)
        if summary is None:
            raise HTTPException(status_code=404, detail="Cache entry not found")
        template = _build_page_template(request, summary.digest)
        return CacheMetadataResponse.from_summary(summary, page_url_template=template)

    @router.get(
        "/cache/{digest}/pages/{page}",
        name="cache-page",
        summary="Download a cached page in the requested format",
    )
    def stream_cache_page(
        digest: str,
        page: int,
        *,
        format: str = Query("parquet"),
        store: CacheStorage = Depends(get_storage),
    ) -> Response:
        handle = _resolve_cache_handle(store, digest)
        if handle is None:
            raise HTTPException(status_code=404, detail="Cache entry not found")
        try:
            return _stream_handle(handle, format=format, page=page)
        except ValueError as exc:  # invalid page or format
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router


def create_cache_app(storage: CacheStorage) -> FastAPI:
    """Create a standalone FastAPI application exposing cache metadata routes."""

    app = FastAPI()
    app.include_router(create_cache_router(storage))
    return app


_ROUTE_ROUTER = APIRouter()


def _get_route_state(request: Request) -> RouteRuntimeState:
    state = getattr(request.app.state, "route_state", None)
    if state is None:
        raise RuntimeError("Route runtime state is not configured")
    return state


@_ROUTE_ROUTER.post(
    "/routes/{slug:path}",
    name="route-execute",
    response_model=RouteExecutionResponse,
    summary="Resolve route parameters, run the query, and return cache metadata",
)
async def execute_route(
    slug: str,
    payload: RouteExecutionRequest,
    request: Request,
    state: RouteRuntimeState = Depends(_get_route_state),
) -> RouteExecutionResponse:
    resolved = _resolve_route_request(state, slug, payload)
    envelope = await run_in_threadpool(
        state.cache.fetch_or_populate, resolved.cache_key
    )
    digest = envelope.entry_digest or resolved.cache_key.digest
    page_template = _build_page_template(request, digest)
    return RouteExecutionResponse.from_envelope(
        envelope,
        slug=resolved.slug,
        digest=digest,
        parameters=resolved.parameters,
        constants=resolved.constants,
        page_url_template=page_template,
    )


_EMPTY_VALIDATION_CONTEXT = ValidationContext(specs={}, allow_unknown_parameters=True)
_TEXT_FORMATS = {"csv", "json", "jsonl"}
_MEDIA_TYPES = {
    "arrow": "application/vnd.apache.arrow.stream",
    "csv": "text/csv",
    "json": "application/json",
    "jsonl": "application/x-ndjson",
    "parquet": "application/octet-stream",
}


def _build_renderer_context(
    store: RequestContextStore, parameter_context: Any
) -> Mapping[str, Any]:
    base_context = store.get()
    context: dict[str, Any] = {}
    for key, value in base_context.items():
        if key == "parameters":
            continue
        context[key] = copy.deepcopy(value)
    context["parameters"] = parameter_context.with_configuration(
        copy.deepcopy(base_context.get("parameters", {}))
    )
    return context


def _build_page_template(request: Request, digest: str) -> str:
    sample = str(request.url_for("cache-page", digest=digest, page=0)).rsplit("/", 1)[0]
    return f"{sample}/{{page}}?format={{format}}"


def _resolve_route_request(
    state: RouteRuntimeState, slug: str, payload: RouteExecutionRequest
) -> ResolvedRouteRequest:
    description = _require_route_description(state.registry, slug)
    parameters = _resolve_payload_parameters(slug, payload, state.validations)
    constants = dict(payload.constants)
    cache_key = CacheKey.from_parts(
        slug,
        parameters=parameters,
        constants=constants,
        invariant_filters=state.cache_config.invariants,
    )
    return ResolvedRouteRequest(
        slug=slug,
        description=description,
        parameters=parameters,
        constants=constants,
        cache_key=cache_key,
    )


def _require_route_description(
    registry: Mapping[str, RouteDescription], slug: str
) -> RouteDescription:
    description = registry.get(slug)
    if description is None:
        raise HTTPException(status_code=404, detail="Route not found")
    return description


def _resolve_payload_parameters(
    slug: str,
    payload: RouteExecutionRequest,
    validations: Mapping[str, ValidationContext],
) -> dict[str, Any]:
    validation = validations.get(slug, _EMPTY_VALIDATION_CONTEXT)
    try:
        parameter_context = validation.resolve(payload.parameters)
    except ParameterBindingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {name: resolved.value for name, resolved in parameter_context.items()}


def _resolve_cache_handle(storage: CacheStorage, digest: str) -> DataHandle | None:
    summary = peek_metadata(storage, digest)
    if summary is None:
        return None
    entry_dir = storage._root / summary.digest
    metadata_path = storage._metadata_path(entry_dir)
    metadata = storage._read_metadata(metadata_path)
    key = metadata.to_cache_key(digest=summary.digest)
    entry = metadata.to_cache_entry(key=key, path=entry_dir)
    table = storage.read_entry(entry)
    return DataHandle(table, entry.page_size, entry.path)


def _stream_handle(handle: DataHandle, *, format: str, page: int) -> Response:
    _ensure_supported_format(handle, format)
    if format == "arrow":
        payload = _read_arrow_payload(handle, page)
        return Response(payload, media_type=_MEDIA_TYPES[format])

    iterator = _iter_stream(handle, format, page)
    return StreamingResponse(iterator, media_type=_MEDIA_TYPES[format])


def _ensure_supported_format(handle: DataHandle, format: str) -> None:
    if format in handle.formats:
        return
    supported = ", ".join(handle.formats)
    raise ValueError(f"Unsupported format '{format}'. Expected one of {supported}")


def _read_arrow_payload(handle: DataHandle, page: int) -> bytes:
    with handle.open("arrow", page=page) as table:
        sink = pa.BufferOutputStream()
        with paipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return sink.getvalue().to_pybytes()


def _iter_stream(handle: DataHandle, format: str, page: int) -> Iterator[bytes]:
    if format in _TEXT_FORMATS:
        yield from _iter_text_chunks(handle, format, page)
        return
    yield from _iter_binary_chunks(handle, format, page)


def _iter_text_chunks(handle: DataHandle, format: str, page: int) -> Iterator[bytes]:
    with handle.open(format, page=page) as stream:
        for chunk in iter(lambda: stream.read(8192), ""):
            yield chunk.encode("utf-8")


def _iter_binary_chunks(handle: DataHandle, format: str, page: int) -> Iterator[bytes]:
    with handle.open(format, page=page) as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            yield chunk


__all__ = ["create_cache_app", "create_cache_router", "create_route_app"]
