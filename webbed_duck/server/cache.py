"""DuckDB query cache with Parquet-backed storage."""

from __future__ import annotations

import contextlib
import hashlib
import json
import io
import shutil
from dataclasses import dataclass, field
from functools import reduce
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pyarrow.csv as pacsv


NULL_SENTINEL = "__null__"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _decode_null(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value == NULL_SENTINEL:
        return None
    return value


def _freeze_str_mapping(mapping: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): value for key, value in mapping.items()})


def _freeze_token_mapping(
    mapping: Mapping[str, Iterable[str]] | Mapping[str, Sequence[str]]
) -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType({str(name): tuple(tokens) for name, tokens in mapping.items()})


def _handle_expiry(
    expires_at: datetime, now: datetime, *, on_expire: Callable[[], None]
) -> bool:
    if now < expires_at:
        return False
    on_expire()
    return True


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, default=str, sort_keys=True)


def _normalize_items(mapping: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not mapping:
        return tuple()
    return tuple(
        (
            str(key),
            NULL_SENTINEL if (value := mapping[key]) is None else _stringify(value),
        )
        for key in sorted(mapping)
    )


def _ensure_sequence(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


def _encode_constant_value(
    key: str,
    value: Any,
    filters: Mapping[str, InvariantFilter],
    invariant_tokens: dict[str, tuple[str, ...]],
) -> str:
    filter_config = filters.get(key)
    if filter_config is None:
        return NULL_SENTINEL if value is None else _stringify(value)
    tokens = filter_config.normalize(value)
    invariant_tokens[key] = tokens
    return filter_config.encode(tokens)


def _compute_digest(route_slug: str, parameters: tuple[tuple[str, str], ...], constants: tuple[tuple[str, str], ...]) -> str:
    payload = json.dumps(
        {
            "route": route_slug,
            "parameters": parameters,
            "constants": constants,
        },
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _ensure_arrow_table(result: Any) -> pa.Table:
    if isinstance(result, pa.Table):
        return result
    if isinstance(result, pa.RecordBatchReader):
        return result.read_all()
    if hasattr(result, "fetch_arrow_table"):
        return result.fetch_arrow_table()
    if hasattr(result, "arrow"):
        return result.arrow()
    raise TypeError("Cache runner must return a pyarrow.Table or DuckDB relation")


@dataclass(frozen=True)
class InvariantFilter:
    """Normalisation and predicate configuration for invariant constants."""

    key: str
    column: str | None = None
    separator: str = "|"
    case_insensitive: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "column", self.column or self.key)

    def normalize(self, value: Any) -> tuple[str, ...]:
        if value is None:
            return (NULL_SENTINEL,)
        candidates: Iterable[Any]
        if isinstance(value, str):
            candidates = value.split(self.separator)
        else:
            candidates = _ensure_sequence(value)
        normalized = tuple(
            token
            for candidate in candidates
            if (token := self._normalize_candidate(candidate)) is not None
        )
        return tuple(sorted(dict.fromkeys(normalized)))

    def _normalize_candidate(self, candidate: Any) -> str | None:
        if candidate is None:
            return NULL_SENTINEL
        token = str(candidate).strip()
        if not token:
            return None
        if self.case_insensitive:
            return token.lower()
        return token

    def encode(self, tokens: tuple[str, ...]) -> str:
        if not tokens:
            return ""
        return self.separator.join(tokens)

    def apply(self, table: pa.Table, tokens: tuple[str, ...]) -> pa.Table:
        predicate = self._predicate_for(table, tokens)
        if predicate is None:
            return table
        if isinstance(predicate, pa.ChunkedArray):
            predicate = predicate.combine_chunks()
        return table.filter(predicate)

    def _predicate_for(
        self, table: pa.Table, tokens: tuple[str, ...]
    ) -> pa.Array | pa.ChunkedArray | None:
        if not tokens:
            return None
        column_name = self.column or self.key
        if column_name not in table.column_names:
            return None
        column_data = table[column_name]
        column_for_compare = (
            pc.utf8_lower(column_data)
            if self.case_insensitive and pa.types.is_string(column_data.type)
            else column_data
        )
        predicate = self._membership_predicate(column_for_compare, tokens)
        return self._include_nulls(predicate, column_data, tokens)

    def _membership_predicate(
        self, column_data: pa.Array | pa.ChunkedArray, tokens: tuple[str, ...]
    ) -> pa.Array | pa.ChunkedArray | None:
        compare_tokens = tuple(token for token in tokens if token != NULL_SENTINEL)
        if not compare_tokens:
            return None
        try:
            token_array = pa.array(compare_tokens, type=column_data.type)
        except (pa.ArrowTypeError, pa.ArrowInvalid):
            token_array = pa.array(compare_tokens).cast(column_data.type)
        return pc.is_in(column_data, value_set=token_array)

    def _include_nulls(
        self,
        predicate: pa.Array | pa.ChunkedArray | None,
        column_data: pa.Array | pa.ChunkedArray,
        tokens: tuple[str, ...],
    ) -> pa.Array | pa.ChunkedArray | None:
        if NULL_SENTINEL not in tokens:
            return predicate
        null_predicate = pc.is_null(column_data)
        if predicate is None:
            return null_predicate
        return pc.or_(predicate, null_predicate)


@dataclass(frozen=True)
class CacheSettings:
    """Cache behaviour configuration, including invariant filter metadata."""

    storage_root: Path = Path(".cache")
    ttl: timedelta = timedelta(minutes=10)
    page_size: int = 500
    invariants: Mapping[str, InvariantFilter] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "storage_root", Path(self.storage_root))
        if self.page_size <= 0:
            msg = "page_size must be a positive integer"
            raise ValueError(msg)
        if self.ttl <= timedelta(0):
            msg = "ttl must be positive"
            raise ValueError(msg)
        object.__setattr__(self, "invariants", _freeze_str_mapping(self.invariants))


CacheConfig = CacheSettings


@dataclass(frozen=True)
class CacheKey:
    """Immutable cache key identifying a cached query result."""

    NULL_SENTINEL: ClassVar[str] = NULL_SENTINEL

    route_slug: str
    parameters: tuple[tuple[str, str], ...]
    constants: tuple[tuple[str, str], ...]
    digest: str
    _raw_parameters: Mapping[str, Any] = field(repr=False)
    _raw_constants: Mapping[str, Any] = field(repr=False)
    _invariant_tokens: Mapping[str, tuple[str, ...]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_parts(
        cls,
        route_slug: str,
        *,
        parameters: Mapping[str, Any] | None = None,
        constants: Mapping[str, Any] | None = None,
        invariant_filters: Mapping[str, InvariantFilter] | None = None,
    ) -> "CacheKey":
        params_copy = {str(key): _decode_null(value) for key, value in (parameters or {}).items()}
        const_copy = {str(key): _decode_null(value) for key, value in (constants or {}).items()}
        normalized_params = _normalize_items(params_copy)
        filters = invariant_filters or {}
        invariant_tokens: dict[str, tuple[str, ...]] = {}
        normalized_constants_tuple = tuple(
            (
                key,
                _encode_constant_value(key, const_copy[key], filters, invariant_tokens),
            )
            for key in sorted(const_copy)
        )
        digest = _compute_digest(route_slug, normalized_params, normalized_constants_tuple)
        return cls(
            route_slug=route_slug,
            parameters=normalized_params,
            constants=normalized_constants_tuple,
            digest=digest,
            _raw_parameters=_freeze_str_mapping(params_copy),
            _raw_constants=_freeze_str_mapping(const_copy),
            _invariant_tokens=_freeze_token_mapping(invariant_tokens),
        )

    @property
    def parameter_values(self) -> Mapping[str, Any]:
        return self._raw_parameters

    @property
    def constant_values(self) -> Mapping[str, Any]:
        return self._raw_constants

    @property
    def invariant_tokens(self) -> Mapping[str, tuple[str, ...]]:
        return self._invariant_tokens


@dataclass(frozen=True)
class CacheEntry:
    """Metadata describing a cached materialisation on disk."""

    key: CacheKey
    path: Path
    created_at: datetime
    expires_at: datetime
    row_count: int
    page_size: int
    invariants: Mapping[str, tuple[str, ...]]


class _BaseAdapter:
    """Base class for data format adapters."""

    format: ClassVar[str]

    def open(self, table: pa.Table):  # pragma: no cover - implemented by subclasses
        raise NotImplementedError


class _ArrowAdapter(_BaseAdapter):
    format = "arrow"

    @contextlib.contextmanager
    def open(self, table: pa.Table):
        yield table


class _ParquetAdapter(_BaseAdapter):
    format = "parquet"

    @contextlib.contextmanager
    def open(self, table: pa.Table):
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink)
        buffer = io.BytesIO(sink.getvalue().to_pybytes())
        try:
            yield buffer
        finally:
            buffer.close()


class _CsvAdapter(_BaseAdapter):
    format = "csv"

    @contextlib.contextmanager
    def open(self, table: pa.Table):
        sink = pa.BufferOutputStream()
        pacsv.write_csv(table, sink)
        buffer = io.BytesIO(sink.getvalue().to_pybytes())
        try:
            yield buffer
        finally:
            buffer.close()


class _JsonAdapter(_BaseAdapter):
    format = "json"

    @contextlib.contextmanager
    def open(self, table: pa.Table):
        payload = json.dumps(table.to_pylist(), default=str)
        stream = io.StringIO(payload)
        try:
            yield stream
        finally:
            stream.close()


_ADAPTERS: Mapping[str, _BaseAdapter] = {
    adapter.format: adapter
    for adapter in (
        _ArrowAdapter(),
        _ParquetAdapter(),
        _CsvAdapter(),
        _JsonAdapter(),
    )
}

SUPPORTED_FORMATS: tuple[str, ...] = tuple(sorted(_ADAPTERS))


def _slice_table(table: pa.Table, page_size: int, page: int | None) -> pa.Table:
    if page is None:
        return table
    if page < 0:
        raise ValueError("Page numbers are 0-indexed")
    start = page * page_size
    if start >= table.num_rows:
        raise ValueError("Requested page exceeds available rows")
    length = min(page_size, table.num_rows - start)
    return table.slice(start, length)


@dataclass(frozen=True)
class DataHandle:
    """Format-aware accessor for cached query payloads."""

    _table: pa.Table
    _page_size: int

    def __post_init__(self) -> None:
        if self._page_size <= 0:
            raise ValueError("page_size must be greater than zero")

    @property
    def page_size(self) -> int:
        return self._page_size

    @property
    def row_count(self) -> int:
        return self._table.num_rows

    @property
    def table(self) -> pa.Table:
        return self._table

    @property
    def page_count(self) -> int:
        if self.row_count == 0:
            return 0
        return math.ceil(self.row_count / self._page_size)

    @property
    def schema(self) -> pa.Schema:
        return self._table.schema

    @property
    def formats(self) -> tuple[str, ...]:
        return SUPPORTED_FORMATS

    def as_arrow(self, page: int | None = None) -> pa.Table:
        return _slice_table(self._table, self._page_size, page)

    @contextlib.contextmanager
    def open(self, format: str, page: int | None = None):
        adapter = _ADAPTERS.get(format)
        if adapter is None:
            raise ValueError(
                f"Unsupported format '{format}'. Expected one of {SUPPORTED_FORMATS}."
            )
        table = self.as_arrow(page)
        with adapter.open(table) as payload:
            yield payload


@dataclass(frozen=True)
class ResponseEnvelope:
    """Immutable payload returned to callers after cache resolution."""

    data: DataHandle
    from_cache: bool
    from_superset: bool
    entry_digest: str | None
    requested_invariants: Mapping[str, tuple[str, ...]]
    cached_invariants: Mapping[str, tuple[str, ...]]
    created_at: datetime | None
    expires_at: datetime | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_invariants", _freeze_token_mapping(self.requested_invariants))
        object.__setattr__(self, "cached_invariants", _freeze_token_mapping(self.cached_invariants))

    @property
    def row_count(self) -> int:
        return self.data.row_count

    @property
    def schema(self) -> pa.Schema:
        return self.data.schema

    @property
    def page_size(self) -> int:
        return self.data.page_size

    @property
    def page_count(self) -> int:
        return self.data.page_count

    @property
    def formats(self) -> tuple[str, ...]:
        return self.data.formats

    @property
    def table(self) -> pa.Table:
        return self.data.table

    def to_pylist(self, page: int | None = None) -> list[dict[str, Any]]:
        return self.data.as_arrow(page).to_pylist()

    def as_arrow(self, page: int | None = None) -> pa.Table:
        return self.data.as_arrow(page)

    def open(self, format: str, page: int | None = None):
        return self.data.open(format, page)


# Backwards-compatible alias for downstream imports.
CacheResult = ResponseEnvelope


class CacheStorage:
    """Disk-backed storage abstraction used by the cache."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _entry_dir(self, key: CacheKey) -> Path:
        return self._root / key.digest

    def load_entry(self, key: CacheKey, now: datetime) -> CacheEntry | None:
        entry_dir = self._entry_dir(key)
        metadata_path = entry_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        data = json.loads(metadata_path.read_text())
        created_at = datetime.fromisoformat(data["created_at"])
        expires_at = created_at + timedelta(seconds=data["ttl_seconds"])
        if _handle_expiry(expires_at, now, on_expire=lambda: self.evict(key)):
            return None
        invariants = _freeze_token_mapping(data.get("invariants", {}))
        fallback_page_size = max(1, int(data.get("page_size") or data["row_count"] or 1))
        return CacheEntry(
            key=key,
            path=entry_dir,
            created_at=created_at,
            expires_at=expires_at,
            row_count=data["row_count"],
            page_size=fallback_page_size,
            invariants=invariants,
        )

    def write_entry(
        self,
        key: CacheKey,
        table: pa.Table,
        *,
        created_at: datetime,
        ttl: timedelta,
        page_size: int,
        invariants: Mapping[str, tuple[str, ...]] | None = None,
    ) -> CacheEntry:
        entry_dir = self._entry_dir(key)
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
        entry_dir.mkdir(parents=True, exist_ok=True)
        batches = list(table.to_batches(max_chunksize=page_size))
        if not batches:
            pq.write_table(table, entry_dir / "page-00000.parquet")
        else:
            for index, batch in enumerate(batches):
                page_table = pa.Table.from_batches([batch])
                pq.write_table(page_table, entry_dir / f"page-{index:05d}.parquet")
        metadata_path = entry_dir / "metadata.json"
        invariant_payload = {
            name: list(tokens)
            for name, tokens in (invariants or {}).items()
        }
        metadata = {
            "created_at": created_at.isoformat(),
            "ttl_seconds": ttl.total_seconds(),
            "row_count": table.num_rows,
            "page_size": page_size,
            "route_slug": key.route_slug,
            "parameters": list(key.parameters),
            "constants": list(key.constants),
            "raw_parameters": dict(key.parameter_values),
            "raw_constants": dict(key.constant_values),
            "invariants": invariant_payload,
        }
        metadata_path.write_text(json.dumps(metadata, sort_keys=True))
        expires_at = created_at + ttl
        return CacheEntry(
            key=key,
            path=entry_dir,
            created_at=created_at,
            expires_at=expires_at,
            row_count=table.num_rows,
            page_size=page_size,
            invariants=_freeze_token_mapping(invariant_payload),
        )

    def read_entry(self, entry: CacheEntry) -> pa.Table:
        pages = sorted(entry.path.glob("page-*.parquet"))
        tables = [pq.read_table(page) for page in pages]
        if not tables:
            return pa.table({})
        if len(tables) == 1:
            return tables[0]
        return pa.concat_tables(tables)

    def evict(self, key: CacheKey) -> None:
        entry_dir = self._entry_dir(key)
        if entry_dir.exists():
            shutil.rmtree(entry_dir)

    def scan_entries(self, *, route_slug: str, now: datetime) -> list[CacheEntry]:
        return [
            entry
            for metadata_path in self._root.glob("*/metadata.json")
            if (
                entry := self._entry_from_metadata(
                    metadata_path, route_slug=route_slug, now=now
                )
            )
            is not None
        ]

    def _entry_from_metadata(
        self, metadata_path: Path, *, route_slug: str, now: datetime
    ) -> CacheEntry | None:
        entry_dir = metadata_path.parent
        data = json.loads(metadata_path.read_text())
        created_at = datetime.fromisoformat(data["created_at"])
        expires_at = created_at + timedelta(seconds=data["ttl_seconds"])
        if data.get("route_slug") != route_slug or _handle_expiry(
            expires_at, now, on_expire=lambda: shutil.rmtree(entry_dir)
        ):
            return None
        raw_params = {str(k): v for k, v in data.get("raw_parameters", {}).items()}
        raw_consts = {str(k): v for k, v in data.get("raw_constants", {}).items()}
        invariants = data.get("invariants", {})
        page_size = max(1, int(data.get("page_size") or data.get("row_count", 1) or 1))
        row_count = int(data.get("row_count", 0))
        key = CacheKey(
            route_slug=data["route_slug"],
            parameters=tuple(tuple(item) for item in data.get("parameters", [])),
            constants=tuple(tuple(item) for item in data.get("constants", [])),
            digest=entry_dir.name,
            _raw_parameters=_freeze_str_mapping(raw_params),
            _raw_constants=_freeze_str_mapping(raw_consts),
            _invariant_tokens=_freeze_token_mapping(invariants),
        )
        return CacheEntry(
            key=key,
            path=entry_dir,
            created_at=created_at,
            expires_at=expires_at,
            row_count=row_count,
            page_size=page_size,
            invariants=key.invariant_tokens,
        )


class Cache:
    """DuckDB cache facade that coordinates IO and freshness checks."""

    def __init__(
        self,
        *,
        config: CacheConfig,
        run_query: Callable[[str, Mapping[str, Any], Mapping[str, Any]], Any],
        clock: Callable[[], datetime] | None = None,
        storage: CacheStorage | None = None,
    ) -> None:
        self._config = config
        self._run_query = run_query
        self._clock = clock or _utc_now
        self._storage = storage or CacheStorage(config.storage_root)
        self._invariants = dict(config.invariants)

    def fetch_or_populate(self, key: CacheKey) -> CacheResult:
        now = self._clock()
        cached = self._resolve_cached_entry(key, now)
        if cached is not None:
            entry, from_superset = cached
            return self._render_entry(
                entry,
                key,
                from_cache=True,
                from_superset=from_superset,
            )

        raw_table = _ensure_arrow_table(
            self._run_query(key.route_slug, dict(key.parameter_values), dict(key.constant_values))
        )
        filtered = self._apply_filters(raw_table, key)
        entry = self._storage.write_entry(
            key,
            filtered,
            created_at=now,
            ttl=self._config.ttl,
            page_size=self._config.page_size,
            invariants=key.invariant_tokens,
        )
        return self._render_entry(
            entry,
            key,
            from_cache=False,
            from_superset=False,
            table=filtered,
            pre_filtered=True,
        )

    def _resolve_cached_entry(
        self, key: CacheKey, now: datetime
    ) -> tuple[CacheEntry, bool] | None:
        entry = self._storage.load_entry(key, now)
        if entry is not None:
            return entry, False
        superset_entry = self._find_superset_entry(key, now)
        if superset_entry is not None:
            return superset_entry, True
        return None

    def _apply_filters(self, table: pa.Table, key: CacheKey) -> pa.Table:
        filtered = self._apply_invariant_filters(table, key)
        return self._apply_constant_filters(filtered, key)

    def _apply_invariant_filters(self, table: pa.Table, key: CacheKey) -> pa.Table:
        filtered = table
        for name, tokens in key.invariant_tokens.items():
            filter_config = self._invariants.get(name)
            if filter_config is None:
                continue
            filtered = filter_config.apply(filtered, tokens)
        return filtered

    def _apply_constant_filters(self, table: pa.Table, key: CacheKey) -> pa.Table:
        return reduce(
            self._filter_by_column,
            (
                (column, target)
                for column, target in key.constant_values.items()
                if column not in self._invariants
            ),
            table,
        )

    def _filter_by_column(
        self, table: pa.Table, column_target: tuple[str, Any]
    ) -> pa.Table:
        column, target = column_target
        if column not in table.column_names:
            return table
        predicate = self._constant_predicate(table[column], target)
        if isinstance(predicate, pa.ChunkedArray):
            predicate = predicate.combine_chunks()
        return table.filter(predicate)

    def _constant_predicate(
        self, column_data: pa.Array | pa.ChunkedArray, target: Any
    ) -> pa.Array | pa.ChunkedArray:
        if target is None:
            return pc.is_null(column_data)
        try:
            scalar = pa.scalar(target, type=column_data.type)
        except (pa.ArrowTypeError, pa.ArrowInvalid):
            scalar = pa.scalar(target)
        return pc.equal(column_data, scalar)

    def _render_entry(
        self,
        entry: CacheEntry,
        key: CacheKey,
        *,
        from_cache: bool,
        from_superset: bool,
        table: pa.Table | None = None,
        pre_filtered: bool = False,
    ) -> CacheResult:
        materialized = table if table is not None else self._storage.read_entry(entry)
        filtered = materialized if pre_filtered else self._apply_filters(materialized, key)
        return self._build_result(
            filtered,
            key=key,
            entry=entry,
            from_cache=from_cache,
            from_superset=from_superset,
        )

    def _build_result(
        self,
        table: pa.Table,
        *,
        key: CacheKey,
        entry: CacheEntry,
        from_cache: bool,
        from_superset: bool,
    ) -> CacheResult:
        data = DataHandle(table, entry.page_size)
        return CacheResult(
            data=data,
            from_cache=from_cache,
            from_superset=from_superset,
            entry_digest=entry.key.digest,
            requested_invariants=_freeze_token_mapping(key.invariant_tokens),
            cached_invariants=entry.invariants,
            created_at=entry.created_at,
            expires_at=entry.expires_at,
        )

    def _find_superset_entry(self, key: CacheKey, now: datetime) -> CacheEntry | None:
        if not key.invariant_tokens:
            return None
        candidates = self._storage.scan_entries(route_slug=key.route_slug, now=now)
        requested_params = dict(key.parameter_values)
        requested_constants = dict(key.constant_values)
        requested_invariants = {
            name: set(tokens)
            for name, tokens in key.invariant_tokens.items()
        }
        return next(
            (
                entry
                for entry in candidates
                if self._entry_is_superset(
                    entry,
                    key,
                    requested_params,
                    requested_constants,
                    requested_invariants,
                )
            ),
            None,
        )

    def _non_invariant_equal(
        self, existing_constants: Mapping[str, Any], requested_constants: Mapping[str, Any]
    ) -> bool:
        invariant_names = set(self._invariants.keys())
        existing_filtered = {
            key: value for key, value in existing_constants.items() if key not in invariant_names
        }
        requested_filtered = {
            key: value for key, value in requested_constants.items() if key not in invariant_names
        }
        return existing_filtered == requested_filtered

    def _entry_is_superset(
        self,
        entry: CacheEntry,
        key: CacheKey,
        requested_params: Mapping[str, Any],
        requested_constants: Mapping[str, Any],
        requested_invariants: Mapping[str, set[str]],
    ) -> bool:
        if entry.key.digest == key.digest:
            return False
        if dict(entry.key.parameter_values) != requested_params:
            return False
        if not self._non_invariant_equal(entry.key.constant_values, requested_constants):
            return False
        entry_invariants = {name: set(tokens) for name, tokens in entry.invariants.items()}
        if entry_invariants.keys() != requested_invariants.keys():
            return False
        return all(
            requested_invariants[name].issubset(entry_invariants[name])
            for name in requested_invariants
        )
