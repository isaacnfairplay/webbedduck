"""DuckDB query cache with Parquet-backed storage."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, ClassVar, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


NULL_SENTINEL = "__null__"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _decode_null(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value == NULL_SENTINEL:
        return None
    return value


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, default=str, sort_keys=True)


def _normalize_items(mapping: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not mapping:
        return tuple()
    normalized: list[tuple[str, str]] = []
    for key in sorted(mapping.keys()):
        value = mapping[key]
        encoded = NULL_SENTINEL if value is None else _stringify(value)
        normalized.append((str(key), encoded))
    return tuple(normalized)


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


def _apply_invariants(table: pa.Table, constants: Mapping[str, Any]) -> pa.Table:
    if not constants:
        return table
    mask: pa.ChunkedArray | None = None
    for column, target in constants.items():
        if column not in table.column_names:
            continue
        column_data = table[column]
        if target is None:
            predicate = pc.is_null(column_data)
        else:
            try:
                scalar = pa.scalar(target, type=column_data.type)
            except (pa.ArrowTypeError, pa.ArrowInvalid):
                scalar = pa.scalar(target)
            predicate = pc.equal(column_data, scalar)
        mask = predicate if mask is None else pc.and_(mask, predicate)
    if mask is None:
        return table
    if isinstance(mask, pa.ChunkedArray):
        mask = mask.combine_chunks()
    return table.filter(mask)


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for cache layout and freshness."""

    storage_root: Path = Path(".cache")
    ttl: timedelta = timedelta(minutes=10)
    page_size: int = 500

    def __post_init__(self) -> None:
        object.__setattr__(self, "storage_root", Path(self.storage_root))
        if self.page_size <= 0:
            msg = "page_size must be a positive integer"
            raise ValueError(msg)
        if self.ttl <= timedelta(0):
            msg = "ttl must be positive"
            raise ValueError(msg)


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

    @classmethod
    def from_parts(
        cls,
        route_slug: str,
        *,
        parameters: Mapping[str, Any] | None = None,
        constants: Mapping[str, Any] | None = None,
    ) -> "CacheKey":
        params_copy = {str(key): _decode_null(value) for key, value in (parameters or {}).items()}
        const_copy = {str(key): _decode_null(value) for key, value in (constants or {}).items()}
        normalized_params = _normalize_items(params_copy)
        normalized_constants = _normalize_items(const_copy)
        digest = _compute_digest(route_slug, normalized_params, normalized_constants)
        return cls(
            route_slug=route_slug,
            parameters=normalized_params,
            constants=normalized_constants,
            digest=digest,
            _raw_parameters=MappingProxyType(params_copy),
            _raw_constants=MappingProxyType(const_copy),
        )

    @property
    def parameter_values(self) -> Mapping[str, Any]:
        return self._raw_parameters

    @property
    def constant_values(self) -> Mapping[str, Any]:
        return self._raw_constants


@dataclass(frozen=True)
class CacheEntry:
    """Metadata describing a cached materialisation on disk."""

    key: CacheKey
    path: Path
    created_at: datetime
    expires_at: datetime
    row_count: int


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
        ttl_seconds = data["ttl_seconds"]
        expires_at = created_at + timedelta(seconds=ttl_seconds)
        if now >= expires_at:
            self.evict(key)
            return None
        return CacheEntry(
            key=key,
            path=entry_dir,
            created_at=created_at,
            expires_at=expires_at,
            row_count=data["row_count"],
        )

    def write_entry(
        self,
        key: CacheKey,
        table: pa.Table,
        *,
        created_at: datetime,
        ttl: timedelta,
        page_size: int,
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
        metadata = {
            "created_at": created_at.isoformat(),
            "ttl_seconds": ttl.total_seconds(),
            "row_count": table.num_rows,
        }
        metadata_path.write_text(json.dumps(metadata, sort_keys=True))
        expires_at = created_at + ttl
        return CacheEntry(
            key=key,
            path=entry_dir,
            created_at=created_at,
            expires_at=expires_at,
            row_count=table.num_rows,
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

    def fetch_or_populate(self, key: CacheKey) -> pa.Table:
        now = self._clock()
        entry = self._storage.load_entry(key, now)
        if entry is not None:
            table = self._storage.read_entry(entry)
            return _apply_invariants(table, key.constant_values)

        raw_table = _ensure_arrow_table(
            self._run_query(key.route_slug, dict(key.parameter_values), dict(key.constant_values))
        )
        filtered = _apply_invariants(raw_table, key.constant_values)
        entry = self._storage.write_entry(
            key,
            filtered,
            created_at=now,
            ttl=self._config.ttl,
            page_size=self._config.page_size,
        )
        stored = self._storage.read_entry(entry)
        return _apply_invariants(stored, key.constant_values)
