"""DuckDB query caching utilities."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableSequence, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

NULL_SENTINEL = "__null__"


def _default_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_timezone(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _serialize_value(value: Any) -> str:
    if value is None:
        return NULL_SENTINEL
    if isinstance(value, (str, int, float, bool)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return json.dumps(value, default=str, separators=(",", ":"), sort_keys=True)


def _normalize_mapping(mapping: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    items = []
    for key in sorted(mapping):
        items.append((key, _serialize_value(mapping[key])))
    return tuple(items)


@dataclass(frozen=True)
class CacheConfig:
    """Configuration for the cache layer."""

    storage_root: Path
    ttl_seconds: int = 3600
    page_size: int = 1024

    def __post_init__(self) -> None:
        if self.page_size <= 0:
            raise ValueError("page_size must be a positive integer")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer")
        object.__setattr__(self, "storage_root", Path(self.storage_root))


@dataclass(frozen=True)
class CacheKey:
    """Canonical representation of a cache key."""

    route_slug: str
    parameters: Tuple[Tuple[str, str], ...]
    constants: Tuple[Tuple[str, str], ...]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", self._build_digest())

    def _build_digest(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.route_slug.encode("utf8"))
        for name, value in (*self.parameters, *self.constants):
            hasher.update(b"\0")
            hasher.update(name.encode("utf8"))
            hasher.update(b"\0")
            hasher.update(value.encode("utf8"))
        return hasher.hexdigest()

    @classmethod
    def from_parts(
        cls,
        route_slug: str,
        parameters: Mapping[str, Any],
        constants: Mapping[str, Any],
    ) -> "CacheKey":
        return cls(route_slug=route_slug, parameters=_normalize_mapping(parameters), constants=_normalize_mapping(constants))


@dataclass(frozen=True)
class CacheEntry:
    key: CacheKey
    created_at: datetime
    page_paths: Tuple[Path, ...]
    row_count: int


@dataclass(frozen=True)
class ParquetIO:
    write_table: Callable[[pa.Table, Path], None]
    read_table: Callable[[Path], pa.Table]

    @staticmethod
    def default() -> "ParquetIO":
        return ParquetIO(writer_default, reader_default)


def writer_default(table: pa.Table, path: Path) -> None:
    pq.write_table(table, path)


def reader_default(path: Path) -> pa.Table:
    return pq.read_table(path)


class CacheError(RuntimeError):
    """Base error for cache operations."""


class Cache:
    """DuckDB result cache backed by Parquet files."""

    def __init__(
        self,
        config: CacheConfig,
        *,
        now: Callable[[], datetime] | None = None,
        parquet_io: ParquetIO | None = None,
    ) -> None:
        self._config = config
        self._now = now or _default_now
        self._parquet_io = parquet_io or ParquetIO.default()
        self._config.storage_root.mkdir(parents=True, exist_ok=True)

    # Public API -----------------------------------------------------
    def cache_dir_for(self, key: CacheKey) -> Path:
        return self._config.storage_root / key.route_slug / key.digest

    def fetch_or_populate(
        self,
        *,
        route_slug: str,
        parameters: Mapping[str, Any],
        constants: Mapping[str, Any],
        runner: Callable[[], Any],
    ) -> pa.Table:
        key = CacheKey.from_parts(route_slug, parameters, constants)
        entry = self._load_entry(key)
        if entry and not self._is_expired(entry):
            table = self._read_entry(entry)
            return self._apply_invariant_filters(table, constants)

        table = self._materialize(runner)
        self._write_entry(key, table)
        return self._apply_invariant_filters(table, constants)

    # Internal helpers -----------------------------------------------
    def _materialize(self, runner: Callable[[], Any]) -> pa.Table:
        result = runner()
        if isinstance(result, pa.Table):
            return result
        if hasattr(result, "to_arrow_table"):
            return result.to_arrow_table()
        if hasattr(result, "fetch_arrow_table"):
            return result.fetch_arrow_table()
        raise CacheError("runner must return a PyArrow table or DuckDB relation")

    def _load_entry(self, key: CacheKey) -> CacheEntry | None:
        entry_dir = self.cache_dir_for(key)
        metadata_path = entry_dir / "entry.json"
        if not metadata_path.exists():
            return None
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            return None
        created_at_raw = metadata.get("created_at")
        row_count = metadata.get("row_count", 0)
        if not created_at_raw:
            return None
        created_at = _ensure_timezone(datetime.fromisoformat(created_at_raw))
        page_paths = tuple(sorted(p for p in entry_dir.glob("page-*.parquet") if p.is_file()))
        return CacheEntry(key=key, created_at=created_at, page_paths=page_paths, row_count=row_count)

    def _is_expired(self, entry: CacheEntry) -> bool:
        expires_at = entry.created_at + timedelta(seconds=self._config.ttl_seconds)
        return self._now() >= expires_at

    def _apply_invariant_filters(self, table: pa.Table, constants: Mapping[str, Any]) -> pa.Table:
        if not constants:
            return table
        mask = None
        for name, value in constants.items():
            if name not in table.column_names:
                raise CacheError(f"invariant column '{name}' missing from cached table")
            if value is None:
                column_mask = pc.is_null(table[name])
            else:
                column_mask = pc.equal(table[name], pa.scalar(value))
            mask = column_mask if mask is None else pc.and_(mask, column_mask)
        if mask is None:
            return table
        return table.filter(mask)

    def _write_entry(self, key: CacheKey, table: pa.Table) -> None:
        entry_dir = self.cache_dir_for(key)
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
        entry_dir.mkdir(parents=True, exist_ok=True)
        total_rows = table.num_rows
        page_size = self._config.page_size
        pages = list(self._iter_pages(table, page_size))
        if not pages:
            pages = [table]
        for index, page in enumerate(pages):
            page_path = entry_dir / f"page-{index:04d}.parquet"
            self._parquet_io.write_table(page, page_path)
        metadata = {
            "created_at": self._now().isoformat(),
            "row_count": total_rows,
            "page_size": page_size,
        }
        (entry_dir / "entry.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))

    def _iter_pages(self, table: pa.Table, page_size: int) -> Iterable[pa.Table]:
        total_rows = table.num_rows
        for offset in range(0, total_rows, page_size):
            yield table.slice(offset, min(page_size, total_rows - offset))

    def _read_entry(self, entry: CacheEntry) -> pa.Table:
        tables: MutableSequence[pa.Table] = []
        for page_path in entry.page_paths:
            tables.append(self._parquet_io.read_table(page_path))
        if not tables:
            return pa.table({})
        if len(tables) == 1:
            return tables[0]
        return pa.concat_tables(tables, promote=True)


__all__ = [
    "Cache",
    "CacheConfig",
    "CacheEntry",
    "CacheError",
    "CacheKey",
    "NULL_SENTINEL",
    "ParquetIO",
]
