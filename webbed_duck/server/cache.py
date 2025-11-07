"""DuckDB-backed result caching helpers."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

NULL_SENTINEL = "__null__"
_METADATA_FILENAME = "metadata.json"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_value(value: object) -> str:
    if value is None:
        return NULL_SENTINEL
    if isinstance(value, str):
        if value == NULL_SENTINEL:
            return value
        return value
    return json.dumps(value, sort_keys=True)


def _normalize_mapping(mapping: Mapping[str, object] | None) -> tuple[tuple[str, str], ...]:
    if not mapping:
        return tuple()
    items: list[tuple[str, str]] = []
    for key in sorted(mapping):
        items.append((key, _normalize_value(mapping[key])))
    return tuple(items)


@dataclass(frozen=True)
class NormalizedKey:
    route_slug: str
    parameters: tuple[tuple[str, str], ...]
    constants: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CacheKey:
    """Deterministic cache keys."""

    normalized: NormalizedKey

    @classmethod
    def build(
        cls,
        route_slug: str,
        *,
        parameters: Mapping[str, object] | None = None,
        constants: Mapping[str, object] | None = None,
    ) -> "CacheKey":
        normalized = NormalizedKey(
            route_slug=route_slug,
            parameters=_normalize_mapping(parameters),
            constants=_normalize_mapping(constants),
        )
        return cls(normalized=normalized)

    @property
    def digest(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.normalized.route_slug.encode("utf-8"))
        for label, pairs in ("parameters", self.normalized.parameters), ("constants", self.normalized.constants):
            hasher.update(label.encode("utf-8"))
            for key, value in pairs:
                hasher.update(key.encode("utf-8"))
                hasher.update(value.encode("utf-8"))
        return hasher.hexdigest()


@dataclass(frozen=True)
class CacheConfig:
    ttl: timedelta
    page_size: int
    storage_root: Path

    def ensure_root(self) -> Path:
        self.storage_root.mkdir(parents=True, exist_ok=True)
        return self.storage_root


@dataclass(frozen=True)
class CacheEntry:
    key: CacheKey
    directory: Path
    created_at: datetime
    row_count: int
    page_paths: tuple[Path, ...]


class CacheIO:
    def __init__(self, root: Path):
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def directory_for(self, key: CacheKey) -> Path:
        return self._root / key.digest

    def load(self, key: CacheKey) -> CacheEntry | None:
        directory = self.directory_for(key)
        metadata_path = directory / _METADATA_FILENAME
        if not metadata_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text())
        created_at = datetime.fromisoformat(metadata["created_at"])
        page_names = metadata.get("pages", [])
        page_paths = tuple(directory / name for name in page_names)
        if not all(path.exists() for path in page_paths):
            return None
        return CacheEntry(
            key=key,
            directory=directory,
            created_at=created_at,
            row_count=metadata.get("row_count", 0),
            page_paths=page_paths,
        )

    def evict(self, key: CacheKey) -> None:
        directory = self.directory_for(key)
        if not directory.exists():
            return
        for child in directory.iterdir():
            if child.is_file():
                child.unlink()
        directory.rmdir()

    def write_entry(
        self,
        key: CacheKey,
        table: pa.Table,
        *,
        created_at: datetime,
        page_size: int,
    ) -> CacheEntry:
        directory = self.directory_for(key)
        if directory.exists():
            for child in directory.iterdir():
                if child.is_file():
                    child.unlink()
        else:
            directory.mkdir(parents=True)

        pages: list[Path] = []
        total_rows = table.num_rows
        if total_rows == 0:
            empty_path = directory / "page-00000.parquet"
            pq.write_table(table, empty_path)
            pages.append(empty_path)
        else:
            start = 0
            index = 0
            while start < total_rows:
                batch = table.slice(start, page_size)
                page_path = directory / f"page-{index:05d}.parquet"
                pq.write_table(batch, page_path)
                pages.append(page_path)
                start += page_size
                index += 1

        metadata_path = directory / _METADATA_FILENAME
        metadata_path.write_text(
            json.dumps(
                {
                    "created_at": created_at.isoformat(),
                    "row_count": total_rows,
                    "pages": [path.name for path in pages],
                },
                sort_keys=True,
            )
        )
        return CacheEntry(
            key=key,
            directory=directory,
            created_at=created_at,
            row_count=total_rows,
            page_paths=tuple(pages),
        )

    def read_table(self, entry: CacheEntry) -> pa.Table:
        tables = [pq.read_table(path) for path in entry.page_paths]
        if not tables:
            return pa.table({})
        return pa.concat_tables(tables)


def _apply_filters(table: pa.Table, filters: Mapping[str, object] | None) -> pa.Table:
    if not filters:
        return table
    mask = None
    for column, requested in filters.items():
        actual_column = table[column]
        normalized = _normalize_filter_value(requested)
        if normalized is None:
            predicate = pc.is_null(actual_column)
        else:
            predicate = pc.equal(actual_column, pa.scalar(normalized))
        mask = predicate if mask is None else pc.and_(mask, predicate)
    if mask is None:
        return table
    return table.filter(mask)


def _normalize_filter_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value == NULL_SENTINEL:
        return None
    if isinstance(value, str):
        return value
    return str(value)


class Cache:
    def __init__(
        self,
        *,
        config: CacheConfig,
        clock: Callable[[], datetime] | None = None,
        io: CacheIO | None = None,
    ) -> None:
        self._config = config
        self._clock = clock or _utcnow
        root = config.ensure_root()
        self._io = io or CacheIO(root)

    def _is_expired(self, entry: CacheEntry) -> bool:
        expires_at = entry.created_at + self._config.ttl
        return self._clock() >= expires_at

    def fetch_or_populate(
        self,
        *,
        key: CacheKey,
        runner: Callable[[CacheKey], pa.Table],
        invariant_filters: Mapping[str, object] | None = None,
    ) -> pa.Table:
        entry = self._io.load(key)
        if entry and self._is_expired(entry):
            self._io.evict(key)
            entry = None

        if entry is None:
            table = runner(key)
            if not isinstance(table, pa.Table):
                raise TypeError("cache runners must return a pyarrow.Table")
            entry = self._io.write_entry(key, table, created_at=self._clock(), page_size=self._config.page_size)
        else:
            table = self._io.read_table(entry)

        return _apply_filters(table, invariant_filters)
