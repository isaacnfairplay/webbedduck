"""DuckDB query result caching utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Tuple

import shutil

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

NULL_SENTINEL = "__null__"


class CacheError(RuntimeError):
    """Base class for cache related errors."""


class CacheConsistencyError(CacheError):
    """Raised when on-disk cache entries are inconsistent with expectations."""


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _ensure_path(path: Path) -> Path:
    return Path(path)


def _sort_key(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True)


def _normalize_value(value: Any) -> Any:
    if value is None:
        return NULL_SENTINEL
    if isinstance(value, Mapping):
        return tuple((str(k), _normalize_value(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        normalised_items = tuple(_normalize_value(item) for item in value)
        return tuple(sorted(normalised_items, key=_sort_key))
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return value


def _normalize_mapping(mapping: Optional[Mapping[str, Any]]) -> Tuple[Tuple[str, Any], ...]:
    if not mapping:
        return tuple()
    return tuple((str(key), _normalize_value(mapping[key])) for key in sorted(mapping))


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _hash_components(route: str, parameters: Tuple[Tuple[str, Any], ...], constants: Tuple[Tuple[str, Any], ...]) -> str:
    payload = {
        "route": route,
        "parameters": {key: _jsonable(value) for key, value in parameters},
        "constants": {key: _jsonable(value) for key, value in constants},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CacheKey:
    """A stable representation of cache identity."""

    route: str
    parameters: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    constants: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", _hash_components(self.route, self.parameters, self.constants))

    @property
    def route_slug(self) -> str:
        return self.route.replace("/", "-")

    @property
    def parameters_dict(self) -> MutableMapping[str, Any]:
        return {key: value for key, value in self.parameters}

    @property
    def constants_dict(self) -> MutableMapping[str, Any]:
        return {key: value for key, value in self.constants}

    @classmethod
    def from_parts(
        cls,
        *,
        route: str,
        parameters: Optional[Mapping[str, Any]] = None,
        constants: Optional[Mapping[str, Any]] = None,
    ) -> "CacheKey":
        return cls(route=route, parameters=_normalize_mapping(parameters), constants=_normalize_mapping(constants))

    @classmethod
    def from_serialized(cls, payload: Mapping[str, Any]) -> "CacheKey":
        return cls.from_parts(
            route=str(payload["route"]),
            parameters=payload.get("parameters", {}),
            constants=payload.get("constants", {}),
        )

    def to_serialized(self) -> Mapping[str, Any]:
        return {
            "route": self.route,
            "parameters": {key: _jsonable(value) for key, value in self.parameters},
            "constants": {key: _jsonable(value) for key, value in self.constants},
            "digest": self.digest,
        }

    def can_satisfy(self, requested: "CacheKey") -> bool:
        if self.route != requested.route:
            return False
        if self.parameters != requested.parameters:
            return False
        stored_constants = self.constants_dict
        requested_constants = requested.constants_dict
        all_keys = set(stored_constants) | set(requested_constants)
        for key in all_keys:
            stored_value = stored_constants.get(key, NULL_SENTINEL)
            requested_value = requested_constants.get(key, NULL_SENTINEL)
            if requested_value == stored_value:
                continue
            if stored_value == NULL_SENTINEL and requested_value != NULL_SENTINEL:
                continue
            if requested_value == NULL_SENTINEL and stored_value != NULL_SENTINEL:
                return False
            return False
        return True


@dataclass(frozen=True)
class CacheConfig:
    storage_root: Path
    ttl: timedelta = timedelta(minutes=5)
    page_size: int = 1024

    def __post_init__(self) -> None:
        if self.page_size <= 0:
            msg = f"page_size must be positive, received {self.page_size}"
            raise ValueError(msg)
        if self.ttl <= timedelta(0):
            msg = "ttl must be positive"
            raise ValueError(msg)
        object.__setattr__(self, "storage_root", _ensure_path(self.storage_root))


@dataclass(frozen=True)
class CacheEntry:
    key: CacheKey
    directory: Path
    created_at: datetime
    page_paths: Tuple[Path, ...]
    row_count: int

    @property
    def metadata_path(self) -> Path:
        return self.directory / "metadata.json"


class Cache:
    """A Parquet-backed cache for DuckDB query results."""

    def __init__(
        self,
        config: CacheConfig,
        *,
        clock: Optional[Callable[[], datetime]] = None,
        read_page: Optional[Callable[[Path], pa.Table]] = None,
        write_page: Optional[Callable[[pa.Table, Path], None]] = None,
    ) -> None:
        self._config = config
        self._clock = clock or _utcnow
        self._read_page = read_page or (lambda path: pq.read_table(path))
        self._write_page = write_page or (lambda table, path: pq.write_table(table, path))
        self._config.storage_root.mkdir(parents=True, exist_ok=True)

    def fetch_or_populate(self, key: CacheKey, runner: Callable[[], pa.Table]) -> pa.Table:
        now = self._clock()
        entry = self._locate_entry(key, now)
        if entry is not None:
            return self._materialize(entry, key)

        raw = runner()
        table = _coerce_table(raw)
        entry = self._persist_entry(key, table, now)
        return self._materialize(entry, key)

    def _entry_directory(self, key: CacheKey) -> Path:
        return self._config.storage_root / key.route_slug / key.digest

    def _persist_entry(self, key: CacheKey, table: pa.Table, created_at: datetime) -> CacheEntry:
        directory = self._entry_directory(key)
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)

        pages = list(_slice_table(table, self._config.page_size))
        if not pages:
            pages = [table]
        page_paths = []
        for index, page in enumerate(pages):
            page_path = directory / f"page-{index:04d}.parquet"
            self._write_page(page, page_path)
            page_paths.append(page_path)

        metadata = {
            "key": key.to_serialized(),
            "created_at": created_at.isoformat(),
            "page_size": self._config.page_size,
            "row_count": table.num_rows,
        }
        (directory / "metadata.json").write_text(json.dumps(metadata, sort_keys=True))
        return CacheEntry(key=key, directory=directory, created_at=created_at, page_paths=tuple(page_paths), row_count=table.num_rows)

    def _locate_entry(self, key: CacheKey, now: datetime) -> Optional[CacheEntry]:
        direct_dir = self._entry_directory(key)
        entry = self._load_entry(direct_dir)
        if entry and entry.key.digest == key.digest and not self._is_expired(entry, now):
            return entry

        route_dir = self._config.storage_root / key.route_slug
        if not route_dir.exists():
            return None

        for child in route_dir.iterdir():
            if child == direct_dir:
                continue
            candidate = self._load_entry(child)
            if candidate is None:
                continue
            if self._is_expired(candidate, now):
                continue
            if candidate.key.can_satisfy(key):
                return candidate
        return None

    def _load_entry(self, directory: Path) -> Optional[CacheEntry]:
        metadata_path = directory / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as exc:
            raise CacheConsistencyError(f"invalid metadata at {metadata_path}") from exc
        key = CacheKey.from_serialized(payload["key"])
        created_at = datetime.fromisoformat(payload["created_at"])  # type: ignore[arg-type]
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        page_paths = tuple(sorted(directory.glob("page-*.parquet")))
        return CacheEntry(
            key=key,
            directory=directory,
            created_at=created_at,
            page_paths=page_paths,
            row_count=int(payload.get("row_count", 0)),
        )

    def _is_expired(self, entry: CacheEntry, now: datetime) -> bool:
        return now - entry.created_at > self._config.ttl

    def _materialize(self, stored_entry: CacheEntry, requested_key: CacheKey) -> pa.Table:
        tables: list[pa.Table] = []
        for page_path in stored_entry.page_paths:
            tables.append(self._read_page(page_path))
        if not tables:
            raise CacheConsistencyError(f"no parquet pages for {stored_entry.directory}")
        table = tables[0] if len(tables) == 1 else pa.concat_tables(tables, promote=True)
        return _filter_for_request(table, stored_entry.key, requested_key)


def _slice_table(table: pa.Table, page_size: int) -> Iterable[pa.Table]:
    total = table.num_rows
    if total == 0:
        yield table
        return
    offset = 0
    while offset < total:
        yield table.slice(offset, page_size)
        offset += page_size


def _filter_for_request(table: pa.Table, stored_key: CacheKey, requested_key: CacheKey) -> pa.Table:
    stored_constants = stored_key.constants_dict
    requested_constants = requested_key.constants_dict
    filters: list[Tuple[str, Any]] = []
    for key, requested_value in requested_constants.items():
        if requested_value == NULL_SENTINEL:
            if stored_constants.get(key, NULL_SENTINEL) != NULL_SENTINEL:
                raise CacheConsistencyError(
                    f"request expected unconstrained {key!r} but cached entry is constrained"
                )
            continue
        stored_value = stored_constants.get(key, NULL_SENTINEL)
        if stored_value == requested_value:
            continue
        if stored_value == NULL_SENTINEL:
            filters.append((key, requested_value))
            continue
        raise CacheConsistencyError(
            f"cached entry constrained {key!r} to {stored_value!r}, incompatible with request"
        )
    if not filters:
        return table

    mask = None
    for column_name, expected_value in filters:
        column_mask = pc.equal(table[column_name], expected_value)
        mask = column_mask if mask is None else pc.and_(mask, column_mask)
    return table.filter(mask)


def _coerce_table(result: Any) -> pa.Table:
    if isinstance(result, pa.Table):
        return result
    if isinstance(result, pa.RecordBatchReader):
        return result.read_all()
    raise CacheError("runner must return a pyarrow.Table or RecordBatchReader")

