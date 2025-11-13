"""DuckDB query cache with Parquet-backed storage."""

from __future__ import annotations

import contextlib
import json
import io
import shutil
import threading
from dataclasses import dataclass, field
from functools import partial
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from collections import defaultdict
from typing import Any, BinaryIO, Callable, ClassVar, Iterable, Iterator, Mapping, TextIO

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pacsv

from .cache_support import (
    NULL_SENTINEL,
    CacheEntry,
    CacheEntryMetadata,
    InvariantFilter,
    compute_digest,
    decode_null,
    encode_constants,
    freeze_str_mapping,
    freeze_token_mapping,
    normalize_items,
)
from .cache_resolution import CacheFilterPipeline

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _handle_expiry(
    expires_at: datetime, now: datetime, *, on_expire: Callable[[], None]
) -> bool:
    if now < expires_at:
        return False
    on_expire()
    return True


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


def _iter_page_tables(table: pa.Table, page_size: int) -> Iterator[pa.Table]:
    yielded = False
    for batch in table.to_batches(max_chunksize=page_size):
        yielded = True
        yield pa.Table.from_batches([batch])
    if not yielded:
        yield table


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
        object.__setattr__(self, "invariants", freeze_str_mapping(self.invariants))


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
        params_copy = {
            str(key): decode_null(value)
            for key, value in (parameters or {}).items()
        }
        const_copy = {
            str(key): decode_null(value)
            for key, value in (constants or {}).items()
        }
        normalized_params = normalize_items(params_copy)
        filters = invariant_filters or {}
        normalized_constants_tuple, invariant_tokens = encode_constants(
            const_copy, filters
        )
        digest_constants = tuple(
            item for item in normalized_constants_tuple if item[0] not in invariant_tokens
        )
        digest = compute_digest(route_slug, normalized_params, digest_constants)
        non_invariant_constants = {
            name: value for name, value in const_copy.items() if name not in filters
        }
        return cls(
            route_slug=route_slug,
            parameters=normalized_params,
            constants=digest_constants,
            digest=digest,
            _raw_parameters=freeze_str_mapping(params_copy),
            _raw_constants=freeze_str_mapping(non_invariant_constants),
            _invariant_tokens=freeze_token_mapping(invariant_tokens),
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


# Shared iterators and encoders for streaming adapters.

def _csv_payload(table: pa.Table, *, include_header: bool) -> str:
    sink = pa.BufferOutputStream()
    pacsv.write_csv(
        table,
        sink,
        write_options=pacsv.WriteOptions(include_header=include_header),
    )
    return sink.getvalue().to_pybytes().decode("utf-8")


class _IterTextIO(io.TextIOBase):
    """Lazy text stream that pulls chunks from an iterator on demand."""

    def __init__(self, chunks: Iterable[str]) -> None:
        self._iterator: Iterator[str] = iter(chunks)
        self._buffer: str = ""
        self._closed = False

    def readable(self) -> bool:  # pragma: no cover - trivial
        return True

    def _read_all(self) -> str:
        payload = self._buffer + "".join(self._iterator)
        self._buffer = ""
        self._iterator = iter(())
        return payload

    def read(self, size: int = -1) -> str:  # pragma: no cover - exercised via callers
        self._ensure_open()
        if size is None or size < 0:
            return self._read_all()
        remaining = size
        pieces: list[str] = []
        if self._buffer:
            head = self._buffer[:remaining]
            pieces.append(head)
            self._buffer = self._buffer[len(head):]
            remaining -= len(head)
        while remaining > 0:
            try:
                chunk = next(self._iterator)
            except StopIteration:
                break
            if remaining < len(chunk):
                pieces.append(chunk[:remaining])
                self._buffer = chunk[remaining:]
                break
            pieces.append(chunk)
            remaining -= len(chunk)
        return "".join(pieces)

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("I/O operation on closed file.")

    def close(self) -> None:  # pragma: no cover - trivial
        if not self._closed:
            self._iterator = iter(())
            self._buffer = ""
            self._closed = True
        super().close()


class _BaseAdapter:
    """Context manager factory that exposes a table in a target format."""

    format: ClassVar[str]

    @contextlib.contextmanager
    def open(
        self, handle: "DataHandle", page: int | None
    ):  # pragma: no cover - overridden by subclasses
        raise NotImplementedError


class _TextAdapter(_BaseAdapter):
    chunker: ClassVar[Callable[[pa.Table], Iterable[str]]]

    @contextlib.contextmanager
    def open(self, handle: "DataHandle", page: int | None):
        table = handle.as_arrow(page)
        stream = _IterTextIO(self.chunker(table))
        try:
            yield stream
        finally:
            stream.close()


class _ArrowAdapter(_BaseAdapter):
    format = "arrow"

    @contextlib.contextmanager
    def open(self, handle: "DataHandle", page: int | None):
        yield handle.as_arrow(page)


class _ParquetAdapter(_BaseAdapter):
    format = "parquet"

    @contextlib.contextmanager
    def open(self, handle: "DataHandle", page: int | None):
        if page is None or handle._entry_path is None:
            sink = pa.BufferOutputStream()
            pq.write_table(handle.as_arrow(page), sink)
            buffer = io.BytesIO(sink.getvalue().to_pybytes())
            try:
                yield buffer
            finally:
                buffer.close()
            return

        file_path = handle._page_path(page)
        with file_path.open("rb") as stream:
            yield stream


def _iter_csv_chunks(table: pa.Table) -> Iterator[str]:
    if table.num_rows == 0:
        payload = _csv_payload(table, include_header=True)
        if payload:
            yield payload
        return

    include_header = True
    for batch in table.to_batches(max_chunksize=table.num_rows or None):
        payload = _csv_payload(
            pa.Table.from_batches([batch]), include_header=include_header
        )
        include_header = False
        if payload:
            yield payload


class _CsvAdapter(_TextAdapter):
    format = "csv"
    chunker: ClassVar[Callable[[pa.Table], Iterable[str]]] = staticmethod(_iter_csv_chunks)


def _iter_json_records(table: pa.Table) -> Iterator[str]:
    for batch in table.to_batches(max_chunksize=table.num_rows or None):
        for record in batch.to_pylist():
            yield json.dumps(record, default=str)


def _iter_json_payloads(lines: bool, table: pa.Table) -> Iterator[str]:
    records = _iter_json_records(table)
    if lines:
        for record in records:
            yield record + "\n"
        return
    yield "["
    first = True
    for record in records:
        if not first:
            yield ","
        yield record
        first = False
    yield "]"


class _JsonAdapter(_TextAdapter):
    format = "json"
    chunker: ClassVar[Callable[[pa.Table], Iterable[str]]] = staticmethod(partial(_iter_json_payloads, False))


class _JsonLinesAdapter(_TextAdapter):
    format = "jsonl"
    chunker: ClassVar[Callable[[pa.Table], Iterable[str]]] = staticmethod(partial(_iter_json_payloads, True))


@dataclass(frozen=True)
class DataHandle:
    """Format-aware accessor for cached query payloads.

    Arrow payloads should be retrieved via :meth:`as_arrow`, which returns a
    :class:`pyarrow.Table` without requiring context management. Binary formats
    such as Parquet yield ``BinaryIO`` handles, while textual formats return a
    ``TextIO`` stream.
    """

    _table: pa.Table
    _page_size: int
    _entry_path: Path | None = None
    _ADAPTERS: ClassVar[dict[str, _BaseAdapter]] = {
        adapter.format: adapter
        for adapter in (
            _ArrowAdapter(),
            _ParquetAdapter(),
            _CsvAdapter(),
            _JsonAdapter(),
            _JsonLinesAdapter(),
        )
    }

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
    def page_count(self) -> int:
        if self.row_count == 0:
            return 0
        return math.ceil(self.row_count / self._page_size)

    @property
    def schema(self) -> pa.Schema:
        return self._table.schema

    @property
    def formats(self) -> tuple[str, ...]:
        return tuple(sorted(self._ADAPTERS))

    @property
    def table(self) -> pa.Table:
        return self._table

    def as_arrow(self, page: int | None = None) -> pa.Table:
        """Return a :class:`pyarrow.Table` slice without context management."""
        return _slice_table(self._table, self._page_size, page)

    @contextlib.contextmanager
    def open(
        self, format: str, page: int | None = None
    ) -> Iterator[BinaryIO | TextIO | pa.Table]:
        """Open a page in the requested format.

        ``"parquet"`` returns a binary file handle, while ``"csv"``, ``"json"``,
        and ``"jsonl"`` yield :class:`io.TextIOBase` instances that stream
        encoded rows. ``"arrow"`` is equivalent to :meth:`as_arrow` and returns a
        :class:`pyarrow.Table`.
        """

        adapter = self._ADAPTERS.get(format)
        if adapter is None:
            raise ValueError(
                f"Unsupported format '{format}'. Expected one of {self.formats}."
            )
        with adapter.open(self, page) as resource:
            yield resource

    def _page_path(self, page: int) -> Path:
        if self._entry_path is None:
            raise ValueError("Parquet pages are unavailable for in-memory results")
        if page < 0:
            raise ValueError("Page numbers are 0-indexed")
        if page >= self.page_count:
            raise ValueError("Requested page exceeds available rows")
        return self._entry_path / f"page-{page:05d}.parquet"


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

    _DELEGATED: ClassVar[tuple[str, ...]] = (
        "row_count",
        "schema",
        "page_size",
        "page_count",
        "formats",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_invariants", freeze_token_mapping(self.requested_invariants))
        object.__setattr__(self, "cached_invariants", freeze_token_mapping(self.cached_invariants))

    def __getattr__(self, name: str) -> Any:
        if name in self._DELEGATED:
            return getattr(self.data, name)
        raise AttributeError(name)

    @property
    def table(self) -> pa.Table:
        return self.data.as_arrow()

    def to_pylist(self, page: int | None = None) -> list[dict[str, Any]]:
        return self.data.as_arrow(page).to_pylist()

    def as_arrow(self, page: int | None = None) -> pa.Table:
        """Return the requested page as a :class:`pyarrow.Table`."""
        return self.data.as_arrow(page)

    def open(
        self, format: str, page: int | None = None
    ) -> Iterator[BinaryIO | TextIO | pa.Table]:
        """Convenience alias for :meth:`DataHandle.open`."""

        return self.data.open(format, page=page)


# Backwards-compatible alias for downstream imports.
CacheResult = ResponseEnvelope


@dataclass(frozen=True)
class CacheMetadataSummary:
    """Lightweight snapshot of an on-disk cache entry."""

    digest: str
    route_slug: str
    parameters: Mapping[str, Any]
    constants: Mapping[str, Any]
    row_count: int
    page_size: int
    page_count: int
    created_at: datetime
    expires_at: datetime
    from_cache: bool
    from_superset: bool
    formats: tuple[str, ...]
    requested_invariants: Mapping[str, tuple[str, ...]]
    cached_invariants: Mapping[str, tuple[str, ...]]


class CacheStorage:
    """Disk-backed storage abstraction used by the cache."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _entry_dir(self, key: CacheKey) -> Path:
        return self._root / key.digest

    def load_entry(self, key: CacheKey, now: datetime) -> CacheEntry | None:
        entry_dir = self._entry_dir(key)
        metadata_path = self._metadata_path(entry_dir)
        if not metadata_path.exists():
            return None
        metadata = self._read_metadata(metadata_path)
        if _handle_expiry(metadata.expires_at, now, on_expire=lambda: self.evict(key)):
            return None
        return metadata.to_cache_entry(key=key, path=entry_dir)

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
        self._prepare_entry_dir(entry_dir)
        for index, page_table in enumerate(_iter_page_tables(table, page_size)):
            pq.write_table(page_table, entry_dir / f"page-{index:05d}.parquet")
        invariant_payload = key.invariant_tokens if invariants is None else invariants
        metadata = CacheEntryMetadata.for_entry(
            key=key,
            table=table,
            created_at=created_at,
            ttl=ttl,
            page_size=page_size,
            invariants=invariant_payload,
        )
        self._write_metadata(entry_dir, metadata)
        return metadata.to_cache_entry(key=key, path=entry_dir)

    def read_entry(self, entry: CacheEntry) -> pa.Table:
        tables = [pq.read_table(page) for page in sorted(entry.path.glob("page-*.parquet"))]
        if not tables:
            return pa.table({})
        return tables[0] if len(tables) == 1 else pa.concat_tables(tables)

    def evict(self, key: CacheKey) -> None:
        if (entry_dir := self._entry_dir(key)).exists():
            shutil.rmtree(entry_dir)

    def scan_entries(self, *, route_slug: str, now: datetime) -> list[CacheEntry]:
        return [
            entry
            for metadata_path in self._root.glob("*/metadata.json")
            if (entry := self._entry_from_metadata(metadata_path, route_slug=route_slug, now=now))
        ]

    def _entry_from_metadata(
        self, metadata_path: Path, *, route_slug: str, now: datetime
    ) -> CacheEntry | None:
        entry_dir = metadata_path.parent
        metadata = self._read_metadata(metadata_path)
        if metadata.route_slug != route_slug:
            return None
        if self._evict_if_expired(entry_dir, metadata.expires_at, now):
            return None
        key = metadata.to_cache_key(digest=entry_dir.name)
        return metadata.to_cache_entry(key=key, path=entry_dir)

    def _evict_if_expired(
        self, entry_dir: Path, expires_at: datetime, now: datetime
    ) -> bool:
        return _handle_expiry(expires_at, now, on_expire=lambda: shutil.rmtree(entry_dir))

    def _metadata_path(self, entry_dir: Path) -> Path:
        return entry_dir / "metadata.json"

    def _prepare_entry_dir(self, entry_dir: Path) -> None:
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
        entry_dir.mkdir(parents=True, exist_ok=True)

    def _read_metadata(self, metadata_path: Path) -> CacheEntryMetadata:
        return CacheEntryMetadata.from_dict(json.loads(metadata_path.read_text()))

    def _write_metadata(self, entry_dir: Path, metadata: CacheEntryMetadata) -> None:
        metadata_path = self._metadata_path(entry_dir)
        metadata_path.write_text(json.dumps(metadata.to_record(), sort_keys=True))


def peek_metadata(
    storage: CacheStorage, digest: str, *, now: datetime | None = None
) -> CacheMetadataSummary | None:
    """Load cache metadata without touching Parquet pages.

    Returns ``None`` when the requested entry is missing or expired.
    """

    entry_dir = storage._root / digest
    metadata_path = storage._metadata_path(entry_dir)
    if not metadata_path.exists():
        return None
    metadata = storage._read_metadata(metadata_path)
    moment = now or _utc_now()
    if storage._evict_if_expired(entry_dir, metadata.expires_at, moment):
        return None
    page_count = 0 if metadata.row_count == 0 else math.ceil(metadata.row_count / metadata.page_size)
    return CacheMetadataSummary(
        digest=digest,
        route_slug=metadata.route_slug,
        parameters=metadata.raw_parameters,
        constants=metadata.raw_constants,
        row_count=metadata.row_count,
        page_size=metadata.page_size,
        page_count=page_count,
        created_at=metadata.created_at,
        expires_at=metadata.expires_at,
        from_cache=True,
        from_superset=False,
        formats=tuple(sorted(DataHandle._ADAPTERS)),
        requested_invariants=freeze_token_mapping({}),
        cached_invariants=metadata.invariants,
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
        self._filters = CacheFilterPipeline(self._invariants)
        self._entry_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._entry_locks_guard = threading.Lock()

    def fetch_or_populate(self, key: CacheKey) -> CacheResult:
        now = self._clock()
        cached = self._resolve_cached_entry(key, now)
        if cached is not None:
            return self._materialise_entry(
                cached,
                key=key,
            )

        return self._populate_entry(key, now)

    def _resolve_cached_entry(
        self, key: CacheKey, now: datetime
    ) -> CacheEntry | None:
        if (entry := self._storage.load_entry(key, now)) is not None:
            return entry
        return None

    def _lock_for_digest(self, digest: str) -> threading.Lock:
        with self._entry_locks_guard:
            return self._entry_locks[digest]

    def _materialise_entry(
        self,
        entry: CacheEntry,
        *,
        key: CacheKey,
    ) -> CacheResult:
        table = self._storage.read_entry(entry)
        filtered = self._filters.apply(table, key)
        return self._build_result(
            filtered,
            key=key,
            entry=entry,
            from_cache=True,
            from_superset=False,
        )

    def _populate_entry(self, key: CacheKey, now: datetime) -> CacheResult:
        entry_lock = self._lock_for_digest(key.digest)
        with entry_lock:
            now = self._clock()
            if (cached := self._resolve_cached_entry(key, now)) is not None:
                return self._materialise_entry(cached, key=key)
            raw_table = _ensure_arrow_table(
                self._run_query(
                    key.route_slug, dict(key.parameter_values), dict(key.constant_values)
                )
            )
            base_table = self._filters.apply(raw_table, key, include_invariants=False)
            entry = self._storage.write_entry(
                key,
                base_table,
                created_at=now,
                ttl=self._config.ttl,
                page_size=self._config.page_size,
                invariants={},
            )
        stored = self._storage.read_entry(entry)
        stored_filtered = self._filters.apply(stored, key)
        return self._build_result(
            stored_filtered,
            key=key,
            entry=entry,
            from_cache=False,
            from_superset=False,
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
        entry_path = entry.path if table.num_rows == entry.row_count else None
        data = DataHandle(table, entry.page_size, entry_path)
        return CacheResult(
            data=data,
            from_cache=from_cache,
            from_superset=from_superset,
            entry_digest=entry.key.digest,
            requested_invariants=freeze_token_mapping(key.invariant_tokens),
            cached_invariants=entry.invariants,
            created_at=entry.created_at,
            expires_at=entry.expires_at,
        )
