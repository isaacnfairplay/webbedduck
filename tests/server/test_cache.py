"""Integration tests for the DuckDB-backed cache layer."""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import Any, Dict

import pytest

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from webbed_duck.server.cache import Cache, CacheConfig, CacheKey, InvariantFilter


class _Clock:
    """Deterministic clock helper for TTL validation."""

    def __init__(self, start: _dt.datetime) -> None:
        self._now = start

    def now(self) -> _dt.datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now = self._now + _dt.timedelta(seconds=seconds)


def _format_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    return str(value)


def _build_duckdb_table(rows: list[tuple[Any, ...]], columns: list[str]) -> pa.Table:
    relation = duckdb.connect(database=":memory:")
    values_clause = ",".join(
        "(" + ",".join(_format_literal(value) for value in row) + ")" for row in rows
    )
    relation.execute(
        "CREATE TABLE data AS SELECT * FROM (VALUES {}) AS t({})".format(
            values_clause,
            ",".join(columns),
        )
    )
    return relation.sql("SELECT * FROM data").arrow().read_all()


@pytest.fixture
def tmp_cache_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "cache"


def test_cache_key_deterministic_and_null_sentinel(tmp_cache_dir: pathlib.Path) -> None:
    key_a = CacheKey.from_parts(
        route_slug="reports/sales",
        parameters={"start": "2024-01-01", "end": None},
        constants={"region": "EU", "channel": None},
    )
    key_b = CacheKey.from_parts(
        route_slug="reports/sales",
        parameters={"end": None, "start": "2024-01-01"},
        constants={"channel": None, "region": "EU"},
    )

    assert key_a.digest == key_b.digest
    assert key_a.parameters == (
        ("end", CacheKey.NULL_SENTINEL),
        ("start", "2024-01-01"),
    )
    assert key_a.constants == (
        ("channel", CacheKey.NULL_SENTINEL),
        ("region", "EU"),
    )


def test_fetch_or_populate_persists_pages_and_enforces_ttl(tmp_cache_dir: pathlib.Path) -> None:
    clock = _Clock(_dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc))
    config = CacheConfig(storage_root=tmp_cache_dir, ttl=_dt.timedelta(seconds=60), page_size=2)

    rows = [
        (1, "north", "2024-01-01"),
        (2, "south", "2024-01-02"),
        (3, "east", "2024-01-03"),
        (4, "west", "2024-01-04"),
        (5, "central", "2024-01-05"),
    ]
    columns = ["id", "region", "day"]

    table = _build_duckdb_table(rows, columns)

    call_counter: dict[str, int] = {"count": 0}

    def runner(route_slug: str, parameters: Dict[str, Any], constants: Dict[str, Any]) -> pa.Table:
        call_counter["count"] += 1
        return table

    cache = Cache(config=config, run_query=runner, clock=clock.now)
    key = CacheKey.from_parts("reports/regions", parameters={"limit": "all"})

    start_time = clock.now()
    first = cache.fetch_or_populate(key)
    assert first.to_pylist() == table.to_pylist()
    assert call_counter["count"] == 1
    assert first.from_cache is False
    assert first.from_superset is False
    assert first.entry_digest == key.digest
    assert first.row_count == table.num_rows
    assert first.page_size == config.page_size
    assert first.page_count == 3
    assert set(first.formats) == {"arrow", "parquet", "csv", "json", "jsonl"}
    assert dict(first.requested_invariants) == {}
    assert dict(first.cached_invariants) == {}
    assert first.created_at == start_time
    assert first.expires_at == start_time + config.ttl

    arrow_view = first.as_arrow()
    assert arrow_view.equals(table)
    page_two = first.as_arrow(page=1)
    assert page_two.num_rows == 2
    assert [row["id"] for row in page_two.to_pylist()] == [3, 4]
    with first.data.open("json", page=0) as json_stream:
        first_page = json.loads(json_stream.read())
        assert [row["id"] for row in first_page] == [1, 2]
    with first.data.open("jsonl", page=0) as jsonl_stream:
        lines = [json.loads(chunk) for chunk in jsonl_stream.read().splitlines()]
        assert [row["id"] for row in lines] == [1, 2]
    with first.data.open("csv", page=2) as csv_stream:
        csv_payload = csv_stream.read()
        assert "central" in csv_payload
    with first.data.open("parquet", page=1) as parquet_stream:
        parquet_bytes = parquet_stream.read()

    page_path = sorted((tmp_cache_dir / key.digest).glob("page-*.parquet"))[1]
    assert parquet_bytes == page_path.read_bytes()

    entry_dir = tmp_cache_dir / key.digest
    assert entry_dir.exists()
    pages = sorted(entry_dir.glob("page-*.parquet"))
    assert len(pages) == 3  # ceil(5 rows / 2 page size)
    metadata = json.loads((entry_dir / "metadata.json").read_text())
    assert metadata["row_count"] == 5
    assert metadata["page_size"] == config.page_size

    second = cache.fetch_or_populate(key)
    assert second.to_pylist() == table.to_pylist()
    assert call_counter["count"] == 1  # hit: runner not called
    assert second.from_cache is True
    assert second.from_superset is False
    assert second.entry_digest == key.digest
    assert second.created_at == first.created_at
    assert second.expires_at == first.expires_at

    clock.advance(120)
    refresh_time = clock.now()
    third = cache.fetch_or_populate(key)
    assert third.to_pylist() == table.to_pylist()
    assert call_counter["count"] == 2  # miss after TTL expiry
    assert third.from_cache is False
    assert third.from_superset is False
    assert third.entry_digest == key.digest
    assert third.created_at == refresh_time
    assert third.row_count == table.num_rows


def test_invariant_filters_and_null_semantics(tmp_cache_dir: pathlib.Path) -> None:
    clock = _Clock(_dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc))
    config = CacheConfig(
        storage_root=tmp_cache_dir,
        ttl=_dt.timedelta(minutes=5),
        page_size=10,
        invariants={
            "segment": InvariantFilter("segment"),
            "channel": InvariantFilter("channel"),
        },
    )

    rows = [
        ("email", None, None),
        ("email", "US", None),
        ("sms", None, "loyal"),
        ("push", "EU", None),
    ]
    columns = ["channel", "region", "segment"]
    table = _build_duckdb_table(rows, columns)

    call_counter: dict[str, int] = {"count": 0}

    def runner(route_slug: str, parameters: Dict[str, Any], constants: Dict[str, Any]) -> pa.Table:
        call_counter["count"] += 1
        return table

    cache = Cache(config=config, run_query=runner, clock=clock.now)

    key = CacheKey.from_parts(
        "reports/audience",
        parameters={"view": "summary"},
        constants={"segment": CacheKey.NULL_SENTINEL, "channel": "email"},
        invariant_filters=config.invariants,
    )

    result = cache.fetch_or_populate(key)
    as_rows = result.to_pylist()
    assert call_counter["count"] == 1
    assert all(row["segment"] is None for row in as_rows)
    assert all(row["channel"] == "email" for row in as_rows)
    assert {row["region"] for row in as_rows} == {None, "US"}
    assert result.from_cache is False
    assert result.from_superset is False
    assert result.entry_digest == key.digest
    assert result.row_count == len(as_rows)
    assert dict(result.requested_invariants) == {
        "channel": ("email",),
        "segment": (CacheKey.NULL_SENTINEL,),
    }
    assert dict(result.cached_invariants) == {}

    again = cache.fetch_or_populate(key)
    assert call_counter["count"] == 1  # cache hit bypasses runner
    assert again.to_pylist() == as_rows
    assert again.from_cache is True
    assert again.from_superset is False
    assert again.entry_digest == key.digest
    assert again.row_count == len(as_rows)
    assert dict(again.cached_invariants) == {}


def test_invariant_requests_share_base_entry_and_metadata(
    tmp_cache_dir: pathlib.Path,
) -> None:
    clock = _Clock(_dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc))
    config = CacheConfig(
        storage_root=tmp_cache_dir,
        ttl=_dt.timedelta(minutes=10),
        page_size=5,
        invariants={
            "region": InvariantFilter("region", separator="|", case_insensitive=True),
            "channel": InvariantFilter("channel"),
        },
    )

    rows = [
        ("email", "US", "north"),
        ("email", "CA", "north"),
        ("email", "MX", "south"),
        ("sms", "US", "alerts"),
    ]
    columns = ["channel", "region", "cohort"]
    table = _build_duckdb_table(rows, columns)

    call_counter: dict[str, int] = {"count": 0}

    def runner(route_slug: str, parameters: Dict[str, Any], constants: Dict[str, Any]) -> pa.Table:
        call_counter["count"] += 1
        return table

    cache = Cache(config=config, run_query=runner, clock=clock.now)

    first_key = CacheKey.from_parts(
        "reports/channels",
        parameters={"view": "regional"},
        constants={"channel": "email", "region": "US|ca"},
        invariant_filters=config.invariants,
    )

    initial = cache.fetch_or_populate(first_key)
    initial_rows = initial.to_pylist()
    assert call_counter["count"] == 1
    assert {row["region"] for row in initial_rows} == {"US", "CA"}
    assert initial.from_cache is False
    assert initial.from_superset is False
    assert initial.entry_digest == first_key.digest
    assert initial.row_count == len(initial_rows)
    assert dict(initial.requested_invariants) == {
        "channel": ("email",),
        "region": ("ca", "us"),
    }
    assert dict(initial.cached_invariants) == {}

    entry_dir = tmp_cache_dir / first_key.digest
    metadata = json.loads((entry_dir / "metadata.json").read_text())
    assert metadata["row_count"] == table.num_rows
    assert metadata["ttl_seconds"] == config.ttl.total_seconds()
    assert metadata["invariants"] == {}

    subset_key = CacheKey.from_parts(
        "reports/channels",
        parameters={"view": "regional"},
        constants={"channel": "email", "region": "ca"},
        invariant_filters=config.invariants,
    )
    assert subset_key.digest == first_key.digest

    subset = cache.fetch_or_populate(subset_key)
    assert call_counter["count"] == 1  # served from shared base cache
    subset_rows = subset.to_pylist()
    assert {row["region"] for row in subset_rows} == {"CA"}
    assert all(row["channel"] == "email" for row in subset_rows)
    with subset.open("parquet", page=0) as parquet_stream:
        streamed_subset = pq.read_table(parquet_stream).to_pylist()
    assert streamed_subset == subset_rows
    assert subset.from_cache is True
    assert subset.from_superset is False
    assert subset.entry_digest == initial.entry_digest
    assert subset.row_count == len(subset_rows)
    assert dict(subset.requested_invariants) == {
        "channel": ("email",),
        "region": ("ca",),
    }
    assert dict(subset.cached_invariants) == {}

    expanded_key = CacheKey.from_parts(
        "reports/channels",
        parameters={"view": "regional"},
        constants={"channel": "email", "region": "us|mx"},
        invariant_filters=config.invariants,
    )
    assert expanded_key.digest == first_key.digest

    expanded = cache.fetch_or_populate(expanded_key)
    assert expanded.to_pylist() == [
        {"channel": "email", "region": "US", "cohort": "north"},
        {"channel": "email", "region": "MX", "cohort": "south"},
    ]
    assert call_counter["count"] == 1  # base cache reused without rerunning query
    assert expanded.from_cache is True
    assert expanded.from_superset is False
    assert expanded.entry_digest == initial.entry_digest
    assert expanded.row_count == 2
    assert dict(expanded.requested_invariants) == {
        "channel": ("email",),
        "region": ("mx", "us"),
    }
    assert dict(expanded.cached_invariants) == {}

    metadata_after_reuse = json.loads((entry_dir / "metadata.json").read_text())
    assert metadata_after_reuse["row_count"] == table.num_rows
    assert metadata_after_reuse["invariants"] == {}


def test_case_insensitive_invariant_tokens(tmp_cache_dir: pathlib.Path) -> None:
    clock = _Clock(_dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc))
    config = CacheConfig(
        storage_root=tmp_cache_dir,
        ttl=_dt.timedelta(minutes=2),
        page_size=20,
        invariants={"segment": InvariantFilter("segment", case_insensitive=True)},
    )

    rows = [
        ("alice", "VIP"),
        ("bob", "vip"),
        ("carol", "Prospect"),
        ("dave", None),
    ]
    columns = ["user", "segment"]
    table = _build_duckdb_table(rows, columns)

    call_counter: dict[str, int] = {"count": 0}

    def runner(route_slug: str, parameters: Dict[str, Any], constants: Dict[str, Any]) -> pa.Table:
        call_counter["count"] += 1
        return table

    cache = Cache(config=config, run_query=runner, clock=clock.now)

    key = CacheKey.from_parts(
        "reports/segments",
        parameters={},
        constants={"segment": "VIP"},
        invariant_filters=config.invariants,
    )

    result = cache.fetch_or_populate(key)
    rows_vip = result.to_pylist()
    assert call_counter["count"] == 1
    assert {row["user"] for row in rows_vip} == {"alice", "bob"}
    assert result.from_cache is False
    assert result.from_superset is False
    assert result.entry_digest == key.digest
    assert result.row_count == len(rows_vip)
    assert dict(result.requested_invariants) == {"segment": ("vip",)}
    assert dict(result.cached_invariants) == {}

    metadata = json.loads((tmp_cache_dir / key.digest / "metadata.json").read_text())
    assert metadata["invariants"] == {}

    # Case-insensitive tokens should produce the same cache key digest
    same_key = CacheKey.from_parts(
        "reports/segments",
        parameters={},
        constants={"segment": "vip"},
        invariant_filters=config.invariants,
    )
    assert same_key.digest == key.digest
    again = cache.fetch_or_populate(same_key)
    assert call_counter["count"] == 1
    assert again.to_pylist() == rows_vip
    assert again.from_cache is True
    assert again.from_superset is False
    assert again.entry_digest == key.digest
    assert again.row_count == len(rows_vip)
    assert dict(again.cached_invariants) == {}


def test_numeric_invariant_tokens_apply_column_type(tmp_cache_dir: pathlib.Path) -> None:
    clock = _Clock(_dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc))
    config = CacheConfig(
        storage_root=tmp_cache_dir,
        ttl=_dt.timedelta(minutes=1),
        page_size=10,
        invariants={"user_id": InvariantFilter("user_id")},
    )

    rows = [
        (1, "alice"),
        (2, "bob"),
        (3, "carol"),
    ]
    columns = ["user_id", "name"]
    table = _build_duckdb_table(rows, columns)

    call_counter: dict[str, int] = {"count": 0}

    def runner(route_slug: str, parameters: Dict[str, Any], constants: Dict[str, Any]) -> pa.Table:
        call_counter["count"] += 1
        return table

    cache = Cache(config=config, run_query=runner, clock=clock.now)

    key = CacheKey.from_parts(
        "reports/users",
        parameters={},
        constants={"user_id": 2},
        invariant_filters=config.invariants,
    )

    result = cache.fetch_or_populate(key)
    assert call_counter["count"] == 1
    assert result.to_pylist() == [{"user_id": 2, "name": "bob"}]
    assert result.from_cache is False
    assert result.from_superset is False
    assert result.entry_digest == key.digest
    assert result.row_count == 1
    assert dict(result.requested_invariants) == {"user_id": ("2",)}
    assert dict(result.cached_invariants) == {}
