"""Integration tests for the DuckDB-backed cache layer."""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import Any, Dict

import pytest

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")

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


def _rows(envelope) -> list[dict[str, Any]]:
    with envelope.data.open("arrow") as stream:
        table = pa.ipc.open_stream(stream).read_all()
    return table.to_pylist()


def _schema_metadata(table: pa.Table) -> list[dict[str, str]]:
    return [{"name": field.name, "type": str(field.type)} for field in table.schema]


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
    assert _rows(first) == table.to_pylist()
    assert call_counter["count"] == 1
    assert first.from_cache is False
    assert first.from_superset is False
    assert first.entry_digest == key.digest
    assert first.row_count == table.num_rows
    assert first.schema == _schema_metadata(table)
    assert first.page_size == config.page_size
    assert first.page_count == 3
    assert first.data.schema == table.schema
    assert first.data.page_count == first.page_count
    assert first.data.row_count == table.num_rows
    assert dict(first.requested_invariants) == {}
    assert dict(first.cached_invariants) == {}
    assert first.created_at == start_time
    assert first.expires_at == start_time + config.ttl

    entry_dir = tmp_cache_dir / key.digest
    assert entry_dir.exists()
    pages = sorted(entry_dir.glob("page-*.parquet"))
    assert len(pages) == 3  # ceil(5 rows / 2 page size)
    metadata = json.loads((entry_dir / "metadata.json").read_text())
    assert metadata["row_count"] == 5
    assert metadata["page_size"] == config.page_size

    second = cache.fetch_or_populate(key)
    assert _rows(second) == table.to_pylist()
    assert call_counter["count"] == 1  # hit: runner not called
    assert second.from_cache is True
    assert second.from_superset is False
    assert second.entry_digest == key.digest
    assert second.created_at == first.created_at
    assert second.expires_at == first.expires_at

    clock.advance(120)
    refresh_time = clock.now()
    third = cache.fetch_or_populate(key)
    assert _rows(third) == table.to_pylist()
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
    as_rows = _rows(result)
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
    assert dict(result.cached_invariants) == dict(result.requested_invariants)

    again = cache.fetch_or_populate(key)
    assert call_counter["count"] == 1  # cache hit bypasses runner
    assert _rows(again) == as_rows
    assert again.from_cache is True
    assert again.from_superset is False
    assert again.entry_digest == key.digest
    assert again.row_count == len(as_rows)


def test_multi_value_invariant_superset_reuse_and_metadata(
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

    superset_key = CacheKey.from_parts(
        "reports/channels",
        parameters={"view": "regional"},
        constants={"channel": "email", "region": "US|ca"},
        invariant_filters=config.invariants,
    )

    superset = cache.fetch_or_populate(superset_key)
    superset_rows = _rows(superset)
    assert call_counter["count"] == 1
    assert {row["region"] for row in superset_rows} == {"US", "CA"}
    assert superset.from_cache is False
    assert superset.from_superset is False
    assert superset.entry_digest == superset_key.digest
    assert superset.row_count == len(superset_rows)
    assert superset.schema == _schema_metadata(table)
    assert superset.page_size == config.page_size
    assert superset.page_count == 1
    assert superset.data.page_count == superset.page_count
    assert dict(superset.requested_invariants) == {
        "channel": ("email",),
        "region": ("ca", "us"),
    }
    assert dict(superset.cached_invariants) == dict(superset.requested_invariants)

    superset_dir = tmp_cache_dir / superset_key.digest
    metadata = json.loads((superset_dir / "metadata.json").read_text())
    assert metadata["row_count"] == 2
    assert metadata["ttl_seconds"] == config.ttl.total_seconds()
    assert metadata["invariants"]["region"] == ["ca", "us"]

    subset_key = CacheKey.from_parts(
        "reports/channels",
        parameters={"view": "regional"},
        constants={"channel": "email", "region": "ca"},
        invariant_filters=config.invariants,
    )

    subset = cache.fetch_or_populate(subset_key)
    assert call_counter["count"] == 1  # served from cached superset
    subset_rows = _rows(subset)
    assert {row["region"] for row in subset_rows} == {"CA"}
    assert all(row["channel"] == "email" for row in subset_rows)
    assert subset.from_cache is True
    assert subset.from_superset is True
    assert subset.entry_digest == superset_key.digest
    assert subset.row_count == len(subset_rows)
    assert subset.page_count == 1
    assert dict(subset.requested_invariants) == {
        "channel": ("email",),
        "region": ("ca",),
    }
    assert dict(subset.cached_invariants) == dict(superset.cached_invariants)

    miss_key = CacheKey.from_parts(
        "reports/channels",
        parameters={"view": "regional"},
        constants={"channel": "email", "region": "us|mx"},
        invariant_filters=config.invariants,
    )

    miss = cache.fetch_or_populate(miss_key)
    assert _rows(miss) == [
        {"channel": "email", "region": "US", "cohort": "north"},
        {"channel": "email", "region": "MX", "cohort": "south"},
    ]
    assert call_counter["count"] == 2  # superset did not cover MX token
    assert miss.from_cache is False
    assert miss.from_superset is False
    assert miss.entry_digest == miss_key.digest
    assert miss.row_count == 2
    assert miss.page_count == 1
    assert dict(miss.requested_invariants) == {
        "channel": ("email",),
        "region": ("mx", "us"),
    }

    miss_dir = tmp_cache_dir / miss_key.digest
    miss_metadata = json.loads((miss_dir / "metadata.json").read_text())
    assert miss_metadata["row_count"] == 2
    assert miss_metadata["invariants"]["region"] == ["mx", "us"]


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
    rows_vip = _rows(result)
    assert call_counter["count"] == 1
    assert {row["user"] for row in rows_vip} == {"alice", "bob"}
    assert result.from_cache is False
    assert result.from_superset is False
    assert result.entry_digest == key.digest
    assert result.row_count == len(rows_vip)
    assert result.page_count == 1
    assert dict(result.requested_invariants) == {"segment": ("vip",)}
    assert dict(result.cached_invariants) == {"segment": ("vip",)}

    metadata = json.loads((tmp_cache_dir / key.digest / "metadata.json").read_text())
    assert metadata["invariants"]["segment"] == ["vip"]

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
    assert _rows(again) == rows_vip
    assert again.from_cache is True
    assert again.from_superset is False
    assert again.entry_digest == key.digest
    assert again.row_count == len(rows_vip)


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
    assert _rows(result) == [{"user_id": 2, "name": "bob"}]
    assert result.page_count == 1
    assert result.from_cache is False
    assert result.from_superset is False
    assert result.entry_digest == key.digest
    assert result.row_count == 1
    assert dict(result.requested_invariants) == {"user_id": ("2",)}
    assert dict(result.cached_invariants) == {"user_id": ("2",)}
