"""Integration tests for the DuckDB-backed cache layer."""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
from typing import Any, Dict

import pytest

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")

from webbed_duck.server.cache import Cache, CacheConfig, CacheKey


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

    first = cache.fetch_or_populate(key)
    assert first.to_pylist() == table.to_pylist()
    assert call_counter["count"] == 1

    entry_dir = tmp_cache_dir / key.digest
    assert entry_dir.exists()
    pages = sorted(entry_dir.glob("page-*.parquet"))
    assert len(pages) == 3  # ceil(5 rows / 2 page size)
    metadata = json.loads((entry_dir / "metadata.json").read_text())
    assert metadata["row_count"] == 5

    second = cache.fetch_or_populate(key)
    assert second.to_pylist() == table.to_pylist()
    assert call_counter["count"] == 1  # hit: runner not called

    clock.advance(120)
    third = cache.fetch_or_populate(key)
    assert third.to_pylist() == table.to_pylist()
    assert call_counter["count"] == 2  # miss after TTL expiry


def test_invariant_filters_and_null_semantics(tmp_cache_dir: pathlib.Path) -> None:
    clock = _Clock(_dt.datetime(2024, 1, 1, tzinfo=_dt.timezone.utc))
    config = CacheConfig(storage_root=tmp_cache_dir, ttl=_dt.timedelta(minutes=5), page_size=10)

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
    )

    result = cache.fetch_or_populate(key)
    as_rows = result.to_pylist()
    assert call_counter["count"] == 1
    assert all(row["segment"] is None for row in as_rows)
    assert all(row["channel"] == "email" for row in as_rows)
    assert {row["region"] for row in as_rows} == {None, "US"}

    again = cache.fetch_or_populate(key)
    assert call_counter["count"] == 1  # cache hit bypasses runner
    assert again.to_pylist() == as_rows
