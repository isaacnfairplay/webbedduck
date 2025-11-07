import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")
import pyarrow.compute as pc

from webbed_duck.server.cache import (
    NULL_SENTINEL,
    Cache,
    CacheConfig,
    CacheKey,
)


def _build_arrow_table(rows: List[Dict[str, Any]]):
    return pa.Table.from_pylist(rows)


def _duckdb_table(num_rows: int) -> pa.Table:
    con = duckdb.connect()
    try:
        con.execute(
            "select range as id, concat('duck_', range) as name, case when range % 2 = 0 then 'north' else 'south' end as region, case when range % 3 = 0 then NULL else 'lake' end as habitat from range(?)",
            [num_rows],
        )
        return con.fetch_arrow_table()
    finally:
        con.close()


def test_cache_key_serialization_uses_null_sentinel():
    key = CacheKey.from_parts(
        route_slug="routes.ducks",
        parameters={"limit": 5, "offset": 0},
        constants={"region": None, "habitat": "lake"},
    )

    assert key.route_slug == "routes.ducks"
    assert ("region", NULL_SENTINEL) in key.constants
    assert ("habitat", "lake") in key.constants
    assert key.digest == CacheKey.from_parts(
        "routes.ducks", {"offset": 0, "limit": 5}, {"habitat": "lake", "region": None}
    ).digest


def test_fetch_or_populate_respects_ttl_and_page_sizing(tmp_path, monkeypatch):
    ttl_seconds = 5
    config = CacheConfig(storage_root=tmp_path, ttl_seconds=ttl_seconds, page_size=3)

    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    clock_times = [now, now + timedelta(seconds=2), now + timedelta(seconds=6)]

    def fake_now():
        return clock_times.pop(0)

    cache = Cache(config, now=fake_now)

    table = _duckdb_table(7)
    calls: List[pa.Table] = []

    def runner() -> pa.Table:
        calls.append(table)
        return table

    result1 = cache.fetch_or_populate(
        route_slug="routes.ducks",
        parameters={"limit": 10},
        constants={},
        runner=runner,
    )

    assert result1.equals(table)
    assert len(calls) == 1

    key = CacheKey.from_parts("routes.ducks", {"limit": 10}, {})
    entry_dir = cache.cache_dir_for(key)
    page_files = sorted(entry_dir.glob("page-*.parquet"))
    assert len(page_files) == math.ceil(table.num_rows / config.page_size)

    # Second call before TTL should hit cache and avoid runner.
    result2 = cache.fetch_or_populate(
        route_slug="routes.ducks",
        parameters={"limit": 10},
        constants={},
        runner=runner,
    )
    assert len(calls) == 1
    assert result2.equals(table)

    # After TTL expires the runner should be invoked again and the cache refreshed.
    refreshed = cache.fetch_or_populate(
        route_slug="routes.ducks",
        parameters={"limit": 10},
        constants={},
        runner=runner,
    )
    assert len(calls) == 2
    assert refreshed.equals(table)


def test_cached_reads_filter_invariants_and_bypass_runner_on_hits(tmp_path):
    config = CacheConfig(storage_root=tmp_path, ttl_seconds=60, page_size=4)
    cache = Cache(config)

    base_table = _duckdb_table(10)

    def runner() -> pa.Table:
        return base_table

    filtered = cache.fetch_or_populate(
        route_slug="routes.ducks",
        parameters={"limit": 10},
        constants={"region": "north", "habitat": None},
        runner=runner,
    )

    expected = base_table.filter(
        pc.and_(
            pc.equal(base_table["region"], pa.scalar("north")),
            pc.is_null(base_table["habitat"]),
        )
    )
    assert filtered.equals(expected)

    def fail_runner() -> pa.Table:
        raise AssertionError("runner should not be called on cache hit")

    second = cache.fetch_or_populate(
        route_slug="routes.ducks",
        parameters={"limit": 10},
        constants={"region": "north", "habitat": None},
        runner=fail_runner,
    )
    assert second.equals(expected)
