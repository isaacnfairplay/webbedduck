"""Integration-style tests for the DuckDB cache layer."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict

import pytest

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")
pc = pytest.importorskip("pyarrow.compute")

from webbed_duck.server.cache import NULL_SENTINEL, Cache, CacheConfig, CacheKey


def _sample_table() -> "pa.Table":
    connection = duckdb.connect()
    try:
        # Build a deterministic table with a mix of invariant values to stress filtering.
        relation = connection.execute(
            """
            select * from (
                values
                    (1, 'west', true),
                    (2, 'east', false),
                    (3, 'west', true),
                    (4, 'north', true),
                    (5, 'east', true)
            ) as t(id, region, active)
            order by id
            """
        )
        return relation.arrow().read_all()
    finally:
        connection.close()


def test_cache_key_digest_is_deterministic_and_null_safe() -> None:
    key = CacheKey.from_parts(
        route="metrics/summary",
        parameters={"limit": 5, "order": ["id", "desc"]},
        constants={"region": None, "active": True},
    )
    # Reordering the dictionaries should not affect the digest.
    same_key = CacheKey.from_parts(
        route="metrics/summary",
        parameters={"order": ["desc", "id"], "limit": 5},
        constants={"active": True, "region": None},
    )

    assert key.digest == same_key.digest
    assert key.parameters_dict == same_key.parameters_dict
    assert key.constants_dict == {
        "region": NULL_SENTINEL,
        "active": True,
    }
    assert key.route_slug == "metrics-summary"


def test_fetch_or_populate_writes_pages_enforces_ttl_and_filters_subsets(tmp_path) -> None:
    base_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    tick: Dict[str, float] = {"offset": 0.0}

    def clock() -> datetime:
        return base_time + timedelta(seconds=tick["offset"])

    table = _sample_table()
    runner_calls: list[float] = []

    def runner() -> "pa.Table":
        runner_calls.append(clock().timestamp())
        return table

    config = CacheConfig(storage_root=tmp_path, ttl=timedelta(seconds=30), page_size=2)
    cache = Cache(config=config, clock=clock)

    general_key = CacheKey.from_parts(
        route="metrics/summary",
        parameters={"limit": 5},
        constants={"region": NULL_SENTINEL, "active": NULL_SENTINEL},
    )

    first = cache.fetch_or_populate(general_key, runner)
    assert runner_calls and len(runner_calls) == 1
    assert first.equals(table)

    entry_dir = tmp_path / general_key.route_slug / general_key.digest
    page_files = sorted(entry_dir.glob("page-*.parquet"))
    assert [path.name for path in page_files] == [
        "page-0000.parquet",
        "page-0001.parquet",
        "page-0002.parquet",
    ]

    metadata = json.loads((entry_dir / "metadata.json").read_text())
    assert metadata["page_size"] == 2
    assert metadata["key"]["constants"]["region"] == NULL_SENTINEL

    repeat = cache.fetch_or_populate(general_key, runner)
    assert len(runner_calls) == 1, "cache hit should bypass the runner"
    assert repeat.equals(table)

    tick["offset"] = 45
    expired = cache.fetch_or_populate(general_key, runner)
    assert len(runner_calls) == 2, "expired entry should re-run the DuckDB query"
    assert expired.equals(table)

    tick["offset"] = 50
    specific_key = CacheKey.from_parts(
        route="metrics/summary",
        parameters={"limit": 5},
        constants={"region": "west", "active": NULL_SENTINEL},
    )

    subset = cache.fetch_or_populate(specific_key, runner)
    assert len(runner_calls) == 2, "subset should reuse cached pages"
    regions = subset.column("region").to_pylist()
    assert regions == ["west", "west"]
    actives = subset.column("active").to_pylist()
    assert all(actives)

