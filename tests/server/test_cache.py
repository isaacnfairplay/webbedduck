import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict

import pytest

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")

from webbed_duck.server.cache import (
    NULL_SENTINEL,
    Cache,
    CacheConfig,
    CacheKey,
)


class FakeClock:
    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime.now(timezone.utc)

    def now(self) -> datetime:
        return self._now

    def advance(self, *, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


def _build_duck_table(region: str) -> pa.Table:
    query = """
        select
            i as row_id,
            case when i % 2 = 0 then 'even' else 'odd' end as parity,
            $1::varchar as region,
            case when i % 3 = 0 then NULL else $2::varchar end as status
        from range(0, 6) as t(i)
    """
    # DuckDB returns a pyarrow.Table directly, which we can use for cache pages.
    reader = duckdb.sql(query, params=[region, "active"]).arrow()
    return reader.read_all()


def test_cache_key_includes_null_sentinel() -> None:
    key_from_none = CacheKey.build(
        route_slug="insights",
        parameters={"region": None},
        constants={"status": None, "flag": "yes"},
    )
    key_from_sentinel = CacheKey.build(
        route_slug="insights",
        parameters={"region": NULL_SENTINEL},
        constants={"status": NULL_SENTINEL, "flag": "yes"},
    )

    assert key_from_none.digest == key_from_sentinel.digest
    assert key_from_none.normalized == key_from_sentinel.normalized


def test_fetch_or_populate_round_trip(tmp_path: Path) -> None:
    config = CacheConfig(
        ttl=timedelta(seconds=10),
        page_size=2,
        storage_root=tmp_path,
    )
    clock = FakeClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
    cache = Cache(config=config, clock=clock.now)

    base_parameters: Dict[str, str | None] = {"region": "north"}
    constants = {"status": NULL_SENTINEL}
    key = CacheKey.build("reports", parameters=base_parameters, constants=constants)

    runner_calls: list[datetime] = []

    def runner(incoming_key: CacheKey) -> pa.Table:
        runner_calls.append(clock.now())
        region_value = dict(incoming_key.normalized.parameters)["region"]
        return _build_duck_table(region_value)

    even_filter = {"parity": "even"}
    result_even = cache.fetch_or_populate(
        key=key,
        runner=runner,
        invariant_filters=even_filter,
    )

    assert runner_calls, "runner should have been invoked on cache miss"
    assert result_even.num_rows == 3
    assert {row["parity"] for row in result_even.to_pylist()} == {"even"}

    cache_dir = tmp_path / key.digest
    pages = sorted(cache_dir.glob("page-*.parquet"))
    assert len(pages) == math.ceil(_build_duck_table("north").num_rows / config.page_size)

    runner_calls.clear()
    odd_filter = {"parity": "odd"}
    result_odd = cache.fetch_or_populate(
        key=key,
        runner=runner,
        invariant_filters=odd_filter,
    )

    assert not runner_calls, "cache hit should avoid runner invocation"
    assert result_odd.num_rows == 3
    assert {row["parity"] for row in result_odd.to_pylist()} == {"odd"}

    clock.advance(seconds=11)
    cache.fetch_or_populate(key=key, runner=runner)
    assert runner_calls, "TTL expiry should force runner invocation"
