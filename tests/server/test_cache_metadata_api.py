import datetime as _dt

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from webbed_duck.server.cache import (
    Cache,
    CacheConfig,
    CacheKey,
    CacheStorage,
    peek_metadata,
)
from webbed_duck.server.http.app import create_cache_app


def _build_table() -> pa.Table:
    return pa.table(
        {
            "id": [1, 2, 3, 4],
            "region": ["north", "south", "east", "west"],
        }
    )


def test_peek_metadata_matches_cached_response_and_skips_parquet(tmp_path, monkeypatch):
    storage_root = tmp_path / "cache"
    config = CacheConfig(storage_root=storage_root, ttl=_dt.timedelta(minutes=5), page_size=2)
    cache = Cache(config=config, run_query=lambda *args, **kwargs: _build_table())
    key = CacheKey.from_parts("reports/regions", parameters={"limit": "all"})

    cache.fetch_or_populate(key)
    cached = cache.fetch_or_populate(key)
    assert cached.from_cache is True

    def _boom(*args, **kwargs):  # pragma: no cover - defensive helper
        raise AssertionError("Parquet pages should not be read during metadata peeks")

    monkeypatch.setattr(pq, "read_table", _boom)

    summary = peek_metadata(cache._storage, key.digest)
    assert summary is not None
    assert summary.digest == key.digest
    assert summary.from_cache is True
    assert summary.from_superset is False
    assert summary.page_count == cached.page_count
    assert summary.formats == cached.formats
    assert summary.row_count == cached.row_count
    assert dict(summary.parameters) == dict(key.parameter_values)
    assert dict(summary.constants) == dict(key.constant_values)
    assert summary.cached_invariants == cached.cached_invariants
    assert summary.requested_invariants == cached.requested_invariants


def test_cache_metadata_endpoint_returns_summary(tmp_path):
    storage_root = tmp_path / "cache"
    config = CacheConfig(storage_root=storage_root, ttl=_dt.timedelta(minutes=5), page_size=3)
    cache = Cache(config=config, run_query=lambda *args, **kwargs: _build_table())
    key = CacheKey.from_parts("reports/regions", parameters={"limit": "all"})

    cache.fetch_or_populate(key)
    cached = cache.fetch_or_populate(key)

    app = create_cache_app(CacheStorage(storage_root))
    client = TestClient(app)
    response = client.get(f"/cache/{key.digest}")
    assert response.status_code == 200

    payload = response.json()
    assert payload["digest"] == key.digest
    assert payload["route_slug"] == "reports/regions"
    assert payload["from_cache"] == cached.from_cache
    assert payload["from_superset"] == cached.from_superset
    assert payload["page_count"] == cached.page_count
    assert payload["page_size"] == cached.page_size
    assert payload["row_count"] == cached.row_count
    assert payload["parameters"] == dict(key.parameter_values)
    assert payload["constants"] == dict(key.constant_values)
    assert payload["requested_invariants"] == dict(cached.requested_invariants)
    assert payload["cached_invariants"] == dict(cached.cached_invariants)

    template_url = payload["page_template"]["url"]
    assert template_url.endswith(
        f"/cache/{key.digest}/pages/{{page}}?format={{format}}"
    )


def test_peek_metadata_rejects_invalid_digest(tmp_path):
    storage_root = tmp_path / "cache"
    storage = CacheStorage(storage_root)

    assert peek_metadata(storage, "../etc/passwd") is None
    assert peek_metadata(storage, "not-a-digest") is None


def test_cache_metadata_endpoint_rejects_invalid_digest(tmp_path):
    storage_root = tmp_path / "cache"
    app = create_cache_app(CacheStorage(storage_root))
    client = TestClient(app)

    response = client.get("/cache/..%2F..%2Fetc%2Fpasswd")
    assert response.status_code == 404
