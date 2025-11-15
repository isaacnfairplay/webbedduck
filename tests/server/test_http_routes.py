import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from webbed_duck.server.cache import CacheConfig
from webbed_duck.server.http.app import create_route_app

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("fastapi")


def _prime_database(database: Path) -> None:
    connection = duckdb.connect(str(database))
    connection.execute(
        "CREATE TABLE metrics (id INTEGER, region VARCHAR, total INTEGER)"
    )
    connection.execute(
        "INSERT INTO metrics VALUES "
        "(1, 'north', 10), (2, 'north', 20), (3, 'south', 30)"
    )
    connection.close()


def _write_template(root: Path) -> Path:
    template_root = root / "routes"
    template_path = template_root / "reports" / "daily.sql"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(
        """
        SELECT id, region, total
        FROM metrics
        WHERE region = '{{ ctx.parameters.region }}'
        AND total >= {{ ctx.constants.misc.min_total }}
        ORDER BY id
        LIMIT {{ ctx.parameters.limit }}
        """,
        encoding="utf-8",
    )
    return template_root


def _build_validations() -> dict[str, Any]:
    return {
        "reports/daily": {
            "parameters": {
                "limit": {
                    "type": "integer",
                    "allow_template": True,
                    "guards": {"range": {"min": 1, "max": 10}},
                },
                "region": {
                    "type": "string",
                    "allow_template": True,
                    "guards": {"choices": ["north", "south"]},
                },
            }
        }
    }


def _build_request_context() -> dict[str, Any]:
    return {"constants": {"str": {}, "misc": {"min_total": 0}}, "parameters": {}}


@pytest.fixture
def route_client(tmp_path: Path) -> TestClient:
    template_root = _write_template(tmp_path)
    database = tmp_path / "metrics.duckdb"
    _prime_database(database)
    cache_dir = tmp_path / "cache"
    config = CacheConfig(storage_root=cache_dir)
    app = create_route_app(
        cache_config=config,
        template_root=template_root,
        request_context=_build_request_context(),
        validations=_build_validations(),
        duckdb_database=database,
    )
    return TestClient(app)


def test_route_execution_hits_cache(route_client: TestClient) -> None:
    payload = {"parameters": {"limit": 2, "region": "north"}}
    first = route_client.post("/routes/reports/daily", json=payload)
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["row_count"] == 2
    assert first_body["from_cache"] is False

    second = route_client.post("/routes/reports/daily", json=payload)
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["from_cache"] is True
    assert second_body["digest"] == first_body["digest"]

    page_url = first_body["page_template"]["url"].format(page=0, format="json")
    streamed = route_client.get(page_url)
    assert streamed.status_code == 200
    as_rows = json.loads(streamed.text)
    assert [row["id"] for row in as_rows] == [1, 2]


def test_route_execution_validates_parameters(route_client: TestClient) -> None:
    invalid = route_client.post(
        "/routes/reports/daily",
        json={"parameters": {"limit": 0, "region": "north"}},
    )
    assert invalid.status_code == 422

    missing = route_client.post("/routes/unknown", json={"parameters": {}})
    assert missing.status_code == 404


def test_route_execution_merges_request_constants(route_client: TestClient) -> None:
    payload = {
        "parameters": {"limit": 3, "region": "north"},
        "constants": {"misc": {"min_total": 15}},
    }
    response = route_client.post("/routes/reports/daily", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["row_count"] == 1
    page_url = body["page_template"]["url"].format(page=0, format="json")
    streamed = route_client.get(page_url)
    rows = json.loads(streamed.text)
    assert [row["total"] for row in rows] == [20]
