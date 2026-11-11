"""POST /api/forecast against a tiny `timesfm25` GGUF, its capability listing, and the guards that
keep chat and forecasting apart."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.models.catalog import ModelCatalog
from app.models.installed_model import InstalledModel
from tests.tiny_gguf_timesfm25 import HORIZON, build_tiny_timesfm25_gguf

TAG = "tiny-timesfm:latest"
SERIES = [3.0, 4.0, 3.5, 5.0, 4.5, 6.0, 5.5, 7.0]


@pytest.fixture
def tiny_forecast_model(models_dir: Path) -> InstalledModel:
    repo_dir = models_dir / "hf.co" / "test-org" / "tiny-timesfm"
    repo_dir.mkdir(parents=True, exist_ok=True)
    gguf_path = build_tiny_timesfm25_gguf(repo_dir / "tiny.gguf")
    installed = InstalledModel(
        tag=TAG,
        path=str(gguf_path),
        architecture="timesfm25",
        capabilities=["forecast"],
        size_bytes=gguf_path.stat().st_size,
        family="timesfm25",
        parameter_size="0.001B",
        context_length=64,
    )
    sidecar = repo_dir / f"{gguf_path.stem}{ModelCatalog.SIDECAR_SUFFIX}"
    sidecar.write_text(json.dumps(installed.__dict__))
    return installed


def _forecast(client: TestClient, **extra: object) -> dict:
    body = {"model": TAG, "series": [SERIES], "horizon": 4, **extra}
    response = client.post("/api/forecast", json=body)
    assert response.status_code == 200, response.text
    return response.json()


class TestForecast:
    def test_forecasts_each_series(
        self, client: TestClient, tiny_forecast_model: InstalledModel
    ) -> None:
        body = _forecast(client, series=[SERIES, SERIES[:3]])

        assert body["horizon"] == 4
        assert len(body["forecasts"]) == 2
        for forecast in body["forecasts"]:
            assert len(forecast["point"]) == len(forecast["mean"]) == 4
            assert len(forecast["quantiles"]) == 9
            assert forecast["point"] == forecast["quantiles"]["0.5"]

    def test_tags_report_only_the_forecast_capability(
        self, client: TestClient, tiny_forecast_model: InstalledModel
    ) -> None:
        models = client.get("/api/tags").json()["models"]
        entry = next(m for m in models if m["name"] == TAG)
        assert entry["capabilities"] == ["forecast"]

    def test_horizon_past_the_model_limit_returns_400(
        self, client: TestClient, tiny_forecast_model: InstalledModel
    ) -> None:
        body = {"model": TAG, "series": [SERIES], "horizon": HORIZON + 1}
        response = client.post("/api/forecast", json=body)
        assert response.status_code == 400
        assert response.json()["code"] == "invalid_forecast_input"

    def test_empty_series_is_rejected(
        self, client: TestClient, tiny_forecast_model: InstalledModel
    ) -> None:
        body = {"model": TAG, "series": [[]], "horizon": 2}
        assert client.post("/api/forecast", json=body).status_code in (400, 422)


class TestWrongModelKind:
    def test_chat_with_a_forecaster_is_refused(
        self, client: TestClient, tiny_forecast_model: InstalledModel
    ) -> None:
        response = client.post(
            "/api/chat",
            json={"model": TAG, "messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert response.status_code == 400
        assert "forecasting model" in response.json()["error"]

    def test_forecast_with_a_chat_model_is_refused(
        self, client: TestClient, tiny_qwen3_model: InstalledModel
    ) -> None:
        body = {"model": "tiny-qwen3:latest", "series": [SERIES], "horizon": 2}
        response = client.post("/api/forecast", json=body)
        assert response.status_code == 400
        assert "not a forecasting model" in response.json()["error"]
