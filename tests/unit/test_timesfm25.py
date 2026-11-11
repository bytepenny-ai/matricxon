"""TimesFm25Architecture + ForecastEngine on a tiny random `timesfm25` GGUF: the forecast's
symmetries (scale, shift, sign), padding never leaking between series, and input checks."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from app.architectures.registry import ArchitectureRegistry
from app.architectures.timesfm25 import TimesFm25Architecture
from app.gguf.loader import GGUFModelLoader
from app.gguf.reader import GGUFReader
from app.runtime.forecast_engine import Forecast, ForecastEngine
from app.server.errors import InvalidForecastInputError
from tests.tiny_gguf_timesfm25 import HORIZON, N_OUT, QUANTILE_LEN, build_tiny_timesfm25_gguf

SERIES = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.5, 7.0, 6.5, 8.0, 9.0]


@pytest.fixture(scope="module")
def gguf_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_timesfm25_gguf(tmp_path_factory.mktemp("tfm") / "tiny.gguf")


@pytest.fixture(scope="module")
def model(gguf_path: Path) -> TimesFm25Architecture:
    return TimesFm25Architecture.from_gguf(GGUFModelLoader(gguf_path, dtype=torch.float32)).eval()


@pytest.fixture(scope="module")
def engine(model: TimesFm25Architecture) -> ForecastEngine:
    return ForecastEngine(SimpleNamespace(tag="tiny", architecture=model))


def _stack(forecast: Forecast) -> torch.Tensor:
    return torch.tensor([forecast.mean, *forecast.quantiles.values()])


class TestArchitecture:
    def test_registry_resolves_timesfm25(self, gguf_path: Path) -> None:
        metadata = GGUFReader(gguf_path).read().metadata
        assert ArchitectureRegistry().resolve(metadata) is TimesFm25Architecture

    def test_predict_shapes(self, model: TimesFm25Architecture) -> None:
        series = torch.randn(3, 12)
        point, quantile = model.predict(series, torch.zeros(3, 12, dtype=torch.bool))
        assert point.shape == (3, HORIZON, N_OUT)
        assert quantile.shape == (3, QUANTILE_LEN, N_OUT)
        assert torch.isfinite(point).all() and torch.isfinite(quantile).all()

    def test_forward_is_refused(self, model: TimesFm25Architecture) -> None:
        with pytest.raises(TypeError):
            model.forward(torch.zeros(1, 4, dtype=torch.long))


class TestForecastEngine:
    def test_one_forecast_per_series_with_every_quantile(self, engine: ForecastEngine) -> None:
        results = engine.forecast([SERIES, SERIES[:5]], horizon=6)
        assert len(results) == 2
        for result in results:
            assert len(result.point) == len(result.mean) == 6
            assert list(result.quantiles) == ["0.1", "0.2", "0.3", "0.4", "0.5"] + [
                "0.6",
                "0.7",
                "0.8",
                "0.9",
            ]
            assert result.point == result.quantiles["0.5"]

    def test_padding_does_not_leak_between_series(self, engine: ForecastEngine) -> None:
        alone = engine.forecast([SERIES[:5]], horizon=HORIZON)[0]
        batched = engine.forecast([SERIES[:5], SERIES * 3], horizon=HORIZON)[0]
        assert torch.allclose(_stack(alone), _stack(batched), atol=1e-4)

    def test_scale_and_shift_carry_through(self, engine: ForecastEngine) -> None:
        base = engine.forecast([SERIES], HORIZON, clamp_non_negative=False)[0]
        moved = [3.0 * v - 20.0 for v in SERIES]
        result = engine.forecast([moved], HORIZON, clamp_non_negative=False)[0]
        assert torch.allclose(_stack(result), 3.0 * _stack(base) - 20.0, atol=1e-3)

    def test_negated_series_mirrors_the_forecast(self, engine: ForecastEngine) -> None:
        base = engine.forecast([SERIES], HORIZON, clamp_non_negative=False)[0]
        flipped = engine.forecast([[-v for v in SERIES]], HORIZON, clamp_non_negative=False)[0]
        assert torch.allclose(torch.tensor(flipped.point), -torch.tensor(base.point), atol=1e-4)
        low, high = flipped.quantiles["0.1"], base.quantiles["0.9"]
        assert torch.allclose(torch.tensor(low), -torch.tensor(high), atol=1e-4)

    def test_non_negative_series_never_forecasts_below_zero(self, engine: ForecastEngine) -> None:
        spiky = [0.0, 10.0, 0.0, 9.0, 0.0, 0.0, 12.0, 0.0]
        result = engine.forecast([spiky], HORIZON)[0]
        assert min(min(v) for v in result.quantiles.values()) >= 0.0

    def test_max_context_reads_only_the_latest_values(self, engine: ForecastEngine) -> None:
        cut = engine.forecast([SERIES], HORIZON, max_context=4)[0]
        tail = engine.forecast([SERIES[-4:]], HORIZON)[0]
        assert torch.allclose(_stack(cut), _stack(tail), atol=1e-5)

    def test_constant_series_stays_finite(self, engine: ForecastEngine) -> None:
        result = engine.forecast([[5.0] * 9], HORIZON)[0]
        assert torch.isfinite(_stack(result)).all()

    @pytest.mark.parametrize("horizon", [0, HORIZON + 1])
    def test_horizon_out_of_range(self, engine: ForecastEngine, horizon: int) -> None:
        with pytest.raises(InvalidForecastInputError):
            engine.forecast([SERIES], horizon)

    @pytest.mark.parametrize("series", [[], [1.0, float("nan")]])
    def test_bad_series(self, engine: ForecastEngine, series: list[float]) -> None:
        with pytest.raises(InvalidForecastInputError):
            engine.forecast([series], 2)
