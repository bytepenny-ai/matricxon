"""Forecasts time series with a forecasting model (TimesFM 2.5). Pre- and post-processing follow
HF transformers' `TimesFm2_5ModelForPrediction` defaults: flip invariance, the continuous quantile
head and the non-negative clamp."""

import math
from dataclasses import dataclass

import torch

from app.architectures.timesfm25 import TimesFm25Architecture
from app.architectures.timesfm25_revin import RevIN
from app.models.handle import ModelHandle
from app.server.errors import InvalidForecastInputError, WrongModelKindError


@dataclass
class Forecast:
    """One series' forecast: `point` is the median, `mean` the mean head, `quantiles` by level."""

    point: list[float]
    mean: list[float]
    quantiles: dict[str, list[float]]


class ForecastEngine:
    def __init__(self, handle: ModelHandle) -> None:
        model = handle.architecture
        if not isinstance(model, TimesFm25Architecture):
            raise WrongModelKindError(f"{handle.tag} is not a forecasting model.")
        self._model = model

    @property
    def max_horizon(self) -> int:
        return self._model.horizon_len

    def forecast(
        self,
        series: list[list[float]],
        horizon: int,
        max_context: int | None = None,
        clamp_non_negative: bool = True,
    ) -> list[Forecast]:
        """One `Forecast` per input series, `horizon` steps each (at most `max_horizon`)."""
        if not 1 <= horizon <= self.max_horizon:
            raise InvalidForecastInputError(f"horizon must be between 1 and {self.max_horizon}")
        context = min(max_context or self._model.context_limit, self._model.context_limit)
        inputs = [self._checked(values)[-context:] for values in series]
        padded_values, padded = self._left_pad(inputs)
        mu, sigma = self._global_stats(inputs)
        normed = RevIN.normalize(padded_values, mu, sigma).masked_fill(padded, 0.0)

        with torch.no_grad():
            point, quantile = self._model.predict(
                torch.cat([normed, -normed]), torch.cat([padded, padded])
            )
        count = len(inputs)
        point = (point[:count] - self._flip(point[count:])) / 2
        quantile = (quantile[:count] - self._flip(quantile[count:])) / 2
        full = self._merge_quantile_head(point[:, :horizon], quantile[:, :horizon])
        full = RevIN.denormalize(full, mu, sigma)
        return [
            self._result(full[i], clamp_non_negative and min(inputs[i]) >= 0) for i in range(count)
        ]

    @staticmethod
    def _checked(values: list[float]) -> list[float]:
        if not values:
            raise InvalidForecastInputError("every series needs at least one value")
        if not all(math.isfinite(v) for v in values):
            raise InvalidForecastInputError("series values must be finite numbers")
        return values

    def _left_pad(self, inputs: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-pads every series on the left to one length, a multiple of the patch length."""
        patch = self._model.patch_len
        length = math.ceil(max(len(v) for v in inputs) / patch) * patch
        values = torch.zeros(len(inputs), length)
        padded = torch.ones(len(inputs), length, dtype=torch.bool)
        for i, v in enumerate(inputs):
            values[i, length - len(v) :] = torch.tensor(v, dtype=torch.float32)
            padded[i, length - len(v) :] = False
        return values, padded

    @staticmethod
    def _global_stats(inputs: list[list[float]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and std over each series' real values (the model is invariant to them; they only
        keep the numbers in a sane range)."""
        stats = [torch.tensor(v, dtype=torch.float64) for v in inputs]
        mu = torch.stack([s.mean() for s in stats]).float()
        sigma = torch.stack([s.std(correction=0) for s in stats]).float()
        return mu, sigma

    @staticmethod
    def _flip(x: torch.Tensor) -> torch.Tensor:
        """The forecast of the negated series, read back: mean stays, quantiles reverse."""
        return torch.cat([x[..., :1], x[..., 1:].flip(-1)], dim=-1)

    def _merge_quantile_head(self, point: torch.Tensor, quantile: torch.Tensor) -> torch.Tensor:
        """Non-median quantiles from the quantile head, shifted onto the point head's median."""
        median = self._model.decode_index
        full = point.clone()
        for idx in range(1, full.shape[-1]):
            if idx != median:
                full[..., idx] = quantile[..., idx] - quantile[..., median] + point[..., median]
        return full

    def _result(self, full: torch.Tensor, clamp: bool) -> Forecast:
        if clamp:
            full = full.clamp(min=0.0)
        levels = self._model.quantiles
        return Forecast(
            point=full[:, self._model.decode_index].tolist(),
            mean=full[:, 0].tolist(),
            quantiles={f"{q:g}": full[:, i + 1].tolist() for i, q in enumerate(levels)},
        )
