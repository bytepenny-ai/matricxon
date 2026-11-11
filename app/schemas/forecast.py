from typing import Annotated

from pydantic import BaseModel, Field, field_validator

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
# Longest series accepted; only the last `max_context` values (at most the model's limit) are read.
MAX_SERIES_LENGTH = 100_000


class ForecastRequest(BaseModel):
    model: str
    # One or more series, oldest value first; each gets its own forecast.
    series: list[list[FiniteFloat]] = Field(min_length=1, max_length=64)
    # Steps to forecast (the model's limit: 128 for TimesFM 2.5).
    horizon: int = Field(ge=1)
    # Read at most this many of the latest values per series (default: the model's limit).
    max_context: int | None = Field(default=None, ge=1)
    # Clamp a forecast at 0 when its series has no negative values.
    clamp_non_negative: bool = True
    keep_alive: int | None = None

    @field_validator("series")
    @classmethod
    def _check_lengths(cls, series: list[list[float]]) -> list[list[float]]:
        for values in series:
            if not 1 <= len(values) <= MAX_SERIES_LENGTH:
                raise ValueError(f"each series needs 1 to {MAX_SERIES_LENGTH} values")
        return series


class SeriesForecast(BaseModel):
    # The median forecast.
    point: list[float]
    # The mean head's forecast.
    mean: list[float]
    # Forecast per quantile level, e.g. "0.1" ... "0.9".
    quantiles: dict[str, list[float]]


class ForecastResponse(BaseModel):
    model: str
    horizon: int
    # One per input series, in order.
    forecasts: list[SeriesForecast]
