import asyncio

from fastapi import APIRouter, Depends

from app.dependencies import get_model_manager
from app.models.manager import ModelManager
from app.models.worker_call import WorkerCall
from app.runtime.forecast_engine import ForecastEngine
from app.schemas.forecast import ForecastRequest, ForecastResponse, SeriesForecast
from app.server.api_keys import require_api_key
from app.server.node_kind import require_compute

router = APIRouter()


class ForecastRequestHandler:
    """Handles `POST /api/forecast`: forecasts each series with a time-series model."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._model_manager = model_manager

    async def handle(self, request: ForecastRequest) -> ForecastResponse:
        handle = await asyncio.to_thread(
            self._model_manager.get_or_load, request.model, request.keep_alive
        )
        engine = ForecastEngine(handle)
        forecasts = await WorkerCall.collect(
            handle.worker,
            lambda: iter(
                engine.forecast(
                    request.series,
                    request.horizon,
                    request.max_context,
                    request.clamp_non_negative,
                )
            ),
        )
        return ForecastResponse(
            model=request.model,
            horizon=request.horizon,
            forecasts=[
                SeriesForecast(point=f.point, mean=f.mean, quantiles=f.quantiles) for f in forecasts
            ],
        )


@router.post(
    "/api/forecast",
    dependencies=[Depends(require_compute), Depends(require_api_key)],
    response_model=ForecastResponse,
)
async def post_forecast(
    request: ForecastRequest, model_manager: ModelManager = Depends(get_model_manager)
) -> ForecastResponse:
    return await ForecastRequestHandler(model_manager).handle(request)
