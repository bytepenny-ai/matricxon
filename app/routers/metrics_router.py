from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from app.dependencies import get_memory_info, get_model_manager, get_model_pins
from app.models.manager import ModelManager
from app.models.pins import ModelPins
from app.routers.node_load_router import NodeLoadHandler
from app.runtime.memory_info import MemoryInfo
from app.server.api_keys import require_api_key
from app.server.metrics import Metrics

router = APIRouter()


class MetricsHandler:
    """`GET /metrics`: the counters (app.server.metrics) plus how busy the node is right now."""

    def __init__(self, model_manager: ModelManager, pins: ModelPins, memory: MemoryInfo) -> None:
        self._load = NodeLoadHandler(model_manager, pins, memory)

    def handle(self) -> str:
        load = self._load.handle()
        gauges = {
            "matricxon_active_replies": ("Replies queued or running now.", load.active),
            "matricxon_loaded_models": ("Models loaded in memory.", len(load.models)),
            "matricxon_loaded_model_bytes": (
                "Size of the loaded models' files.",
                sum(m.size_bytes for m in load.models),
            ),
            "matricxon_compute_threads": ("Compute threads.", load.threads),
            "matricxon_pinned_models": (
                "Pinned models loaded.",
                sum(m.pinned for m in load.models),
            ),
        }
        if load.free_memory_bytes is not None:
            gauges["matricxon_free_memory_bytes"] = ("Memory available.", load.free_memory_bytes)
        return Metrics.shared().text(gauges)


@router.get("/metrics", dependencies=[Depends(require_api_key)], response_class=PlainTextResponse)
def get_metrics(
    model_manager: ModelManager = Depends(get_model_manager),
    pins: ModelPins = Depends(get_model_pins),
    memory: MemoryInfo = Depends(get_memory_info),
) -> PlainTextResponse:
    return PlainTextResponse(
        MetricsHandler(model_manager, pins, memory).handle(), media_type="text/plain; version=0.0.4"
    )
