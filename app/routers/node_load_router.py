from fastapi import APIRouter, Depends

from app.config import settings
from app.dependencies import get_memory_info, get_model_manager, get_model_pins
from app.models.manager import ModelManager
from app.models.pins import ModelPins
from app.runtime.cpu_cores import CpuCores
from app.runtime.memory_info import MemoryInfo
from app.schemas.node_load import LoadedModelLoad, NodeLoadResponse, PinStatus

router = APIRouter()


class NodeLoadHandler:
    """This node's load for cluster routing: unlike a client's own in-flight count, `active` covers
    every client (several pAIring instances, other tools). Cheap: no disk or network work."""

    def __init__(self, model_manager: ModelManager, pins: ModelPins, memory: MemoryInfo) -> None:
        self._model_manager = model_manager
        self._pins = pins
        self._memory = memory

    def handle(self) -> NodeLoadResponse:
        now = self._model_manager.now()
        pins = [
            PinStatus(tag=p.tag, memory=p.memory, state=p.state, attempt=p.attempt, error=p.error)
            for p in self._pins.report()
        ]
        models = [
            LoadedModelLoad(
                name=h.tag,
                size_bytes=h.size_bytes,
                active=h.worker.busy_count,
                pinned=h.pinned,
                memory="vram" if h.on_gpu else "ram",
                counted=h.counts_toward_limit,
                expires_in_seconds=None if h.pinned else max(0, int(h.expires_in(now))),
            )
            for h in self._model_manager.list_loaded()
        ]
        return NodeLoadResponse(
            active=sum(m.active for m in models),
            models=models,
            threads=CpuCores.threads(settings.torch_threads),
            free_memory_bytes=self._memory.free_ram_bytes(),
            node_kind=settings.node_kind,
            total_memory_bytes=self._memory.total_ram_bytes(),
            gpu=self._memory.gpu(),
            runs_on="gpu" if self._model_manager.on_gpu else "cpu",
            max_loaded_models=self._model_manager.settings.max_loaded,
            pins=pins,
        )


@router.get("/api/node/load", response_model=NodeLoadResponse)
def get_node_load(
    model_manager: ModelManager = Depends(get_model_manager),
    pins: ModelPins = Depends(get_model_pins),
    memory: MemoryInfo = Depends(get_memory_info),
) -> NodeLoadResponse:
    return NodeLoadHandler(model_manager, pins, memory).handle()
