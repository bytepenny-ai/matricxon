"""GET /api/node/load: how busy this node is, for cluster routing (pAIring's pool), and its memory
and pins for pAIring's Memory pinning page."""

from typing import Literal

from pydantic import BaseModel


class LoadedModelLoad(BaseModel):
    name: str
    size_bytes: int
    active: int  # replies queued or running on this model
    pinned: bool = False
    memory: Literal["ram", "vram"] = "ram"  # where its weights are
    counted: bool = True  # takes one of max_loaded_models (chat models; helpers don't)
    expires_in_seconds: int | None = None  # until its keep-alive unloads it; None while pinned


class GpuMemory(BaseModel):
    name: str
    total_bytes: int
    free_bytes: int


class PinStatus(BaseModel):
    """One pin (app.models.pins): only `loaded` counts as pinned for routing."""

    tag: str
    memory: Literal["auto", "ram", "vram"] = "auto"  # where it loads (app.models.placement)
    state: Literal["loading", "loaded", "failed", "unloaded"]
    attempt: int  # tries so far (of ModelPins.RETRIES)
    error: str | None = None


class NodeLoadResponse(BaseModel):
    active: int  # replies queued or running on every model, from every client
    models: list[LoadedModelLoad]
    threads: int  # compute threads (MATRICXON_TORCH_THREADS, else CPU count)
    free_memory_bytes: int | None  # MemAvailable; None when unknown
    node_kind: str  # "compute" or "voter"
    total_memory_bytes: int | None = None  # MemTotal; None when unknown
    gpu: GpuMemory | None = (
        None  # the GPU this server has (also when it runs on the CPU); None without
    )
    runs_on: Literal["cpu", "gpu"] = (
        "cpu"  # MATRICXON_DEVICE: where models load unless pinned elsewhere
    )
    max_loaded_models: int = 0  # chat models kept at once; pinned chat models count toward it
    pins: list[PinStatus] = []
