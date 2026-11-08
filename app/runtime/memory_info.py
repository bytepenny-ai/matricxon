"""This machine's memory for reports (GET /api/node/load): total RAM, and the GPU's when on one."""

from app.models.load_dtype import available_memory_bytes
from app.models.placement import ModelPlacement
from app.schemas.node_load import GpuMemory


class MemoryInfo:
    def __init__(self, placement: ModelPlacement, meminfo: str = "/proc/meminfo") -> None:
        self._placement = placement
        self._meminfo = meminfo

    def total_ram_bytes(self) -> int | None:
        """MemTotal; None off Linux."""
        try:
            with open(self._meminfo) as f:
                line = next((line for line in f if line.startswith("MemTotal:")), "")
        except OSError:
            return None
        parts = line.split()
        return int(parts[1]) * 1024 if len(parts) >= 2 else None

    @staticmethod
    def free_ram_bytes() -> int | None:
        return available_memory_bytes()

    def gpu(self) -> GpuMemory | None:
        """The GPU's name and memory (also on a CPU server that has one: models may be pinned to
        it); None without a GPU or when it can't be read."""
        gpu = self._placement.gpu()
        if gpu is None:
            return None
        try:
            total, free = gpu.total_memory_bytes(), gpu.free_memory_bytes()
        except RuntimeError:
            return None
        if total is None or free is None:
            return None
        return GpuMemory(name=gpu.name, total_bytes=total, free_bytes=free)
