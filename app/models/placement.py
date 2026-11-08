"""Where one model loads: the server's device (MATRICXON_DEVICE) unless it has its own wish
(LoadSettings.model_memory, set by pins: app.models.pins).

- "ram": the CPU, also on a GPU server.
- "vram": a GPU, also on a server that runs on the CPU (when PyTorch sees one); none: refused.
- "auto": a GPU when one is used and the model fits in its free memory, else the CPU.
"""

from app.runtime.compute_device import ComputeDevice
from app.server.errors import DeviceUnavailableError

MEMORY_WISHES = ("auto", "ram", "vram")


class ModelPlacement:
    def __init__(self, device: ComputeDevice) -> None:
        self._device = device
        self._cpu = ComputeDevice.resolve("cpu")
        self._gpu: ComputeDevice | None = device if device.is_gpu else None
        self._gpu_checked = device.is_gpu

    @property
    def server_device(self) -> ComputeDevice:
        return self._device

    def gpu(self) -> ComputeDevice | None:
        """The server's GPU, or the first one PyTorch sees (read once); None without one."""
        if not self._gpu_checked:
            self._gpu_checked = True
            for spec in ("cuda", "mps"):
                try:
                    self._gpu = ComputeDevice.resolve(spec)
                    break
                except DeviceUnavailableError:
                    continue
        return self._gpu

    def device_for(self, wish: str | None, fits_on_gpu: bool = True) -> ComputeDevice:
        """The device for a model with this wish; `fits_on_gpu` decides "auto"."""
        if wish == "ram":
            return self._cpu
        if wish == "vram":
            gpu = self.gpu()
            if gpu is None:
                raise DeviceUnavailableError("this server has no GPU: pin the model to RAM instead")
            return gpu
        if wish == "auto" and self._device.is_gpu and not fits_on_gpu:
            return self._cpu
        return self._device

    @staticmethod
    def misplaced(wish: str | None, on_gpu: bool) -> bool:
        """A loaded model that sits on the other memory than its wish (it must load again)."""
        return (wish == "ram" and on_gpu) or (wish == "vram" and not on_gpu)
