"""Applies setting changes to a running server, so most Settings need no restart."""

import logging

from app.cluster.provisioning import EnvFile
from app.config import Settings
from app.models.manager import ModelManager
from app.models.pins import ModelPins, normalize_pins
from app.native.fused_ops import FusedOps
from app.native.gemm import NativeGemm
from app.schemas.runtime_settings import RuntimeSettingsResult, RuntimeSettingsUpdate
from app.server.log_filters import LOG_LEVELS_BY_SETTING

logger = logging.getLogger(__name__)

# Settings name -> ModelManager.set_limits keyword.
_MANAGER_NOW = {
    "max_loaded_models": "max_loaded",
    "default_keep_alive_seconds": "default_keep_alive_seconds",
    "memory_safety_margin": "memory_safety_margin",
    "prompt_cache_slots": "prompt_cache_slots",
    "prompt_cache_budget_mb": "prompt_cache_budget_mb",
    "max_decode_batch": "max_decode_batch",
}
_MANAGER_NEXT_LOAD = {
    "enable_quantized_native_compute": "enable_quantized_native_compute",
    "gpu_weight_mode": "gpu_weight_mode",
}


class RuntimeSettingsApplier:
    """Writes each changed value to `settings` (so routers read it) and to the live objects."""

    PINS_ENV = "MATRICXON_PINNED_MODELS"

    def __init__(
        self,
        manager: ModelManager,
        settings: Settings,
        pins: ModelPins | None = None,
        env_file: EnvFile | None = None,
    ) -> None:
        self._manager = manager
        self._settings = settings
        self._pins = pins
        self._env_file = env_file

    def _apply_pins(self, values: dict[str, object]) -> None:
        """Checks the pins against the (new) limit before anything changes: raises PinLimitError.
        Saved in .env, so they load again at the next start."""
        if self._pins is None:
            return
        sent = "pinned_models" in values
        tags = values.pop("pinned_models", None)
        if not sent and "max_loaded_models" not in values:
            return
        # null comes here as the default ("") and means no pins, like [].
        wanted = normalize_pins(tags if isinstance(tags, list) else []) if sent else None
        max_loaded = int(values.get("max_loaded_models", self._settings.max_loaded_models))
        self._pins.check(wanted if wanted is not None else self._pins.tags(), max_loaded)
        if wanted is None:
            return
        # .env first: if it can't be written, nothing has changed (no pins that vanish at restart).
        if self._env_file is not None:
            if wanted:
                self._env_file.set({self.PINS_ENV: ",".join(wanted)})
            else:
                self._env_file.remove([self.PINS_ENV])
        self._settings.pinned_models = ",".join(wanted)
        self._pins.set(wanted)

    def _values(self, change: RuntimeSettingsUpdate) -> dict[str, object]:
        """Fields sent in the request; a null means the default."""
        values = {}
        for name in change.model_fields_set:
            value = getattr(change, name)
            values[name] = Settings.model_fields[name].default if value is None else value
        return values

    def apply(self, change: RuntimeSettingsUpdate) -> RuntimeSettingsResult:
        values = self._values(change)
        pinned = "pinned_models" in values
        self._apply_pins(values)
        for name, value in values.items():
            setattr(self._settings, name, value)

        limits = {_MANAGER_NOW[n]: v for n, v in values.items() if n in _MANAGER_NOW}
        limits |= {_MANAGER_NEXT_LOAD[n]: v for n, v in values.items() if n in _MANAGER_NEXT_LOAD}
        if limits:
            self._manager.set_limits(**limits)
        if "log_level" in values:
            logging.getLogger().setLevel(LOG_LEVELS_BY_SETTING[self._settings.log_level])
        if "gemv_backend" in values:
            # Models load their kernels at load time, so loaded ones keep theirs.
            NativeGemm.configure(
                self._settings.gemv_backend, self._settings.torch_threads, self._settings.wide_simd
            )
            FusedOps.configure(self._settings.enable_fused_ops)

        next_load = {*_MANAGER_NEXT_LOAD, "gemv_backend"}
        now = [n for n in values if n not in next_load] + (["pinned_models"] if pinned else [])
        result = RuntimeSettingsResult(
            applied_now=sorted(now),
            applied_next_load=sorted(n for n in values if n in next_load),
        )
        logger.info("settings changed without restart: %s", ", ".join(sorted(now)) or "none")
        return result
