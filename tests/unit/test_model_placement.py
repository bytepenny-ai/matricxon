"""Where a model loads (app.models.placement) and pins that choose RAM or VRAM (app.models.pins)."""

from pathlib import Path

import pytest

from app.models.catalog import ModelCatalog
from app.models.load_limit import LoadLimit
from app.models.manager import ModelManager
from app.models.pins import ModelPins, PinLimitError, normalize_pins, parse_pins, split_pin
from app.models.placement import ModelPlacement
from app.server.errors import DeviceUnavailableError
from tests.unit.test_model_manager import _install_tiny_model


class Device:
    def __init__(self, gpu: bool) -> None:
        self.is_gpu = gpu


GPU, CPU = Device(True), Device(False)


def _placement(server: Device, gpu: Device | None) -> ModelPlacement:
    placement = ModelPlacement.__new__(ModelPlacement)
    placement._device, placement._cpu, placement._gpu, placement._gpu_checked = (
        server,
        CPU,
        gpu,
        True,
    )
    return placement


def test_pins_carry_their_memory() -> None:
    assert split_pin(" m:1 | vram ") == ("m:1", "vram") and split_pin("m:1") == ("m:1", "auto")
    assert normalize_pins(["a|auto", "b|ram", "a|vram", " "]) == ["a|vram", "b|ram"]
    with pytest.raises(PinLimitError, match="auto, ram or vram"):
        split_pin("m|disk")


def test_a_bad_env_entry_is_skipped_not_fatal() -> None:
    assert parse_pins("a, b|disk, c|ram") == ["a", "c|ram"]


def test_each_wish_picks_its_device() -> None:
    on_gpu_server = _placement(GPU, GPU)
    assert on_gpu_server.device_for(None) is GPU and on_gpu_server.device_for("ram") is CPU
    assert (
        on_gpu_server.device_for("auto") is GPU
        and on_gpu_server.device_for("auto", fits_on_gpu=False) is CPU
    )
    cpu_server_with_gpu = _placement(CPU, GPU)
    assert (
        cpu_server_with_gpu.device_for("vram") is GPU
        and cpu_server_with_gpu.device_for("auto") is CPU
    )
    with pytest.raises(DeviceUnavailableError, match="no GPU"):
        _placement(CPU, None).device_for("vram")


def test_misplaced_models_load_again() -> None:
    assert ModelPlacement.misplaced("ram", on_gpu=True) and ModelPlacement.misplaced(
        "vram", on_gpu=False
    )
    assert not ModelPlacement.misplaced("auto", on_gpu=True) and not ModelPlacement.misplaced(
        None, on_gpu=False
    )


@pytest.fixture
def manager(tmp_path: Path) -> ModelManager:
    _install_tiny_model(tmp_path / "models", "model-a:latest")
    return ModelManager(ModelCatalog(tmp_path / "models"), max_loaded=2)


def _pins(manager: ModelManager, tags: list[str], gpu: Device | None) -> ModelPins:
    catalog = manager._builder._catalog
    installed = lambda: {m.tag: m.size_bytes for m in catalog.list_installed()}  # noqa: E731
    return ModelPins(
        manager, LoadLimit(catalog), installed, tags=tags, placement=_placement(CPU, gpu)
    )


def test_vram_pins_are_refused_without_a_gpu(manager: ModelManager) -> None:
    with pytest.raises(PinLimitError, match="no GPU"):
        _pins(manager, [], gpu=None).check(["model-a:latest|vram"], max_loaded=2)
    _pins(manager, [], gpu=GPU).check(["model-a:latest|vram"], max_loaded=2)


def test_the_loader_gets_each_pins_memory(manager: ModelManager) -> None:
    pins = _pins(manager, ["model-a:latest|ram"], gpu=None)
    assert manager.settings.model_memory == {"model-a:latest": "ram"} and pins.tags() == [
        "model-a:latest|ram"
    ]
    pins.set([])
    assert manager.settings.model_memory == {}


def test_a_pin_moved_to_the_other_memory_loads_again(manager: ModelManager) -> None:
    pins = _pins(manager, ["model-a:latest"], gpu=None)
    pins.load_one(pins._due())
    first = manager.list_loaded()[0]
    first.on_gpu = True  # as if it had loaded on a GPU
    pins.set(["model-a:latest|ram"])
    assert [p.state for p in pins.report()] == ["loading"]
    pins.load_one(pins._due())
    again = manager.list_loaded()[0]
    assert again is not first and not again.on_gpu and again.pinned
    assert [(p.memory, p.state) for p in pins.report()] == [("ram", "loaded")]
