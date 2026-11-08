"""Pinned models (app.models.pins): kept loaded, loaded in the background, retried, reported."""

import threading
import time
from pathlib import Path

import pytest

from app.cluster.provisioning import EnvFile
from app.config import Settings
from app.models.catalog import ModelCatalog
from app.models.load_limit import LoadLimit
from app.models.manager import ModelManager
from app.models.pins import ModelPins, PinLimitError, parse_pins
from app.runtime.memory_info import MemoryInfo
from app.schemas.runtime_settings import RuntimeSettingsUpdate
from app.server.runtime_settings import RuntimeSettingsApplier
from tests.unit.test_model_manager import _install_tiny_model

TAGS = ("model-a:latest", "model-b:latest", "model-c:latest")


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def manager(tmp_path: Path, clock: Clock) -> ModelManager:
    for tag in TAGS:
        _install_tiny_model(tmp_path / "models", tag)
    catalog = ModelCatalog(tmp_path / "models")
    return ModelManager(catalog, max_loaded=2, default_keep_alive_seconds=300, clock=clock)


def _pins(manager: ModelManager, tags: list[str], **kwargs: object) -> ModelPins:
    catalog = manager._builder._catalog
    installed = lambda: {m.tag: m.size_bytes for m in catalog.list_installed()}  # noqa: E731
    return ModelPins(manager, LoadLimit(catalog), installed, tags=tags, **kwargs)


def _load_all(pins: ModelPins) -> None:
    while (state := pins._due()) is not None:
        pins.load_one(state)


def test_parse_pins_drops_blanks_and_repeats() -> None:
    assert parse_pins(" a:1, ,b:2,a:1 ") == ["a:1", "b:2"] and parse_pins("") == []


def test_a_pinned_model_never_expires(manager: ModelManager, clock: Clock) -> None:
    pins = _pins(manager, ["model-a:latest"])
    _load_all(pins)
    clock.now += 10_000  # far past the 300 s keep-alive
    assert [h.tag for h in manager.list_loaded()] == ["model-a:latest"]
    assert [(p.tag, p.state) for p in pins.report()] == [("model-a:latest", "loaded")]


def test_a_pinned_model_is_never_evicted_to_make_room(manager: ModelManager, clock: Clock) -> None:
    pins = _pins(manager, ["model-a:latest"])
    _load_all(pins)
    manager.get_or_load("model-b:latest")
    clock.now += 1
    manager.get_or_load("model-c:latest")  # the limit is 2: b goes, never the pinned a
    assert sorted(h.tag for h in manager.list_loaded()) == ["model-a:latest", "model-c:latest"]


def test_unpinning_makes_it_an_ordinary_model(manager: ModelManager, clock: Clock) -> None:
    pins = _pins(manager, ["model-a:latest"])
    _load_all(pins)
    pins.set([])
    assert not manager.list_loaded()[0].pinned
    clock.now += 301
    assert manager.list_loaded() == []  # keep-alive applies again


def test_unloaded_by_hand_then_saved_again_loads_again(manager: ModelManager) -> None:
    pins = _pins(manager, ["model-a:latest"])
    _load_all(pins)
    manager.unload("model-a:latest")
    assert [p.state for p in pins.report()] == ["unloaded"]
    pins.set(["model-a:latest"])
    assert [p.state for p in pins.report()] == ["loading"]
    _load_all(pins)
    assert [p.state for p in pins.report()] == ["loaded"] and manager.list_loaded()[0].pinned


class FailingManager:
    settings = type("Limits", (), {"max_loaded": 2, "model_memory": {}})()

    def get_or_load(self, tag: str) -> None:
        raise MemoryError("not enough memory to load it")

    def list_loaded(self) -> list:
        return []


def test_a_failing_pin_is_retried_then_given_up(clock: Clock) -> None:
    helper = type("Helpers", (), {"counts": lambda self, tag: False})()
    pins = ModelPins(FailingManager(), helper, lambda: {"x": 1}, tags=["x"], clock=clock)
    for attempt in (1, 2):
        pins.load_one(pins._due())
        assert pins._due() is None  # waits RETRY_SECONDS before the next try
        clock.now += ModelPins.RETRY_SECONDS
        assert pins.report()[0].state == "loading" and pins.report()[0].attempt == attempt
    pins.load_one(pins._due())
    report = pins.report()[0]
    assert (report.state, report.attempt, report.error) == (
        "failed",
        3,
        "not enough memory to load it",
    )
    assert pins._due() is None  # given up until saved again or a restart


def test_the_background_thread_loads_pins(manager: ModelManager) -> None:
    pins = _pins(manager, ["model-a:latest"])
    pins.start()
    try:
        deadline = time.monotonic() + 30
        while pins.report()[0].state != "loaded" and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        pins.stop()
    assert pins.report()[0].state == "loaded"


def test_too_many_chat_pins_or_a_missing_model_are_refused(manager: ModelManager) -> None:
    pins = _pins(manager, [])
    pins.check(["model-a:latest", "model-b:latest"], max_loaded=2)
    with pytest.raises(PinLimitError, match="at most 2"):
        pins.check(list(TAGS), max_loaded=2)
    with pytest.raises(PinLimitError, match="not installed"):
        pins.check(["nope:latest"], max_loaded=2)


def test_runtime_settings_save_pins_to_env_and_refuse_a_lower_limit(
    manager: ModelManager, tmp_path: Path
) -> None:
    pins, env = _pins(manager, []), EnvFile(tmp_path / ".env")
    settings = Settings(_env_file=None)
    applier = RuntimeSettingsApplier(manager, settings, pins, env)
    result = applier.apply(
        RuntimeSettingsUpdate(pinned_models=["model-a:latest", "model-b:latest"])
    )
    assert "pinned_models" in result.applied_now
    assert (
        tmp_path / ".env"
    ).read_text() == "MATRICXON_PINNED_MODELS=model-a:latest,model-b:latest\n"
    assert pins.tags() == ["model-a:latest", "model-b:latest"]
    with pytest.raises(PinLimitError):
        applier.apply(RuntimeSettingsUpdate(max_loaded_models=1))
    assert settings.max_loaded_models == 2  # refused: nothing changed
    applier.apply(RuntimeSettingsUpdate(pinned_models=[]))
    assert (tmp_path / ".env").read_text() == "" and pins.tags() == []


def test_memory_info_reads_total_ram(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16384000 kB\nMemAvailable:    8000000 kB\n")

    class NoGpu:
        def gpu(self) -> None:
            return None

    info = MemoryInfo(NoGpu(), str(meminfo))
    assert info.total_ram_bytes() == 16384000 * 1024 and info.gpu() is None
    assert MemoryInfo(NoGpu(), str(tmp_path / "missing")).total_ram_bytes() is None


def test_smaller_pins_load_first(manager: ModelManager) -> None:
    sizes = {"big:1": 3_000, "tiny:1": 20, "mid:1": 500}
    pins = ModelPins(manager, LoadLimit(None), lambda: sizes, tags=list(sizes))
    assert pins._due().tag == "tiny:1"
    pins._states["tiny:1"].state = "loaded"
    assert pins._due().tag == "mid:1"


@pytest.mark.parametrize("pin_first", [True, False])
def test_a_chat_during_a_pin_load_shares_the_one_load(
    manager: ModelManager, monkeypatch: pytest.MonkeyPatch, pin_first: bool
) -> None:
    real_load = manager._load
    loads: list[str] = []

    def slow_load(tag: str, keep_alive_seconds: int | None, now: float):
        loads.append(tag)
        time.sleep(0.3)  # the other caller arrives mid-load
        return real_load(tag, keep_alive_seconds, now)

    monkeypatch.setattr(manager, "_load", slow_load)
    pins = _pins(manager, ["model-a:latest"])
    chat: list = []
    pin = threading.Thread(target=lambda: pins.load_one(pins._due()))
    ask = threading.Thread(target=lambda: chat.append(manager.get_or_load("model-a:latest")))
    first, second = (pin, ask) if pin_first else (ask, pin)
    first.start()
    time.sleep(0.05)
    second.start()
    first.join()
    second.join()
    assert loads == ["model-a:latest"]  # one load, not two
    assert chat[0] is manager.list_loaded()[0] and chat[0].pinned
    assert [p.state for p in pins.report()] == ["loaded"]
