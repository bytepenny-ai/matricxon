"""Pins in harder situations: null, a hand-edited .env past the limit, a voter-only node, a deleted
pinned model, an unwritable .env, and a slow models folder at start."""

from pathlib import Path

import pytest

from app.cluster.provisioning import EnvFile
from app.config import Settings
from app.models.catalog import ModelCatalog
from app.models.load_limit import LoadLimit
from app.models.manager import ModelManager
from app.models.pins import ModelPins, PinLimitError
from app.routers.delete_router import DeleteRequestHandler, PinRemover
from app.schemas.delete import DeleteRequest
from app.schemas.runtime_settings import RuntimeSettingsUpdate
from app.server.runtime_settings import RuntimeSettingsApplier
from tests.unit.test_model_manager import _install_tiny_model

TAGS = ("model-a:latest", "model-b:latest", "model-c:latest")


@pytest.fixture
def catalog(tmp_path: Path) -> ModelCatalog:
    for tag in TAGS:
        _install_tiny_model(tmp_path / "models", tag)
    return ModelCatalog(tmp_path / "models")


@pytest.fixture
def manager(catalog: ModelCatalog) -> ModelManager:
    return ModelManager(catalog, max_loaded=2)


def _pins(
    manager: ModelManager, catalog: ModelCatalog, tags: list[str], **kwargs: object
) -> ModelPins:
    installed = lambda: {m.tag: m.size_bytes for m in catalog.list_installed()}  # noqa: E731
    return ModelPins(manager, LoadLimit(catalog), installed, tags=tags, **kwargs)


def _applier(manager: ModelManager, pins: ModelPins, env: EnvFile | None) -> RuntimeSettingsApplier:
    return RuntimeSettingsApplier(manager, Settings(_env_file=None), pins, env)


def test_null_removes_every_pin_like_an_empty_list(manager, catalog, tmp_path) -> None:
    pins = _pins(manager, catalog, [])
    env = EnvFile(tmp_path / ".env")
    applier = _applier(manager, pins, env)
    applier.apply(RuntimeSettingsUpdate(pinned_models=["model-a:latest"]))
    assert pins.tags() == ["model-a:latest"]
    applier.apply(RuntimeSettingsUpdate.model_validate({"pinned_models": None}))
    assert pins.tags() == [] and "PINNED" not in (tmp_path / ".env").read_text()


def test_a_hand_edited_env_past_the_limit_fails_the_extra_pin_at_once(manager, catalog) -> None:
    pins = _pins(manager, catalog, list(TAGS))  # 3 chat pins, max_loaded 2: the API would refuse
    while (state := pins._due()) is not None:
        pins.load_one(state)
    report = {p.tag: p for p in pins.report()}
    assert sorted(p.state for p in report.values()) == ["failed", "loaded", "loaded"]
    failed = next(p for p in report.values() if p.state == "failed")
    assert failed.attempt == 0 and "over the limit" in failed.error  # no retries
    assert len(manager.list_loaded()) == 2


def test_a_voter_only_node_refuses_pins(manager, catalog) -> None:
    pins = _pins(manager, catalog, [], voter_only=True)
    with pytest.raises(PinLimitError, match="voter-only"):
        pins.check(["model-a:latest"], max_loaded=2)
    pins.check([], max_loaded=2)  # removing every pin is always fine


def test_deleting_a_pinned_model_removes_its_pin(manager, catalog, tmp_path) -> None:
    pins = _pins(manager, catalog, [])
    env = EnvFile(tmp_path / ".env")
    applier = _applier(manager, pins, env)
    applier.apply(RuntimeSettingsUpdate(pinned_models=["model-a:latest|ram", "model-b:latest"]))
    DeleteRequestHandler(catalog, manager, PinRemover(applier, pins)).handle(
        DeleteRequest(model="model-a:latest")
    )
    assert pins.tags() == ["model-b:latest"]
    assert (tmp_path / ".env").read_text() == "MATRICXON_PINNED_MODELS=model-b:latest\n"


class BrokenEnv(EnvFile):
    def set(self, values: dict[str, str]) -> None:
        raise PermissionError(".env is read-only")


def test_an_unwritable_env_changes_nothing(manager, catalog, tmp_path) -> None:
    pins = _pins(manager, catalog, ["model-a:latest"])
    applier = _applier(manager, pins, BrokenEnv(tmp_path / ".env"))
    with pytest.raises(PermissionError):
        applier.apply(RuntimeSettingsUpdate(pinned_models=["model-b:latest"]))
    assert pins.tags() == ["model-a:latest"] and applier._settings.pinned_models == ""


def test_starting_never_reads_the_models_folder(manager, catalog) -> None:
    reads: list[int] = []

    def slow_folder() -> dict[str, int]:
        reads.append(1)
        return {"model-a:latest": 10}

    pins = ModelPins(manager, LoadLimit(catalog), slow_folder, tags=["model-a:latest"])
    assert reads == []  # read later, by the loader thread
    assert pins._due().size == 10 and reads == [1]


def test_removing_a_pin_is_never_blocked_by_another_missing_model(manager, catalog) -> None:
    pins = _pins(
        manager, catalog, ["model-a:latest", "gone:latest"]
    )  # gone's model was deleted earlier
    pins.check(["gone:latest"], max_loaded=2)  # dropping model-a: fine, gone is not new
    with pytest.raises(PinLimitError, match="not installed"):
        pins.check(["gone:latest", "never:latest"], max_loaded=2)  # adding a missing one: refused


def test_a_deleted_model_fails_its_pin_at_once(manager, catalog) -> None:
    pins = _pins(manager, catalog, ["gone:latest"])
    pins.load_one(pins._due())
    report = pins.report()[0]
    assert (report.state, report.attempt) == ("failed", 1)  # no retries a minute apart
    assert "not installed any more" in report.error


def test_a_pin_load_that_ends_after_its_model_was_deleted_unloads_it(
    manager, catalog, monkeypatch
) -> None:
    pins = _pins(manager, catalog, ["model-a:latest"])
    real = manager.get_or_load

    def load_then_deleted(tag: str):
        handle = real(tag)
        pins.set([])  # the pin is removed while it loads (pAIring's cleanup after a delete)
        catalog.delete(tag)
        return handle

    monkeypatch.setattr(manager, "get_or_load", load_then_deleted)
    pins.load_one(pins._due())
    assert manager.list_loaded() == []  # not kept as an ordinary model on a deleted file


def test_a_pin_load_that_ends_after_unpinning_keeps_an_installed_model(
    manager, catalog, monkeypatch
) -> None:
    pins = _pins(manager, catalog, ["model-a:latest"])
    real = manager.get_or_load

    def load_then_unpinned(tag: str):
        handle = real(tag)
        pins.set([])
        return handle

    monkeypatch.setattr(manager, "get_or_load", load_then_unpinned)
    pins.load_one(pins._due())
    loaded = manager.list_loaded()
    assert [h.tag for h in loaded] == ["model-a:latest"] and not loaded[
        0
    ].pinned  # ordinary: keep-alive applies
