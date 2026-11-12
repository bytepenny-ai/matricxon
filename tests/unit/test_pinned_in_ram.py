"""A pinned model's weights are copied into RAM at load (app.gguf.mmap_source), never re-read from
disk: a mapped file's pages are dropped when RAM is tight, and reading them back from a slow disk
cost a short reply about a minute. Unpinned models stay mapped."""

import os
from pathlib import Path

import pytest

from app.gguf.mmap_source import GGUFMemoryMap
from app.models.catalog import ModelCatalog
from app.models.manager import ModelManager
from app.models.weight_prefetch import WeightPrefetch
from tests.unit.test_model_manager import _install_tiny_model
from tests.unit.test_model_pins import TAGS, Clock, _load_all, _pins


@pytest.fixture
def manager(tmp_path: Path) -> ModelManager:
    for tag in TAGS:
        _install_tiny_model(tmp_path / "models", tag)
    catalog = ModelCatalog(tmp_path / "models")
    return ModelManager(catalog, max_loaded=2, default_keep_alive_seconds=300, clock=Clock())


def test_a_resident_map_holds_the_whole_file_and_is_never_prefetched(tmp_path: Path) -> None:
    path = tmp_path / "weights.gguf"
    data = os.urandom(3 * 1024 * 1024 + 17)
    path.write_bytes(data)
    resident = GGUFMemoryMap(path, resident=True)
    assert bytes(resident._mmap[:]) == data and resident.resident
    assert WeightPrefetch.start(path) is False  # in RAM already
    resident.close()
    assert path not in WeightPrefetch._resident
    mapped = GGUFMemoryMap(path)
    assert not mapped.resident and bytes(mapped._mmap[:10]) == data[:10]
    mapped.close()


def test_a_pinned_model_loads_into_ram_and_an_ordinary_one_stays_mapped(
    manager: ModelManager,
) -> None:
    pins = _pins(manager, ["model-a:latest"])
    _load_all(pins)
    ordinary = manager.get_or_load("model-b:latest")
    loaded = {h.tag: h.resident for h in manager.list_loaded()}
    assert loaded == {"model-a:latest": True, "model-b:latest": False} and not ordinary.resident


def test_a_model_loaded_before_it_was_pinned_is_loaded_again_into_ram(
    manager: ModelManager,
) -> None:
    first = manager.get_or_load("model-a:latest")
    assert not first.resident
    pins = _pins(manager, ["model-a:latest"])
    _load_all(pins)
    (handle,) = manager.list_loaded()
    assert handle.resident and handle is not first
