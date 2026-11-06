"""Every removal of a saved prompt cache is logged with its reason (cache_limits.py)."""

import logging
import time
from pathlib import Path

import pytest

from tests.unit.test_cache_store import DTYPE, MODEL, _slot_with_tag, _spill, _store, small_blocks

__all__ = ["small_blocks"]  # the autouse fixture shrinking the blocks, used here too

LOGGER = "app.runtime.cache_store"


def _removals(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "prompt cache:" in r.getMessage()]


def test_ttl_and_budget_removals_say_why(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    store = _store(tmp_path)
    _spill(store, list(range(24)))
    (name,) = store._index
    store._index[name].saved_at = time.time() - 25 * 3600
    with caplog.at_level(logging.INFO, logger=LOGGER):
        store.enforce_limits()
    assert any("older than 24 h" in m and "24 tokens" in m for m in _removals(caplog))

    caplog.clear()
    store = _store(tmp_path / "b")
    _spill(store, list(range(24)))
    store.policy.budget_mb = 0  # lowered after the save, as an admin would
    with caplog.at_level(logging.INFO, logger=LOGGER):
        store.enforce_limits()
    assert store.usage()[0] == 0
    assert any("over the 0 MB limit" in m for m in _removals(caplog))


def test_a_deleted_chat_and_a_clear_say_why(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store(tmp_path)
    store.spill_async(MODEL, _slot_with_tag(list(range(100, 124)), "chat-a-long-id"))
    store.spill_async(MODEL, _slot_with_tag(list(range(200, 224)), "chat-b"))
    store.flush()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        store.forget("chat-a-long-id")
        store.clear()
    messages = _removals(caplog)
    assert any("chat-a-l (24 tokens" in m and "its chat was deleted" in m for m in messages)
    assert any("chat-b" in m and "cleared or turned off" in m for m in messages)


def test_a_newer_copy_replacing_an_older_one_says_why(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store(tmp_path)
    store.spill_async(MODEL, _slot_with_tag(list(range(100, 124)), "chat-a"))
    store.flush()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        store.spill_async(MODEL, _slot_with_tag(list(range(100, 132)), "chat-a"))
        store.flush()
    assert any("replaced by a newer copy" in m for m in _removals(caplog))


def test_a_restore_into_ram_is_not_logged_as_a_removal(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store(tmp_path)
    _spill(store, list(range(24)))
    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert store.take(MODEL, list(range(30)), DTYPE, 0) is not None
    assert not any("removed" in m for m in _removals(caplog))


def test_a_file_deleted_outside_matricxon_is_noticed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = _store(tmp_path)
    store.spill_async(MODEL, _slot_with_tag(list(range(100, 124)), "chat-x"))
    store.flush()
    (file,) = (tmp_path / "c").glob("*.mxc")
    file.unlink()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        store._refresh()
    assert any("chat-x" in m and "disappeared from the folder" in m for m in _removals(caplog))
