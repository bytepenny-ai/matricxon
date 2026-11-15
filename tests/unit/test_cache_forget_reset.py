"""A chat whose saved prompt caches were deleted (forget: a deleted chat, a channel reset)
refuses copies of what it held before, but a reply after the delete is saved again. Before, a
reset channel was never saved again until Matricxon restarted, which then re-read it all."""

import time
from pathlib import Path

from app.runtime.slot_codec import SlotCodec
from tests.unit.test_cache_store import MODEL, _slot_with_tag, _store, small_blocks  # noqa: F401

IDS = list(range(100, 124))


def test_a_reply_after_the_delete_is_saved_again(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.forget("chan")
    old = _slot_with_tag(IDS, "chan")  # held from before the reset (changed_at: never)
    store.spill_async(MODEL, old)
    store.flush()
    assert store.usage()[0] == 0

    new = _slot_with_tag(IDS, "chan")
    new.changed_at = time.monotonic()  # a reply finished after the reset
    store.spill_async(MODEL, new)
    store.flush()
    assert store.usage()[0] == 1


def test_a_copy_queued_before_the_delete_is_still_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    slot = _slot_with_tag(IDS, "chan")
    slot.changed_at = time.monotonic()
    state = SlotCodec.capture(slot)
    store.forget("chan")  # the reset lands after this reply, before its copy is written
    store.checkpoint_async(MODEL, state, "chan", slot)
    store.flush()
    assert store.usage()[0] == 0
