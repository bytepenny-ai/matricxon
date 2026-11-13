"""Saving a chat's cache as a chain (app.runtime.cache_chain): one full copy, then pieces holding
only the new tokens. Restores must be exact; anything uncertain must fall back to a full copy."""

import json
import struct
import time
from pathlib import Path

import pytest
import torch

from app.architectures.llama import LlamaArchitecture
from app.gguf.loader import GGUFModelLoader
from app.runtime import cache_chain, cache_store, slot_codec
from app.runtime.cache_files import MAGIC
from app.runtime.cache_tier import ModelCacheTier
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.prompt_cache import PromptCache
from app.runtime.prompt_slot import CacheSlot
from app.runtime.slot_codec import SlotCodec
from tests.tiny_gguf_llama import EOS_TOKEN_ID, build_tiny_llama_gguf
from tests.unit.test_cache_store import CTX, DTYPE, MODEL, StubArchitecture, _slot, _store
from tests.unit.test_prompt_cache_checkpoint import _store as checkpoint_store

TAG = "chat-1"


@pytest.fixture(autouse=True)
def small_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_store, "BLOCK_TOKENS", 8)
    monkeypatch.setattr(slot_codec, "MIN_PERSIST_TOKENS", 8)


def _grow(slot: CacheSlot, count: int) -> None:
    """The chat goes on: `count` new tokens with their own keys and values."""
    start = len(slot.token_ids)
    for layer in range(3):
        slot.cache.update(layer, torch.randn(1, 2, count, 4), torch.randn(1, 2, count, 4))
    slot.cache.advance(count)
    slot.token_ids.extend(range(5000 + start, 5000 + start + count))
    slot.prompt_len = len(slot.token_ids)


def _save(store: cache_store.PersistentCacheStore, slot: CacheSlot) -> None:
    store.checkpoint_async(MODEL, SlotCodec.capture(slot), TAG, slot)
    store.flush()


def _files(tmp_path: Path) -> list[Path]:
    """The cache files (to count; pick one by its place via `store._index`, not by order)."""
    return sorted((tmp_path / "c").glob("*.mxc"), key=lambda p: p.stat().st_mtime_ns)


def _restore(store: cache_store.PersistentCacheStore, slot_ids: list[int]) -> CacheSlot | None:
    taken = store.take(MODEL, slot_ids + [1], DTYPE, min_tokens=0, tag=TAG)
    return None if taken is None else SlotCodec.decode(taken[0], StubArchitecture(), CTX, DTYPE)


def _same_cache(a: CacheSlot, b: CacheSlot) -> bool:
    length = len(a.token_ids)
    ka, va = a.cache.export(length)
    kb, vb = b.cache.export(length)
    return all(torch.equal(x, y) for x, y in zip([*ka, *va], [*kb, *vb], strict=True))


def _chat(store: cache_store.PersistentCacheStore, tokens: int = 64) -> CacheSlot:
    slot = _slot(list(range(tokens)))
    slot.tag = TAG
    _save(store, slot)
    return slot


def test_a_growing_chat_adds_small_pieces_and_restores_exactly(tmp_path: Path) -> None:
    store = _store(tmp_path)
    slot = _chat(store)
    (base,) = _files(tmp_path)
    for _ in range(3):
        _grow(slot, 4)
        _save(store, slot)
    files = _files(tmp_path)
    assert len(files) == 4 and store.chat_count() == 1
    # only the new tokens; the base by name (files saved in one clock tick share an mtime)
    assert all(f.stat().st_size < base.stat().st_size / 3 for f in files if f != base)

    restored = _restore(store, slot.token_ids)
    assert restored is not None and restored.token_ids == slot.token_ids
    assert restored.prompt_len == slot.prompt_len and _same_cache(restored, slot)
    assert _files(tmp_path) == []  # consumed, as a single file was


def test_nothing_new_writes_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    slot = _chat(store)
    _save(store, slot)
    assert len(_files(tmp_path)) == 1


def test_a_long_chain_is_replaced_by_a_full_copy(tmp_path: Path) -> None:
    store = _store(tmp_path)
    slot = _chat(store, tokens=100)
    for _ in range(cache_chain.MAX_PIECES):
        _grow(slot, 2)
        _save(store, slot)
    assert len(_files(tmp_path)) == 1 + cache_chain.MAX_PIECES
    _grow(slot, 2)
    _save(store, slot)
    assert len(_files(tmp_path)) == 1  # a new full copy; the old chain was removed
    restored = _restore(store, slot.token_ids)
    assert restored is not None and _same_cache(restored, slot)


def test_pieces_past_half_the_base_make_a_full_copy(tmp_path: Path) -> None:
    store = _store(tmp_path)
    slot = _chat(store, tokens=40)
    _grow(slot, 30)  # more than half of the 40-token base
    _save(store, slot)
    assert len(_files(tmp_path)) == 1


def test_a_rewound_or_branched_chat_is_saved_whole(tmp_path: Path) -> None:
    store = _store(tmp_path)
    slot = _chat(store)
    slot.rewind(40)  # an edited message: the saved tokens are no longer this chat's start
    _grow(slot, 8)
    _save(store, slot)
    files = _files(tmp_path)
    assert len(files) == 1
    restored = _restore(store, slot.token_ids)
    assert restored is not None and restored.token_ids == slot.token_ids
    assert _same_cache(restored, slot)


def test_a_base_removed_outside_makes_the_next_save_whole(tmp_path: Path) -> None:
    store = _store(tmp_path)
    slot = _chat(store)
    _files(tmp_path)[0].unlink()  # e.g. another node took it, or someone cleaned the folder
    _grow(slot, 4)
    _save(store, slot)
    store._refresh()
    assert len(_files(tmp_path)) == 1
    restored = _restore(store, slot.token_ids)
    assert restored is not None and _same_cache(restored, slot)


def test_a_piece_without_its_base_is_never_restored_and_is_cleaned_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    slot = _chat(store)
    _grow(slot, 4)
    _save(store, slot)
    by_seq = {e.seq: tmp_path / "c" / n for n, e in store._index.items()}
    base, piece = by_seq[0], by_seq[1]
    base.unlink()
    store._refresh()
    assert _restore(store, slot.token_ids) is None and piece.exists()  # within the grace period
    later = time.time() + cache_chain.BROKEN_GRACE_SECONDS + 1
    monkeypatch.setattr(cache_chain.time, "time", lambda: later)
    monkeypatch.setattr(cache_store.time, "time", lambda: later)
    store.enforce_limits()
    assert not piece.exists()


def test_a_gap_restores_up_to_the_last_whole_piece(tmp_path: Path) -> None:
    store = _store(tmp_path)
    slot = _chat(store)
    states = []
    for _ in range(3):
        _grow(slot, 4)
        _save(store, slot)
        states.append(list(slot.token_ids))
    second = next(n for n, e in store._index.items() if e.seq == 2)
    (tmp_path / "c" / second).unlink()  # the second piece is gone
    store._refresh()
    restored = _restore(store, slot.token_ids)
    assert restored is not None and restored.token_ids == states[0]  # base + first piece


def test_a_damaged_piece_is_a_miss_and_its_chain_is_removed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    slot = _chat(store)
    _grow(slot, 4)
    _save(store, slot)
    piece = next(tmp_path / "c" / n for n, e in store._index.items() if e.seq == 1)
    raw = bytearray(piece.read_bytes())
    raw[-5] ^= 0xFF
    piece.write_bytes(bytes(raw))
    assert _restore(store, slot.token_ids) is None
    assert _files(tmp_path) == []


def test_limits_take_a_chain_whole_and_age_it_by_its_newest_piece(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path, ttl_hours=1)
    slot = _chat(store)
    base = _files(tmp_path)[0]
    later = time.time() + 3000  # the base is 50 minutes old
    monkeypatch.setattr(cache_store.time, "time", lambda: later)
    monkeypatch.setattr("app.runtime.cache_writer.time.time", lambda: later)
    _grow(slot, 4)
    _save(store, slot)
    monkeypatch.setattr(cache_store.time, "time", lambda: later + 1200)  # base 70 min, piece 20
    store.enforce_limits()
    assert base.exists() and len(_files(tmp_path)) == 2
    monkeypatch.setattr(cache_store.time, "time", lambda: later + 4000)
    store.enforce_limits()
    assert _files(tmp_path) == []  # the whole chain, never half of it


def test_over_the_budget_a_chain_goes_whole(tmp_path: Path) -> None:
    store = _store(tmp_path)
    slot = _chat(store)
    _grow(slot, 4)
    _save(store, slot)
    store.policy.budget_mb = 0  # nothing fits
    store.enforce_limits()
    assert _files(tmp_path) == []


def test_a_chat_leaving_ram_replaces_its_queued_copy(tmp_path: Path) -> None:
    store = _store(tmp_path)
    slot = _chat(store)
    _grow(slot, 4)
    store._writer.submit(time.sleep, 0.2)  # the writer is busy: the next copy queues
    store.checkpoint_async(MODEL, SlotCodec.capture(slot), TAG, slot)
    _grow(slot, 4)
    store.spill_async(MODEL, slot)
    store.flush()
    assert len(_files(tmp_path)) == 2  # one piece, never two for the same place
    restored = _restore(store, slot.token_ids)
    assert restored is not None and _same_cache(restored, slot)


def test_another_node_takes_the_whole_chain_and_this_one_starts_over(tmp_path: Path) -> None:
    mine, theirs = _store(tmp_path, node_id="a"), _store(tmp_path, node_id="b")
    slot = _chat(mine)
    _grow(slot, 4)
    _save(mine, slot)
    theirs._refresh()
    taken = theirs.take(MODEL, slot.token_ids + [1], DTYPE, min_tokens=0, tag=TAG)
    assert taken is not None
    assert SlotCodec.decode(taken[0], StubArchitecture(), CTX, DTYPE).token_ids == slot.token_ids
    _grow(slot, 4)
    _save(mine, slot)  # its chain is gone: a full copy, not a piece after nothing
    restored = _restore(mine, slot.token_ids)
    assert restored is not None and _same_cache(restored, slot)


def test_a_deleted_chat_loses_every_piece_and_saves_no_more(tmp_path: Path) -> None:
    store = _store(tmp_path)
    slot = _chat(store)
    _grow(slot, 4)
    _save(store, slot)
    assert store.forget(TAG) == 2 and _files(tmp_path) == []
    _grow(slot, 4)
    _save(store, slot)
    assert _files(tmp_path) == []


def test_a_failed_save_makes_the_next_one_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    slot = _chat(store)
    real_write = store._write
    calls = {"n": 0}

    def flaky(*args: object, **kwargs: object) -> bool:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(store, "_write", flaky)
    _grow(slot, 4)
    _save(store, slot)  # fails
    assert slot.persisted is None
    _grow(slot, 4)
    _save(store, slot)
    assert len(_files(tmp_path)) == 1  # a full copy, not a piece after a gap
    restored = _restore(store, slot.token_ids)
    assert restored is not None and _same_cache(restored, slot)


def test_a_file_saved_before_chains_still_restores(tmp_path: Path) -> None:
    store = _store(tmp_path)
    slot = _slot(list(range(64)))
    encoded = SlotCodec.encode(slot)
    meta = {
        "model_id": MODEL,
        "num_ctx": encoded.num_ctx,
        "dtype_name": encoded.dtype_name,
        "length": 64,
        "blocks": store._cipher.prefix_fingerprints(encoded.token_ids, 8),
        "saved_at": time.time(),
        "tag": TAG,
    }
    sealed_meta = store._cipher.seal(json.dumps(meta).encode(), MAGIC)
    body = store._cipher.seal(encoded.payload, MAGIC + sealed_meta)
    (tmp_path / "c" / "old.mxc").write_bytes(
        MAGIC + struct.pack(">I", len(sealed_meta)) + sealed_meta + body
    )
    store._refresh()
    restored = _restore(store, slot.token_ids)
    assert restored is not None and _same_cache(restored, slot)


def test_join_refuses_parts_that_dont_continue(tmp_path: Path) -> None:
    slot = _slot(list(range(64)))
    state = SlotCodec.capture(slot)
    base = SlotCodec.serialize(state).payload
    _grow(slot, 4)
    piece = SlotCodec.serialize_piece(SlotCodec.capture(slot), 64).payload
    assert len(SlotCodec.join([base, piece])["token_ids"]) == 68
    with pytest.raises(ValueError):
        SlotCodec.join([piece])  # no base
    with pytest.raises(ValueError):
        SlotCodec.join([base, piece, piece])  # a piece twice


def test_a_hybrid_cache_is_always_saved_whole() -> None:
    saved = cache_chain.PersistedChain("c", 0, 10, hash(tuple(range(10))), CTX, str(DTYPE), 10, 0)
    plan = cache_chain.SavePlan.decide(saved, list(range(12)), CTX, str(DTYPE), True, True)
    assert plan is not None and plan.full
    piece = cache_chain.SavePlan.decide(saved, list(range(12)), CTX, str(DTYPE), False, True)
    assert piece is not None and not piece.full and piece.start == 10 and piece.seq == 1


def test_a_real_chat_saved_in_pieces_answers_the_same_after_a_restart(tmp_path: Path) -> None:
    """End to end, on a real (tiny) model: replies saved as pieces, then a fresh process restores
    the chain and its next reply matches one computed without any cache."""
    model = LlamaArchitecture.from_gguf(
        GGUFModelLoader(build_tiny_llama_gguf(tmp_path / "l.gguf"), dtype=torch.float32)
    )
    greedy = SamplingConfig(temperature=0.0, num_predict=4, num_ctx=96)

    def reply(pool: PromptCache | None, ids: list[int]) -> list[int]:
        engine = ChatEngine(model, {EOS_TOKEN_ID}, prompt_cache=pool, prefill_chunk=8)
        return engine.generate(
            GenerationRequest(torch.tensor([ids]), greedy, cache_tag=TAG)
        ).token_ids

    try:
        store = checkpoint_store(tmp_path, save_every=1)
        pool = PromptCache(4, tier=ModelCacheTier(store, "m|1"))
        ids = [3 + (i * 7) % 200 for i in range(40)]
        for turn in range(3):
            ids = ids + reply(pool, ids) + [11 + turn, 12 + turn]
            time.sleep(1.1)  # past save_every_seconds: each reply is saved
        store.flush()
        assert store.usage()[0] >= 2  # a base and at least one piece

        fresh = PromptCache(4, tier=ModelCacheTier(checkpoint_store(tmp_path), "m|1"))
        restored = reply(fresh, ids)
        assert fresh.reused_tokens >= len(ids) - 8
        assert restored == reply(None, ids)  # the same answer as reading the whole chat again
    finally:
        model.close()
