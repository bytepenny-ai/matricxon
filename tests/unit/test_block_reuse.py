"""Block reuse (app/runtime/block_reuse.py): a prompt changed in the middle keeps the cached parts
that didn't change, copied to their new positions, instead of re-reading everything after the
change."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from app.architectures.gemma4 import Gemma4Architecture
from app.architectures.llama import LlamaArchitecture
from app.architectures.rope import apply_rotary_pos_emb
from app.gguf.loader import GGUFModelLoader
from app.runtime.block_reuse import BlockPlanner, PrefillStep, TokenRunIndex
from app.runtime.chat_engine import ChatEngine
from app.runtime.chat_generation import ChatGeneration
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.kv_cache import KVCache
from app.runtime.prompt_cache import PromptCache
from app.runtime.prompt_slot import CacheSlot
from app.schemas.chat import CacheReuse, ChatRequest
from tests.tiny_gguf_gemma4 import build_tiny_gemma4_gguf
from tests.tiny_gguf_llama import EOS_TOKEN_ID, build_tiny_llama_gguf

SYSTEM = [3 + (i * 5) % 90 for i in range(10)]
NOTE_A = [100 + (i * 7) % 50 for i in range(12)]
NOTE_B = [150 + (i * 3) % 40 for i in range(18)]  # a longer, different note
SKILLS = [20 + (i * 11) % 80 for i in range(30)]
CHAT = [60 + (i * 13) % 90 for i in range(26)]
QUESTION = [200, 201, 202, 203]
MIN = 16
GREEDY = SamplingConfig(temperature=0.0, num_predict=2, num_ctx=200)
CTX = (200, torch.float32)


@pytest.fixture(params=["llama", "gemma4"])
def model(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[object]:
    if request.param == "llama":
        arch = LlamaArchitecture.from_gguf(
            GGUFModelLoader(build_tiny_llama_gguf(tmp_path / "m.gguf"), dtype=torch.float32)
        )
    else:
        arch = Gemma4Architecture.from_gguf(
            GGUFModelLoader(build_tiny_gemma4_gguf(tmp_path / "m.gguf"), dtype=torch.float32),
            dtype=torch.float32,
        )
    yield arch
    arch.close()


def _run(
    model: object, pool: PromptCache, ids: list[int], tag: str = "chat", block_min: int = MIN
) -> list[int]:
    engine = ChatEngine(model, {EOS_TOKEN_ID}, prompt_cache=pool, prefill_chunk=8)
    request = GenerationRequest(
        torch.tensor([ids]), GREEDY, cache_tag=tag, block_reuse_min=block_min
    )
    return engine.generate(request).token_ids


def _slot(ids: list[int], tag: str = "chat") -> CacheSlot:
    return CacheSlot(KVCache([(1, 4)], 200), CTX, list(ids), tag=tag)


def _newest(pool: PromptCache, tag: str) -> CacheSlot:
    return max((s for s in pool._slots if s.tag == tag), key=lambda s: s.last_used)


# Planning


def test_an_edited_note_keeps_the_parts_after_it():
    old = _slot(SYSTEM + NOTE_A + SKILLS + CHAT)
    new = SYSTEM + NOTE_B + SKILLS + CHAT + QUESTION
    runs = BlockPlanner(MIN).runs(new, len(SYSTEM), [old], "chat")
    assert [(r.src_start, r.dst_start, r.count) for r in runs] == [
        (len(SYSTEM + NOTE_A), len(SYSTEM + NOTE_B), len(SKILLS + CHAT))
    ]


def test_a_removed_and_an_added_block():
    old = _slot(SYSTEM + NOTE_A + SKILLS + CHAT)
    removed = SYSTEM + SKILLS + CHAT + QUESTION
    runs = BlockPlanner(MIN).runs(removed, len(SYSTEM), [old], "chat")
    assert runs and runs[0].src_start == len(SYSTEM + NOTE_A) and runs[0].dst_start == len(SYSTEM)
    added = SYSTEM + NOTE_A + NOTE_B + SKILLS + CHAT + QUESTION
    start = len(SYSTEM + NOTE_A)
    runs = BlockPlanner(MIN).runs(added, start, [old], "chat")
    assert runs[0].dst_start == start + len(NOTE_B) and runs[0].count == len(SKILLS + CHAT)


def test_the_last_token_is_never_copied_and_short_runs_are_read():
    old = _slot(SYSTEM + SKILLS)
    runs = BlockPlanner(MIN).runs(SKILLS, 0, [old], "chat")
    assert runs[0].count == len(SKILLS) - 1
    assert BlockPlanner(MIN).runs(SKILLS[:10] + [250] * 20, 0, [old], "chat") == []


def test_steps_alternate_read_and_copied_parts_and_end_with_a_read():
    old = _slot(SYSTEM + NOTE_A + SKILLS + CHAT)
    new = SYSTEM + NOTE_B + SKILLS + CHAT + QUESTION
    runs = BlockPlanner(MIN).runs(new, len(SYSTEM), [old], "chat")
    steps = BlockPlanner.steps(runs, len(SYSTEM), len(new))
    copy_end = len(SYSTEM + NOTE_B + SKILLS + CHAT)
    assert steps == [
        PrefillStep(len(SYSTEM), len(SYSTEM + NOTE_B), False),
        PrefillStep(len(SYSTEM + NOTE_B), copy_end, True),
        PrefillStep(copy_end, len(new), False),
    ]


def test_the_chats_own_conversation_wins_a_tie():
    other, own = _slot(SKILLS, "other"), _slot(SKILLS, "chat")
    found = TokenRunIndex([other, own]).longest(SKILLS + [9], 0, len(SKILLS), "chat")
    assert found is not None and found[0] is own


# Re-rotation


def test_copied_keys_equal_keys_rotated_fresh_at_their_new_place(model):
    """Every layer type (Gemma 4's local and global ropes differ); both directions."""
    shapes = model.kv_cache_layer_shapes
    n = 30
    source = KVCache(shapes, 64)
    raw = [torch.randn(1, heads, n, dim) for heads, dim in shapes]
    for i in range(len(shapes)):
        cos, sin = model._rope_for_layer(i)(torch.arange(n))
        source._k[i][:, :, :n] = apply_rotary_pos_emb(raw[i], raw[i], cos, sin)[0]
        source._v[i][:, :, :n] = raw[i]
    source._length = n
    for src, dst in ((10, 3), (4, 20)):
        target = KVCache(shapes, 64)
        target.write_ahead(dst, source, src, 8, model.context_shift_rotations(src - dst))
        assert target.length == 0  # committed only by `advance`
        for i in range(len(shapes)):
            cos, sin = model._rope_for_layer(i)(torch.arange(dst, dst + 8))
            part = raw[i][:, :, src : src + 8]
            fresh = apply_rotary_pos_emb(part, part, cos, sin)[0]
            assert torch.allclose(target._k[i][:, :, dst : dst + 8], fresh, atol=1e-4)
            assert torch.equal(target._v[i][:, :, dst : dst + 8], part)


def test_bad_copies_are_refused():
    source = KVCache([(1, 4)], 16)
    source.advance(10)
    target = KVCache([(1, 4)], 16)
    target.advance(4)
    with pytest.raises(ValueError):
        target.write_ahead(2, source, 0, 2, None)  # over committed positions
    with pytest.raises(ValueError):
        target.write_ahead(4, source, 5, 6, None)  # past the source's length
    with pytest.raises(ValueError):
        target.write_ahead(12, source, 0, 6, None)  # past the capacity
    with pytest.raises(ValueError):
        KVCache([(2, 4)], 16).write_ahead(0, source, 0, 2, None)  # another model's shape


# A real (tiny) model


def test_an_edited_note_reads_only_the_note_and_the_question(model, caplog):
    pool = PromptCache(4)
    _run(model, pool, SYSTEM + NOTE_A + SKILLS + CHAT + QUESTION)
    edited = SYSTEM + NOTE_B + SKILLS + CHAT + QUESTION
    with caplog.at_level(logging.INFO):
        reply = _run(model, pool, edited)
    assert "block reuse: kept the first" in caplog.text
    assert pool.reused_tokens >= len(SYSTEM) + len(SKILLS + CHAT)
    assert len(reply) > 0
    slot = _newest(pool, "chat")
    assert slot.token_ids[: len(edited)] == edited  # the cache's tokens are the prompt's
    cold = PromptCache(4)
    _run(model, cold, edited, block_min=0)
    n = len(edited)
    # layer 0's keys and values depend only on the token and its position: exact after the copy
    assert torch.allclose(
        slot.cache._k[0][:, :, :n], _newest(cold, "chat").cache._k[0][:, :, :n], atol=1e-4
    )
    assert torch.allclose(
        slot.cache._v[0][:, :, :n], _newest(cold, "chat").cache._v[0][:, :, :n], atol=1e-4
    )


def test_the_next_turn_continues_the_assembled_cache_exactly(model):
    pool = PromptCache(4)
    _run(model, pool, SYSTEM + NOTE_A + SKILLS + CHAT + QUESTION)
    edited = SYSTEM + NOTE_B + SKILLS + CHAT + QUESTION
    reply = _run(model, pool, edited)
    _run(model, pool, edited + reply + [210, 211])
    assert pool.reused_tokens >= len(edited)
    assert pool.prefill_steps == []  # a plain continuation: no search


def test_a_new_chat_copies_blocks_another_chat_holds(model):
    pool = PromptCache(4)
    _run(model, pool, SYSTEM + NOTE_A + SKILLS + QUESTION, tag="first")
    _run(model, pool, SYSTEM + NOTE_B + SKILLS + [205, 206], tag="second")
    assert pool.reused_tokens >= len(SYSTEM) + len(SKILLS)


def test_the_sources_are_left_as_they_were(model):
    pool = PromptCache(4)
    first = SYSTEM + NOTE_A + SKILLS + CHAT + QUESTION
    _run(model, pool, first)
    source = _newest(pool, "chat")
    tokens, keys = list(source.token_ids), source.cache._k[-1][:, :, : len(first)].clone()
    _run(model, pool, SYSTEM + NOTE_B + SKILLS + CHAT + QUESTION, tag="other")
    assert source.token_ids == tokens
    assert torch.equal(source.cache._k[-1][:, :, : len(first)], keys)


def test_off_by_default_and_for_architectures_that_cant_rotate(model, monkeypatch, caplog):
    edited = SYSTEM + NOTE_B + SKILLS + CHAT + QUESTION
    pool = PromptCache(4)
    _run(model, pool, SYSTEM + NOTE_A + SKILLS + CHAT + QUESTION)
    with caplog.at_level(logging.INFO):
        _run(model, pool, edited, block_min=0)
    assert "block reuse" not in caplog.text and pool.reused_tokens == len(SYSTEM)
    monkeypatch.setattr(type(model), "supports_context_shift", False)
    pool = PromptCache(4, context_shift=False)
    _run(model, pool, SYSTEM + NOTE_A + SKILLS + CHAT + QUESTION)
    _run(model, pool, edited)
    assert pool.reused_tokens == len(SYSTEM) and pool.prefill_steps == []


def test_a_prompt_that_cant_be_pooled_is_never_assembled(model):
    pool = PromptCache(4)
    _run(model, pool, SYSTEM + NOTE_A + SKILLS + CHAT + QUESTION)
    edited = SYSTEM + NOTE_B + SKILLS + CHAT + QUESTION
    _, n = pool.acquire(
        model, edited, 200, torch.float32, reusable=False, tag="chat", block_min=MIN
    )
    assert n == 0 and pool.prefill_steps == []


# The request


def test_the_request_asks_for_blocks_explicitly():
    plain = ChatRequest(model="m", messages=[])
    assert plain.cache_reuse.mode == "exact"
    asked = ChatRequest(model="m", messages=[], cache_reuse={"mode": "blocks", "min_tokens": 64})
    made = ChatGeneration().generation_request(asked, object(), [1, 2, 3])
    assert made.block_reuse_min == 64
    assert ChatGeneration().generation_request(plain, object(), [1, 2, 3]).block_reuse_min == 0
    with pytest.raises(ValueError):
        CacheReuse(min_tokens=2)
