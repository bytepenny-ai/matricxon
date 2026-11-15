"""Thinking off means off: the reasoning-opening tokens are never picked when a request sets
think=False (chat_stop_tokens.thinking_start_token_ids, Sampler), and only then."""

from types import SimpleNamespace

import torch

from app.gguf.metadata import GGUFMetadata
from app.runtime.chat_generation import ChatGeneration
from app.runtime.chat_stop_tokens import thinking_start_token_ids
from app.runtime.generation_request import SamplingConfig
from app.runtime.sampler import Sampler


def test_finds_gemma_and_qwen_thinking_tokens_only_when_the_vocabulary_has_them() -> None:
    gemma = GGUFMetadata({"tokenizer.ggml.tokens": ["<pad>", "<|channel>", "<channel|>", "hi"]})
    qwen = GGUFMetadata({"tokenizer.ggml.tokens": ["a", "<think>", "</think>"]})
    plain = GGUFMetadata({"tokenizer.ggml.tokens": ["a", "b"]})
    assert thinking_start_token_ids(gemma) == frozenset({1})
    assert thinking_start_token_ids(qwen) == frozenset({1})
    assert thinking_start_token_ids(plain) == frozenset()
    assert thinking_start_token_ids(GGUFMetadata({})) == frozenset()


def test_a_banned_token_is_never_picked_greedy_or_sampled() -> None:
    logits = torch.tensor([0.0, 9.0, 1.0, 0.5])
    greedy = Sampler(SamplingConfig(temperature=0.0, banned_token_ids=frozenset({1})))
    assert greedy.sample(logits, []) == 2
    sampled = Sampler(SamplingConfig(temperature=1.0, top_k=0, banned_token_ids=frozenset({1})))
    assert all(sampled.sample(logits, []) != 1 for _ in range(50))
    assert logits[1] == 9.0  # the model's own logits untouched


def _request(think: bool | None) -> SimpleNamespace:
    options = SimpleNamespace(
        temperature=0.1, top_p=0.9, top_k=40, repeat_penalty=1.1, num_ctx=2048, num_predict=-1,
        seed=None, min_p=0.0, presence_penalty=0.0, frequency_penalty=0.0,
    )  # fmt: skip
    reuse = SimpleNamespace(mode="exact", min_tokens=0)
    return SimpleNamespace(
        options=options, format=None, think=think, cache_tag="", cache_reuse=reuse
    )


def test_only_an_explicit_think_false_blocks_the_tokens() -> None:
    handle = SimpleNamespace(thinking_start_token_ids=frozenset({100}))
    banned = {
        think: ChatGeneration()
        .generation_request(_request(think), handle, [1, 2, 3])
        .sampling.banned_token_ids
        for think in (False, True, None)
    }
    assert banned == {False: frozenset({100}), True: frozenset(), None: frozenset()}
