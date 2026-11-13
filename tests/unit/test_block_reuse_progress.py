"""What a block-reuse reply reports (app/runtime/block_reuse.py): copied tokens count as taken
from the cache, and progress counts only the tokens actually read (a client's read-speed sample
must not include copies)."""

from pathlib import Path

import torch

from app.architectures.llama import LlamaArchitecture
from app.gguf.loader import GGUFModelLoader
from app.runtime.chat_engine import ChatEngine
from app.runtime.generation_request import GenerationRequest, SamplingConfig
from app.runtime.live_requests import LiveRequest
from app.runtime.prompt_cache import PromptCache
from tests.tiny_gguf_llama import EOS_TOKEN_ID, build_tiny_llama_gguf

START = [3 + (i * 5) % 90 for i in range(10)]
REST = [20 + (i * 11) % 80 for i in range(40)]
GREEDY = SamplingConfig(temperature=0.0, num_predict=1, num_ctx=128)


def test_copied_tokens_are_reported_as_cached_and_not_as_read(tmp_path: Path):
    model = LlamaArchitecture.from_gguf(
        GGUFModelLoader(build_tiny_llama_gguf(tmp_path / "m.gguf"), dtype=torch.float32)
    )
    try:
        pool = PromptCache(4)

        def run(ids: list[int]) -> LiveRequest:
            live = LiveRequest("r", "m")
            engine = ChatEngine(model, {EOS_TOKEN_ID}, prompt_cache=pool, live=live)
            engine.generate(
                GenerationRequest(torch.tensor([ids]), GREEDY, cache_tag="c", block_reuse_min=16)
            )
            return live

        run(START + [200, 201, 202] + REST + [7])
        edited = START + [210, 211] + REST + [7]
        live = run(edited)
        assert live.reused_tokens == pool.reused_tokens == len(START) + len(REST)
        assert (
            live.prefilled_tokens == len(edited) - live.reused_tokens
        )  # the edit and the last token
    finally:
        model.close()
