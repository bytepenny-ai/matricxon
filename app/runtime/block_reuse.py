"""Block reuse: a prompt changed in the middle (a note edited, a skill added or removed) keeps the
cached keys/values of the parts that didn't change, instead of re-reading everything after the
change. The client asks for it per request (`ChatRequest.cache_reuse`, "blocks"); by default only
an exact prefix is reused.

After the exact prefix, the new prompt is matched against every cached conversation in RAM: each
run of at least `min_run` equal tokens is copied into a new cache at its new position, its keys
re-rotated (`KVCache.write_ahead`), and only the tokens between runs are read. Approximate: a
copied run was read with other text before it, so it carries a trace of that text (the trade
llama.cpp's `--cache-reuse` makes). Exact parts stay exact: the prefix, and every token read.

Runs are copied while the pool is locked, into a cache no other reply can touch, so the sources
may change right after. Only for plain KV caches of architectures whose keys can be re-rotated
(`supports_context_shift`), and never the prompt's last token (its logits start the reply).
"""

from dataclasses import dataclass

import torch

from app.architectures.base import ModelArchitecture
from app.runtime.kv_cache import KVCache
from app.runtime.prompt_slot import CacheSlot


@dataclass(frozen=True)
class CopiedRun:
    source: CacheSlot
    src_start: int
    dst_start: int
    count: int


@dataclass(frozen=True)
class PrefillStep:
    """Prompt positions `[start, end)`: copied ones are in the cache already, the rest are read."""

    start: int
    end: int
    copied: bool


class TokenRunIndex:
    """Where each `KEY` tokens long sequence appears in the source conversations."""

    KEY = 8
    MAX_PLACES = 8  # per sequence: repeated boilerplate shouldn't make lookups slow

    def __init__(self, sources: list[CacheSlot]) -> None:
        self._places: dict[tuple[int, ...], list[tuple[CacheSlot, int]]] = {}
        for slot in sources:
            ids = slot.token_ids
            for j in range(len(ids) - self.KEY + 1):
                places = self._places.setdefault(tuple(ids[j : j + self.KEY]), [])
                if len(places) < self.MAX_PLACES:
                    places.append((slot, j))

    def longest(
        self, prompt_ids: list[int], at: int, end: int, prefer_tag: str
    ) -> tuple[CacheSlot, int, int] | None:
        """(slot, start, length) of the longest cached run equal to `prompt_ids[at:]`, stopping
        before `end`; the chat's own conversation wins a tie."""
        if at + self.KEY > end:
            return None
        best: tuple[CacheSlot, int, int] | None = None
        for slot, j in self._places.get(tuple(prompt_ids[at : at + self.KEY]), ()):
            ids = slot.token_ids
            length = self.KEY
            while (
                at + length < end
                and j + length < len(ids)
                and ids[j + length] == prompt_ids[at + length]
            ):
                length += 1
            own = bool(prefer_tag) and slot.tag == prefer_tag
            if best is None or length > best[2] or (length == best[2] and own):
                best = (slot, j, length)
        return best


class BlockPlanner:
    def __init__(self, min_run: int) -> None:
        self._min_run = max(min_run, TokenRunIndex.KEY)

    def runs(
        self, prompt_ids: list[int], start: int, sources: list[CacheSlot], tag: str
    ) -> list[CopiedRun]:
        """The runs to copy into prompt positions `[start, len - 1)`, in order, without overlaps."""
        end = len(prompt_ids) - 1
        if end - start < self._min_run or not sources:
            return []
        index = TokenRunIndex(sources)
        runs: list[CopiedRun] = []
        at = start
        while at < end:
            found = index.longest(prompt_ids, at, end, tag)
            if found is not None and found[2] >= self._min_run:
                slot, src, length = found
                runs.append(CopiedRun(slot, src, at, length))
                at += length
            else:
                at += 1
        return runs

    @staticmethod
    def steps(runs: list[CopiedRun], start: int, prompt_len: int) -> list[PrefillStep]:
        """The prompt from `start` as read and copied pieces, in order."""
        steps: list[PrefillStep] = []
        at = start
        for run in runs:
            if run.dst_start > at:
                steps.append(PrefillStep(at, run.dst_start, False))
            steps.append(PrefillStep(run.dst_start, run.dst_start + run.count, True))
            at = run.dst_start + run.count
        if at < prompt_len:
            steps.append(PrefillStep(at, prompt_len, False))
        return steps


class BlockAssembler:
    """Builds the new conversation's cache: the exact prefix, then the copied runs."""

    def __init__(self, architecture: ModelArchitecture, min_run: int) -> None:
        self._architecture = architecture
        self._planner = BlockPlanner(min_run)

    def usable(self, cache_like: object) -> bool:
        return self._architecture.supports_context_shift and isinstance(cache_like, KVCache)

    def assemble(
        self,
        slots: list[CacheSlot],
        best: CacheSlot | None,
        n: int,
        prompt_ids: list[int],
        key: tuple[int, torch.dtype],
        tag: str,
    ) -> tuple[CacheSlot, list[PrefillStep], int] | None:
        """(new slot holding the prefix with the runs written ahead, the prefill steps from `n`,
        tokens copied), or None when nothing worth copying was found."""
        sources = [s for s in slots if s.key[1] == key[1] and self.usable(s.cache)]
        runs = self._planner.runs(prompt_ids, n, sources, tag)
        if not runs:
            return None
        if best is not None:
            slot = best.fork(n, key[0])
        else:
            cache = self._architecture.build_cache(max_seq_len=key[0], dtype=key[1])
            if not self.usable(cache):
                return None
            slot = CacheSlot(cache, key, tag=tag)
        rotations: dict[int, list | None] = {}
        for run in runs:
            delta = run.src_start - run.dst_start
            if delta not in rotations:
                rotations[delta] = (
                    self._architecture.context_shift_rotations(delta) if delta else None
                )
            slot.cache.write_ahead(
                run.dst_start, run.source.cache, run.src_start, run.count, rotations[delta]
            )
        copied = sum(run.count for run in runs)
        return slot, BlockPlanner.steps(runs, n, len(prompt_ids)), copied
