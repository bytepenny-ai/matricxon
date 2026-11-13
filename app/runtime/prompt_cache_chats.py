import functools
import logging
import threading
from collections.abc import Callable
from typing import Concatenate, ParamSpec, TypeVar

import torch

from app.architectures.base import ModelArchitecture
from app.runtime.block_reuse import BlockAssembler
from app.runtime.cache_miss_diagnosis import CacheMissDiagnosis
from app.runtime.cache_tier import ModelCacheTier
from app.runtime.prompt_slot import CacheSlot

logger = logging.getLogger("app.runtime.prompt_cache")

P = ParamSpec("P")
R = TypeVar("R")


def locked(
    method: Callable[Concatenate["PromptCacheChats", P], R],
) -> Callable[Concatenate["PromptCacheChats", P], R]:
    """Runs a PromptCache method under its lock: the worker thread uses the pool while HTTP threads
    ask about, or spill, a chat (cluster routing)."""

    @functools.wraps(method)
    def wrapper(self: "PromptCacheChats", *args: P.args, **kwargs: P.kwargs) -> R:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class PromptCacheChats:
    """What `PromptCache` (prompt_cache.py) does per chat rather than per slot: context shift of a
    trimmed chat, cluster hand-off, saving to the disk tier and explaining a miss. The pool sets the
    attributes below."""

    _slots: list[CacheSlot]
    _tier: ModelCacheTier | None
    _lock: threading.RLock
    _add: Callable[[CacheSlot], CacheSlot]
    CONTINUATION_TAIL_TOKENS: int
    reused_tokens: int
    prefill_steps: list

    def _shift_trimmed_chat(
        self,
        architecture: ModelArchitecture,
        prompt_ids: list[int],
        key: tuple[int, torch.dtype],
        tag: str,
    ) -> None:
        """If this chat's cached prompt is the new one with its oldest messages trimmed away, cuts
        those tokens out of the cache (re-rotating the rest) instead of re-reading the chat.
        Approximate: the kept tokens were read while the dropped ones were visible, so they carry a
        trace of them, the trade llama.cpp's context shifting makes.

        A chat can have several copies (a branch is kept when a message is edited or deleted):
        the one keeping the most tokens is shifted. Taking the first found once shifted an old
        branch and re-read a 1366-token reply the newest copy held."""
        best: tuple[int, CacheSlot, tuple[int, int, int]] | None = None
        for slot in self._slots:
            if slot.tag != tag or slot.key[1] != key[1] or slot.in_use:
                continue
            gap = slot.find_gap(prompt_ids)
            if gap is not None and (best is None or gap[0] + gap[2] > best[0]):
                best = (gap[0] + gap[2], slot, gap)
        if best is None:
            return
        _, slot, (keep, dropped, run) = best
        rotations = architecture.context_shift_rotations(dropped)
        if rotations is None:
            return
        slot.shift_out(keep, dropped, rotations)
        logger.info(
            "context shift: dropped %d trimmed tokens after the first %d, kept the next %d",
            dropped,
            keep,
            run,
        )

    def _assemble_blocks(
        self,
        architecture: ModelArchitecture,
        prompt_ids: list[int],
        key: tuple[int, torch.dtype],
        best: CacheSlot | None,
        n: int,
        tag: str,
        min_run: int,
    ) -> tuple[object, int] | None:
        """Block reuse (block_reuse.py): a new slot with the prefix and the runs copied from the
        pool, its steps in `prefill_steps`; None when nothing is worth copying."""
        if not architecture.supports_context_shift or len(prompt_ids) - 1 - n < min_run:
            return None
        if best is not None and best.hybrid:
            return None
        # A chat that only continues (nothing cached past the match but the end of its last turn)
        # has nothing to copy from its own conversation: no search, it costs a pass over the pool.
        continues = (
            best is not None
            and best.tag == tag
            and n >= len(best.token_ids) - self.CONTINUATION_TAIL_TOKENS
        )
        if continues:
            return None
        made = BlockAssembler(architecture, min_run).assemble(
            self._slots, best, n, prompt_ids, key, tag
        )
        if made is None:
            return None
        slot, steps, copied = made
        slot.prompt_len = len(prompt_ids)
        slot.in_use = True
        slot.tag = tag or slot.tag
        self._add(slot)
        self.reused_tokens = n + copied
        self.prefill_steps = steps
        logger.info(
            "block reuse: kept the first %d tokens, copied %d more in %d runs, reading %d of %d",
            n,
            copied,
            sum(1 for s in steps if s.copied),
            len(prompt_ids) - n - copied,
            len(prompt_ids),
        )
        return slot.cache, n

    def _explain_miss(self, prompt_ids: list[int], key: tuple[int, torch.dtype], tag: str) -> str:
        parts = [CacheMissDiagnosis.ram(self._slots, prompt_ids, key, tag)]
        if self._tier is not None:
            parts.append(self._tier.explain(prompt_ids, key[1], tag))
        return "; ".join(
            f"{label}: {text}" for label, text in zip(("RAM", "disk"), parts, strict=False)
        )

    @locked
    def holds(self, tag: str) -> bool:
        """Does this pool hold a conversation of the chat `tag` (idle or in use)?"""
        return bool(tag) and any(slot.tag == tag for slot in self._slots)

    @locked
    def spill_chat(self, tag: str) -> int:
        """Moves the chat's idle conversations from RAM to the disk tier (the model stays loaded) so
        another node can pick them up. Returns how many; one still replying stays."""
        if self._tier is None or not tag:
            return 0
        moving = [s for s in self._slots if s.tag == tag and not s.in_use]
        for slot in moving:
            self._slots.remove(slot)
            self._tier.spill(slot)
        return len(moving)

    @locked
    def persist_all(self) -> None:
        """Queues every idle slot for the disk tier: the model is unloading or the server shutting
        down, so nothing else will use these RAM copies."""
        if self._tier is not None:
            for slot in self._slots:
                if not slot.in_use:
                    self._tier.spill(slot)
