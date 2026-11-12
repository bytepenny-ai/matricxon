"""Removing saved prompt caches: the TTL and size limits, a deleted chat and the admin's clear,
each removal logged with its reason (part of `PersistentCacheStore`)."""

import logging
import time
from pathlib import Path

from app.runtime.cache_chain import ChainIndex
from app.runtime.cache_files import CacheFilesBase

logger = logging.getLogger("app.runtime.cache_store")


class CacheLimits(CacheFilesBase):
    def _remove(self, name: str, path: Path | None = None, reason: str | None = None) -> None:
        with self._lock:
            entry = self._index.pop(name, None)
        (path or self._dir / name).unlink(missing_ok=True)
        if reason:  # None: a normal restore into RAM
            logger.info(
                "prompt cache: removed saved chat %s - %s",
                entry.describe() if entry else name,
                reason,
            )

    def enforce_limits(self) -> None:
        """Drops saved chats past the TTL, then the oldest until the total is within budget, and
        chains missing a piece for a while (only on the node owning the cleanup job). A chain
        (cache_chain.py) goes whole: aged by its newest piece, never cut in two."""
        if not self.may_clean():
            return
        with self._lock:
            groups = ChainIndex.groups(list(self._index.items()))
        now = time.time()
        doomed: list[tuple[list[str], str]] = []
        keep = []
        for group in groups:
            if now - group.saved_at > self.policy.ttl_seconds:
                doomed.append((group.names_newest_first, f"older than {self.policy.ttl_hours} h"))
            elif group.broken_for(now):
                doomed.append((group.names_newest_first, "a piece of its chain is missing"))
            else:
                keep.append(group)
        keep.sort(key=lambda g: g.saved_at)
        total = sum(g.size for g in keep)
        while keep and total > self.policy.budget_bytes:
            group = keep.pop(0)
            total -= group.size
            reason = f"over the {self.policy.budget_mb} MB limit (oldest first)"
            doomed.append((group.names_newest_first, reason))
        for names, reason in doomed:
            for index, name in enumerate(names):
                self._remove(name, reason=reason if index == 0 else None)

    def forget(self, tag: str) -> int:
        """Deletes every stored file of the chat `tag` (and refuses its later spills); returns how
        many files went."""
        if not tag:
            return 0
        self._refresh()
        with self._lock:
            self._forgotten.add(tag)
            names = [n for n, e in self._index.items() if e.tag == tag]
        for name in names:
            self._remove(name, reason="its chat was deleted")
        return len(names)

    def clear(self) -> None:
        self._refresh()
        with self._lock:
            names = list(self._index)
        for name in names:
            self._remove(name, reason="saved caches were cleared or turned off")
