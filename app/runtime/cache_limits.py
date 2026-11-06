"""Removing saved prompt caches: the TTL and size limits, a deleted chat and the admin's clear,
each removal logged with its reason (part of `PersistentCacheStore`)."""

import logging
import time
from pathlib import Path

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
        """Drops files past the TTL, then the oldest until the total is within budget (only on the
        node owning the cleanup job)."""
        if not self.may_clean():
            return
        with self._lock:
            now = time.time()
            expired = [
                n for n, e in self._index.items() if now - e.saved_at > self.policy.ttl_seconds
            ]
            doomed = list(expired)
            keep = sorted(
                ((n, e) for n, e in self._index.items() if n not in doomed),
                key=lambda item: item[1].saved_at,
            )
            total = sum(e.size for _, e in keep)
            while keep and total > self.policy.budget_bytes:
                name, entry = keep.pop(0)
                doomed.append(name)
                total -= entry.size
        for name in doomed:
            reason = (
                f"older than {self.policy.ttl_hours} h"
                if name in expired
                else f"over the {self.policy.budget_mb} MB limit (oldest first)"
            )
            self._remove(name, reason=reason)

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
