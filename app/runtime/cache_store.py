import json
import logging
import struct
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.runtime.cache_chain import ChainIndex
from app.runtime.cache_crypto import CacheCipher, CacheDecryptError
from app.runtime.cache_files import MAGIC, SUFFIX, StoredEntry
from app.runtime.cache_limits import CacheLimits
from app.runtime.cache_policy import PersistencePolicy
from app.runtime.cache_reader import CacheReader
from app.runtime.cache_superseded import SupersededCopies
from app.runtime.cache_writer import CacheWriter
from app.runtime.slot_codec import SlotState

logger = logging.getLogger(__name__)

# Token granularity of prefix matching between a saved cache and a new prompt.
BLOCK_TOKENS = 64

__all__ = ["BLOCK_TOKENS", "MAGIC", "SUFFIX", "PersistentCacheStore", "StoredEntry"]


class PersistentCacheStore(CacheWriter, CacheReader, CacheLimits):
    """Encrypted, size- and age-limited prompt-cache files on disk: the tier below the RAM pool.
    Writing is in cache_writer.py, restoring in cache_reader.py, the limits and admin removals in
    cache_limits.py; this class owns the directory index.

    A slot is written when the pool evicts it, when its model unloads and on shutdown, never during
    a reply. File layout: `MAGIC | len | sealed meta | sealed body`, both AES-256-GCM sealed (body
    bound to meta), so nothing about a conversation (tokens, model, length) is readable without the
    key. The index is rebuilt at startup from the meta headers; a file failing authentication (wrong
    key, corruption) is deleted.

    A stored cache is consumed on restore (`take`): the RAM pool then owns it and writes it back
    when evicted. Limits come from the live `PersistencePolicy`: oldest files go first when over
    budget, and files older than the TTL are dropped.
    """

    # A lookup that finds nothing re-reads the shared dir if this long since the last read.
    MISS_RESCAN_SECONDS = 0.1

    def __init__(
        self,
        directory: Path,
        cipher: CacheCipher,
        policy: PersistencePolicy,
        node_id: str = "local",
        rescan_seconds: float = 5.0,
        scan_budget_seconds: float | None = None,
    ) -> None:
        self._dir = directory
        self._cipher = cipher
        self.policy = policy
        self._node_id = node_id
        self._rescan_seconds = rescan_seconds
        self._last_scan = 0.0
        # In a cluster only the owner of the cleanup lease deletes files by age or budget, so none
        # removes files the others count on. Alone, every node cleans.
        self.may_clean: Callable[[], bool] = lambda: True
        self._rejected: set[str] = set()
        self._lock = threading.Lock()
        self._index: dict[str, StoredEntry] = {}
        self._forgotten: set[str] = set()
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cache-spill")
        self._checkpoints: dict[tuple[str, str], SlotState] = {}  # newest copy waiting per chat
        self._spills_pending = 0  # chats leaving RAM, not yet on disk
        self._dir.mkdir(parents=True, exist_ok=True)
        self._startup_scan(scan_budget_seconds)

    # reading the directory

    def _startup_scan(self, budget_seconds: float | None) -> None:
        """Indexes saved chats before the server answers. Past `budget_seconds` (None: no limit) the
        rest continues on the writer thread, from where it stopped: a cold read costs ~140 ms per
        file on an NFS-shared dir, and a cluster's dir can hold many chats. Spills queue behind it;
        a chat not yet indexed is re-read once, like any miss."""
        started = time.monotonic()
        deadline = None if budget_seconds is None else started + budget_seconds
        if self._scan(deadline):
            self.enforce_limits()
            logger.info(
                "prompt cache: indexed %d saved chats in %.1fs",
                len(self._index),
                time.monotonic() - started,
            )
            return
        logger.info(
            "prompt cache: indexed %d saved chats in %.1fs, the rest continues in the background",
            len(self._index),
            time.monotonic() - started,
        )
        self._writer.submit(self._finish_startup_scan, started)

    def _finish_startup_scan(self, started: float) -> None:
        try:
            self._refresh(drop_unreadable=True)
            self.enforce_limits()
        except Exception:
            logger.exception("prompt cache: background scan of %s failed", self._dir)
            return
        logger.info(
            "prompt cache: all %d saved chats indexed after %.1fs",
            len(self._index),
            time.monotonic() - started,
        )

    def _scan(self, deadline: float | None = None) -> bool:
        """Startup: drops this node's own leftovers (other nodes' are theirs) and indexes. False
        when `deadline` came first."""
        leftovers = [
            *self._dir.glob(f"{self._node_id}-*.tmp"),
            *self._dir.glob(f"*{self._claim_suffix}"),
        ]
        for path in leftovers:
            path.unlink(missing_ok=True)
        if leftovers:
            logger.info(
                "prompt cache: removed %d unfinished file(s) left by an earlier run", len(leftovers)
            )
        return self._refresh(drop_unreadable=True, deadline=deadline)

    def _refresh(self, drop_unreadable: bool = False, deadline: float | None = None) -> bool:
        """Syncs the index with the directory: files other nodes wrote or removed. Stops early
        (False) at `deadline`; a later call continues with the files still unknown."""
        self._last_scan = time.monotonic()
        on_disk = {p.name: p for p in self._dir.glob(f"*{SUFFIX}")}
        with self._lock:
            known = set(self._index)
        for name in known - on_disk.keys():
            with self._lock:
                entry = self._index.pop(name, None)
            if entry is not None:
                # Not removed here: another node took it, or something outside Matricxon deleted it.
                logger.info(
                    "prompt cache: saved chat %s disappeared from the folder "
                    "(taken by another node, or deleted outside Matricxon)",
                    entry.describe(),
                )
        for name, path in on_disk.items():
            if name in known or name in self._rejected:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                return False
            try:
                entry = self._read_meta(path)
            except FileNotFoundError:
                continue
            except (CacheDecryptError, OSError, ValueError, KeyError, TypeError):
                if drop_unreadable:
                    logger.warning(
                        "prompt cache: removed file %s - unreadable (wrong key or damaged)", name
                    )
                    path.unlink(missing_ok=True)
                else:
                    self._rejected.add(name)
                continue
            with self._lock:
                self._index[name] = entry
        return True

    @property
    def _claim_suffix(self) -> str:
        return f".{self._node_id}.claim"

    def _rescan_if_due(self) -> None:
        if time.monotonic() - self._last_scan >= self._rescan_seconds:
            self._refresh()

    def _block_tokens(self) -> int:
        return BLOCK_TOKENS  # read at call time: tests shrink it

    def _read_meta(self, path: Path) -> StoredEntry:
        with path.open("rb") as handle:
            if handle.read(len(MAGIC)) != MAGIC:
                raise ValueError("not a cache file")
            (meta_len,) = struct.unpack(">I", handle.read(4))
            meta = json.loads(self._cipher.open(handle.read(meta_len), MAGIC))
        return StoredEntry(**meta, size=path.stat().st_size)

    def chat_count(self) -> int:
        """Distinct chats with a saved copy (one chat can have several files, e.g. branches)."""
        self._rescan_if_due()
        with self._lock:
            items = list(self._index.items())
        return SupersededCopies.chat_count(e for _g, _n, e in ChainIndex.heads(items))

    def usage(self) -> tuple[int, int]:
        """(files, total bytes) currently stored."""
        self._rescan_if_due()
        with self._lock:
            return len(self._index), sum(e.size for e in self._index.values())
