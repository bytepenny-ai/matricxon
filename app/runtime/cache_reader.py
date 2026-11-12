import logging
import os
import struct
import time

import torch

from app.runtime.cache_chain import ChainIndex
from app.runtime.cache_crypto import CacheDecryptError
from app.runtime.cache_files import MAGIC, CacheFilesBase, StoredEntry

logger = logging.getLogger("app.runtime.cache_store")


class CacheReader(CacheFilesBase):
    """The restoring half of `PersistentCacheStore`: finds the stored chat sharing the longest
    prefix with a prompt and consumes its files (a chain: its base and pieces, cache_chain.py)."""

    def take(
        self,
        model_id: str,
        prompt_ids: list[int],
        dtype: torch.dtype,
        min_tokens: int,
        tag: str = "",
    ) -> tuple[list[bytes], str] | None:
        """The saved chat sharing the longest prefix (more than `min_tokens`) with `prompt_ids`, as
        its parts in order (a base, then its pieces), removed from disk, with the chat tag it was
        saved under; or None. Matching uses the sealed per-block hashes of each chain's last
        piece, so files that wouldn't help are never read, let alone consumed."""
        if not self.policy.enabled:
            return None
        # A save still being written (e.g. this model's own unload a moment ago) would be missed.
        # Copies of chats still in RAM (`checkpoint_async`) aren't waited for.
        if self._spills_pending:
            self._writer.submit(lambda: None).result()
        self._rescan_if_due()
        wanted = self._cipher.prefix_fingerprints(prompt_ids, self._block_tokens())
        best = self._best_match(model_id, dtype, wanted, min_tokens, tag)
        if best is None and time.monotonic() - self._last_scan >= self.MISS_RESCAN_SECONDS:
            self._refresh()
            best = self._best_match(model_id, dtype, wanted, min_tokens, tag)
        if best is None:
            return None
        bodies = self._consume_chain(best[1])
        return None if bodies is None else (bodies, best[2])

    def _best_match(
        self, model_id: str, dtype: torch.dtype, wanted: list[str], min_tokens: int, tag: str = ""
    ) -> tuple[int, list[str], str] | None:
        """(shared blocks, the chain's file names head first, chat tag) of the saved chat sharing
        the longest prefix (only whole chains; a file from before chains is one of its own). Any
        precision: a restore converts it (`KVCache.load`). On a tie (a trimmed chat shares only its
        system prompt with every copy) the asking chat's own copy wins, then the same precision,
        then the copy holding more tokens: a chat's old branch must not beat its newest state."""
        best: tuple[int, list[str], str] | None = None
        best_rank: tuple[int, bool, bool, int] | None = None
        block = self._block_tokens()
        with self._lock:
            items = [(n, e) for n, e in self._index.items() if e.model_id == model_id]
        for group, _name, entry in ChainIndex.heads(items):
            shared = 0
            for mine, theirs in zip(wanted, entry.blocks, strict=False):
                if mine != theirs:
                    break
                shared += 1
            rank = (
                shared,
                bool(tag) and entry.tag == tag,
                entry.dtype_name == str(dtype),
                entry.length,
            )
            if shared * block > min_tokens and (best_rank is None or rank > best_rank):
                head = group.head_seq
                names = [group.pieces[s] for s in range(head, -1, -1)]  # the head first
                best, best_rank = (shared, names, entry.tag), rank
        return best

    def inspect(self, prompt_ids: list[int]) -> tuple[list[StoredEntry], list[str]]:
        """The files' headers plus `prompt_ids`' block hashes, to explain a miss."""
        with self._lock:
            items = list(self._index.items())
        entries = [entry for _group, _name, entry in ChainIndex.heads(items)]  # one per saved chat
        return entries, self._cipher.prefix_fingerprints(prompt_ids, self._block_tokens())

    def _consume_chain(self, names: list[str]) -> list[bytes] | None:
        """Claims a chain's files, its head first (of several nodes, whoever gets the head owns the
        chain), reads them and deletes them; the bodies come back base first. A part another node
        took, or one that fails to read, makes it a miss, and what was claimed is removed."""
        bodies: list[bytes] = []
        for index, name in enumerate(names):
            body = self._consume(name)
            if body is None:
                for rest in names[index + 1 :]:
                    self._remove(rest, reason="its chain is incomplete")
                if index:
                    logger.warning(
                        "prompt cache: a saved chat's chain was incomplete, not restored"
                    )
                return None
            bodies.append(body)
        return bodies[::-1]

    def _consume(self, name: str) -> bytes | None:
        """Claims the file by renaming it (atomic: of several nodes, one wins), reads, deletes."""
        path = self._dir / f"{name}{self._claim_suffix}"
        try:
            os.replace(self._dir / name, path)
        except FileNotFoundError:
            with self._lock:
                self._index.pop(name, None)
            return None
        try:
            with path.open("rb") as handle:
                handle.seek(len(MAGIC))
                (meta_len,) = struct.unpack(">I", handle.read(4))
                meta = handle.read(meta_len)
                body = self._cipher.open(handle.read(), MAGIC + meta)
        except (CacheDecryptError, OSError, struct.error):
            self._remove(name, path, reason="unreadable (wrong key or damaged)")
            return None
        self._remove(name, path)
        return body
