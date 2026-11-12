import logging
import os
import struct
import tempfile
import time
import uuid

from app.runtime.cache_chain import ChainIndex, SavePlan
from app.runtime.cache_files import MAGIC, SUFFIX, CacheFilesBase, StoredEntry
from app.runtime.cache_superseded import SupersededCopies
from app.runtime.prompt_slot import CacheSlot
from app.runtime.slot_codec import EncodedSlot, SlotCodec, SlotState

logger = logging.getLogger("app.runtime.cache_store")


class CacheWriter(CacheFilesBase):
    """The writing half of `PersistentCacheStore`: chats leaving RAM (`spill_async`) and copies of
    chats still in RAM (`checkpoint_async`), encoded, sealed and written by one background
    thread."""

    def spill_async(self, model_id: str, slot: CacheSlot) -> None:
        """Queues `slot` for the background writer to encode and write; the caller must not touch
        the slot afterwards (the pool has dropped it). A copy of the same chat still queued is
        dropped: this one is newer, and two saves of one chat must never both add a piece."""
        if self.policy.enabled and self.policy.budget_bytes > 0:
            with self._lock:
                self._spills_pending += 1
                self._checkpoints.pop((model_id, slot.tag), None)
            self._writer.submit(self._spill, model_id, slot)

    def checkpoint_async(
        self, model_id: str, state: SlotState, tag: str, slot: CacheSlot | None = None
    ) -> None:
        """Writes a copy of a chat still in RAM. A newer copy of the same chat queued before this
        one is written replaces it, so a slow disk never falls behind by more than one copy. `slot`
        records what was saved, so the next copy adds only a piece (cache_chain.py)."""
        if not (self.policy.enabled and self.policy.budget_bytes > 0):
            return
        with self._lock:
            queued = (model_id, tag) in self._checkpoints
            self._checkpoints[(model_id, tag)] = (state, slot)
        if not queued:
            self._writer.submit(self._write_checkpoint, model_id, tag)

    def _write_checkpoint(self, model_id: str, tag: str) -> None:
        with self._lock:
            queued = self._checkpoints.pop((model_id, tag), None)
        if queued is None:
            return  # the chat left RAM meanwhile: its spill writes the newest state
        state, slot = queued
        try:
            if tag not in self._forgotten:
                self._save(model_id, state, tag, slot)
        except Exception:  # noqa: BLE001 - a failed copy only costs a later re-prefill
            self._lost_track(slot)
            logger.exception("saving a prompt cache copy failed")

    def holds_copy(self, model_id: str, tag: str, length: int) -> bool:
        """Is a file of this chat with exactly `length` tokens on disk (or queued)?"""
        with self._lock:
            return (model_id, tag) in self._checkpoints or any(
                e.model_id == model_id and e.tag == tag and e.length == length
                for e in self._index.values()
            )

    def flush(self) -> None:
        """Blocks until every queued spill is on disk (shutdown)."""
        started = time.monotonic()
        self._writer.submit(lambda: None).result()
        if time.monotonic() - started > 1.0:
            logger.info("prompt caches written to disk in %.1fs", time.monotonic() - started)

    def _spill(self, model_id: str, slot: CacheSlot) -> None:
        try:
            state = SlotCodec.capture(slot)
            if state is not None and slot.tag not in self._forgotten:
                self._save(model_id, state, slot.tag, slot)
        except Exception:  # noqa: BLE001 - a failed spill only costs a later re-prefill
            self._lost_track(slot)
            logger.exception("persisting a prompt cache failed")
        finally:
            with self._lock:
                self._spills_pending -= 1

    def _save(self, model_id: str, state: SlotState, tag: str, slot: CacheSlot | None) -> None:
        """Writes a full copy, or only a piece when the chat just grew since its last save on a
        chain that is still whole (SavePlan.decide; writer thread only)."""
        token_ids = state.tensors["token_ids"]
        saved = getattr(slot, "persisted", None)
        hybrid = bool(state.tensors["snapshots"])
        plan = SavePlan.decide(
            saved,
            token_ids,
            state.num_ctx,
            state.dtype_name,
            hybrid,
            saved is not None and self._chain_on_disk(model_id, saved.chain, saved.seq),
        )
        if plan is None:
            return
        encoded = (
            SlotCodec.serialize(state)
            if plan.full
            else SlotCodec.serialize_piece(state, plan.start)
        )
        written = self._write(model_id, encoded, tag, plan)
        if slot is not None:
            slot.persisted = (
                plan.after(saved, token_ids, state.num_ctx, state.dtype_name) if written else None
            )

    def _chain_on_disk(self, model_id: str, chain: str, seq: int) -> bool:
        """Every file of `chain` up to `seq` is indexed and still in the folder (another node or a
        limit may have removed one); else the next save is a full copy."""
        with self._lock:
            items = [(n, e) for n, e in self._index.items() if e.model_id == model_id]
        if not ChainIndex.whole_on_disk(items, chain, seq):
            return False
        return all((self._dir / n).exists() for n, e in items if e.chain == chain)

    @staticmethod
    def _lost_track(slot: CacheSlot | None) -> None:
        """A failed save: the next one writes a full copy rather than a piece after a gap."""
        if slot is not None:
            slot.persisted = None

    def _write(
        self, model_id: str, encoded: EncodedSlot, tag: str = "", plan: SavePlan | None = None
    ) -> bool:
        """True when the file is on disk and kept (not too big, its chat not deleted meanwhile)."""
        plan = plan or SavePlan(True, uuid.uuid4().hex, 0, 0)
        entry = StoredEntry(
            model_id,
            encoded.num_ctx,
            encoded.dtype_name,
            len(encoded.token_ids),
            self._cipher.prefix_fingerprints(encoded.token_ids, self._block_tokens()),
            time.time(),
            tag,
            plan.chain,
            plan.seq,
            plan.start,
        )
        meta = self._cipher.seal(entry.meta_json(), MAGIC)
        body = self._cipher.seal(encoded.payload, MAGIC + meta)
        blob = MAGIC + struct.pack(">I", len(meta)) + meta + body
        if len(blob) > self.policy.budget_bytes:
            return False
        fd, tmp = tempfile.mkstemp(dir=self._dir, prefix=f"{self._node_id}-", suffix=".tmp")
        with os.fdopen(fd, "wb") as handle:
            handle.write(blob)
        name = f"{uuid.uuid4().hex}{SUFFIX}"
        os.replace(tmp, self._dir / name)
        entry.size = len(blob)
        with self._lock:
            self._index[name] = entry
            gone = tag in self._forgotten
            superseded = SupersededCopies.find(name, entry, list(self._index.items()))
        if gone:
            self._remove(name, reason="its chat was deleted while it was being saved")
        for old in superseded:
            self._remove(old, reason="replaced by a newer copy of the same chat")
        self.enforce_limits()
        return not gone
