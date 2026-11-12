import io
from dataclasses import dataclass

import torch

from app.architectures.base import ModelArchitecture
from app.runtime.mamba_cache import HybridSnapshot, NemotronHHybridCache
from app.runtime.prompt_slot import CacheSlot

# Tokens below this are cheaper to recompute than to write and read back.
MIN_PERSIST_TOKENS = 64


@dataclass
class EncodedSlot:
    """A slot ready to be sealed: what the store indexes it by, plus the serialized tensors."""

    token_ids: list[int]
    num_ctx: int
    dtype_name: str
    payload: bytes


@dataclass
class SlotState:
    """A slot's saved part, copied out (`SlotCodec.capture`) and not yet serialized."""

    tensors: dict
    num_ctx: int
    dtype_name: str


class SlotCodec:
    """Turns a `CacheSlot` into bytes and back. Only what a later request can reuse is written: a
    plain cache's whole KV; a hybrid cache's KV up to its newest recurrent snapshot (past that
    nothing can be resumed) plus the snapshots, never the live recurrent state, which a snapshot
    always replaces before a slot is used again (`CacheSlot.rewind`/`fork`)."""

    @staticmethod
    def encode(slot: CacheSlot) -> EncodedSlot | None:
        """None when the slot isn't worth persisting (too short, or a hybrid without snapshots)."""
        state = SlotCodec.capture(slot)
        return None if state is None else SlotCodec.serialize(state)

    @staticmethod
    def capture(slot: CacheSlot) -> SlotState | None:
        """A copy of what `encode` writes, independent of the slot from then on (the KV is cloned),
        so it can be serialized on another thread while the slot serves the next reply."""
        keep = max(slot.snapshots, default=0) if slot.hybrid else len(slot.token_ids)
        if keep < MIN_PERSIST_TOKENS:
            return None
        keys, values = slot.cache.export(keep)
        num_ctx, dtype = slot.key
        state = {
            "token_ids": slot.token_ids[:keep],
            "prompt_len": min(slot.prompt_len, keep),
            "keys": keys,
            "values": values,
            "snapshots": [
                {"length": s.length, "conv": s.conv_state, "ssm": s.ssm_state}
                for s in slot.snapshots.values()
            ],
        }
        return SlotState(state, num_ctx, str(dtype))

    @staticmethod
    def serialize(state: SlotState) -> EncodedSlot:
        buffer = io.BytesIO()
        torch.save(state.tensors, buffer)
        return EncodedSlot(
            state.tensors["token_ids"], state.num_ctx, state.dtype_name, buffer.getvalue()
        )

    @staticmethod
    def serialize_piece(state: SlotState, start: int) -> EncodedSlot:
        """Only the tokens from `start` on (a chain's piece, cache_chain.py): their ids and their
        keys/values. `token_ids` stays the whole list, for the file's header (length, matching)."""
        tensors = state.tensors
        piece = {
            "start": start,
            "token_ids": tensors["token_ids"][start:],
            "prompt_len": tensors["prompt_len"],
            "keys": [k[:, :, start:, :].clone() for k in tensors["keys"]],
            "values": [v[:, :, start:, :].clone() for v in tensors["values"]],
            "snapshots": [],
        }
        buffer = io.BytesIO()
        torch.save(piece, buffer)
        return EncodedSlot(tensors["token_ids"], state.num_ctx, state.dtype_name, buffer.getvalue())

    @staticmethod
    def join(payloads: list[bytes]) -> dict:
        """A base and its pieces, in order, as one saved state. Raises `ValueError` unless each
        piece starts exactly where the state so far ends and its sizes agree."""
        if not payloads:
            raise ValueError("no saved parts")
        load = lambda p: torch.load(io.BytesIO(p), weights_only=True, map_location="cpu")  # noqa: E731
        state = load(payloads[0])
        if state.get("start", 0) != 0:
            raise ValueError("a chain must begin with its full copy")
        for payload in payloads[1:]:
            piece = load(payload)
            length = len(state["token_ids"])
            added = len(piece["token_ids"])
            if state["snapshots"] or piece.get("start") != length or added == 0:
                raise ValueError("a saved piece doesn't continue the chat")
            if len(piece["keys"]) != len(state["keys"]) or len(piece["values"]) != len(
                state["values"]
            ):
                raise ValueError("a saved piece has a different number of layers")
            for part in (*piece["keys"], *piece["values"]):
                if part.shape[2] != added:
                    raise ValueError("a saved piece's sizes disagree")
            state["keys"] = [
                torch.cat([k, p.to(k.dtype)], dim=2)
                for k, p in zip(state["keys"], piece["keys"], strict=True)
            ]
            state["values"] = [
                torch.cat([v, p.to(v.dtype)], dim=2)
                for v, p in zip(state["values"], piece["values"], strict=True)
            ]
            state["token_ids"] = [*state["token_ids"], *piece["token_ids"]]
            state["prompt_len"] = piece["prompt_len"]
        if state["keys"] and state["keys"][0].shape[2] != len(state["token_ids"]):
            raise ValueError("a saved cache's tokens and keys disagree")
        return state

    @staticmethod
    def decode(
        payload: bytes | list[bytes],
        architecture: ModelArchitecture,
        num_ctx: int,
        dtype: torch.dtype,
    ) -> CacheSlot:
        """Rebuilds the slot on a fresh cache from `architecture`, from one file or a chain's parts
        in order (`join`). Raises `ValueError` if it doesn't fit this model (a stale save) or the
        parts don't fit together; the caller treats it as a miss."""
        state = SlotCodec.join(payload if isinstance(payload, list) else [payload])
        cache = architecture.build_cache(max_seq_len=num_ctx, dtype=dtype)
        snapshots = {
            # Converted like the KV in `cache.load`: the model may have loaded in another precision.
            s["length"]: HybridSnapshot(
                s["length"], [t.to(dtype) for t in s["conv"]], [t.to(dtype) for t in s["ssm"]]
            )
            for s in state["snapshots"]
        }
        if bool(snapshots) != isinstance(cache, NemotronHHybridCache):
            raise ValueError("saved cache kind differs from this model's")
        cache.load(state["keys"], state["values"])
        return CacheSlot(
            cache, (num_ctx, dtype), state["token_ids"], snapshots, prompt_len=state["prompt_len"]
        )
