"""Saving a chat's prompt cache as a chain: one full copy (the base), then pieces holding only the
tokens added since the last save. A reply used to rewrite the whole cache (about 225 MB for a
2,500-token Gemma 4 chat) to a slow disk each time; a piece is a few MB.

Safety first: anything uncertain writes a full copy instead (a new chain; the old one is then
superseded and removed). A piece is written only when the chat's saved tokens are still exactly
its start (a rewound, branched or trimmed chat changes them), the chain is whole on disk, the
cache kind, context and precision are unchanged, and the chain is still short (`MAX_PIECES`,
`MAX_PIECE_SHARE`). A hybrid cache (recurrent state) is always saved whole.

A chain is restored only when complete (base and every piece in order); the file headers, sealed
with the body, say which chain, which place and which first token each file holds, so pieces can't
be mixed up. Limits treat a chain as one: aged by its newest piece, removed whole. A chain missing
a piece (a crash, or another node mid-restore) is removed once older than `BROKEN_GRACE_SECONDS`.
"""

import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass

from app.runtime.cache_files import StoredEntry

MAX_PIECES = 8  # pieces after the base before a new full copy
MAX_PIECE_SHARE = 0.5  # the pieces' tokens, as a share of the base's, before a new full copy
BROKEN_GRACE_SECONDS = 600.0


@dataclass(frozen=True)
class PersistedChain:
    """What a slot last saved (set by the writer thread, read only there): the chain on disk and the
    tokens it holds, to tell whether only a piece needs adding."""

    chain: str
    seq: int
    length: int
    tokens_hash: int
    num_ctx: int
    dtype_name: str
    base_tokens: int
    piece_tokens: int

    @staticmethod
    def tokens_hash_of(token_ids: list[int]) -> int:
        return hash(tuple(token_ids))


@dataclass(frozen=True)
class SavePlan:
    """How to save one copy: a full base (`start` 0, a new chain) or a piece from `start`."""

    full: bool
    chain: str
    seq: int
    start: int

    @staticmethod
    def decide(
        saved: PersistedChain | None,
        token_ids: list[int],
        num_ctx: int,
        dtype_name: str,
        hybrid: bool,
        chain_whole: bool,
    ) -> "SavePlan | None":
        """None: nothing new since the last save (the same tokens are on disk)."""
        if saved is not None and chain_whole and not hybrid:
            same_kind = saved.num_ctx == num_ctx and saved.dtype_name == dtype_name
            prefix = token_ids[: saved.length]
            continues = (
                same_kind
                and len(token_ids) >= saved.length
                and PersistedChain.tokens_hash_of(prefix) == saved.tokens_hash
            )
            if continues and len(token_ids) == saved.length:
                return None
            added = len(token_ids) - saved.length
            short = (
                saved.seq < MAX_PIECES
                and saved.piece_tokens + added <= saved.base_tokens * MAX_PIECE_SHARE
            )
            if continues and short:
                return SavePlan(False, saved.chain, saved.seq + 1, saved.length)
        return SavePlan(True, uuid.uuid4().hex, 0, 0)

    def after(
        self, saved: PersistedChain | None, token_ids: list[int], num_ctx: int, dtype_name: str
    ) -> PersistedChain:
        """The slot's saved state once this plan's file is written."""
        base = len(token_ids) if self.full else saved.base_tokens
        pieces = 0 if self.full else saved.piece_tokens + len(token_ids) - self.start
        return PersistedChain(
            self.chain,
            self.seq,
            len(token_ids),
            PersistedChain.tokens_hash_of(token_ids),
            num_ctx,
            dtype_name,
            base,
            pieces,
        )


@dataclass
class ChainGroup:
    """One saved chat's files: its base and pieces by place, newest saved first for the limits."""

    key: str
    pieces: dict[int, str]  # seq -> file name
    entries: dict[int, StoredEntry]

    @property
    def head_seq(self) -> int:
        """The last place reached without a gap from the base (-1: no base)."""
        seq = -1
        while seq + 1 in self.pieces:
            seq += 1
        return seq

    @property
    def whole(self) -> bool:
        return self.head_seq == max(self.pieces) and self.head_seq >= 0

    @property
    def head(self) -> tuple[str, StoredEntry] | None:
        seq = self.head_seq
        return None if seq < 0 else (self.pieces[seq], self.entries[seq])

    @property
    def names_newest_first(self) -> list[str]:
        """Restore claims the head first (of several nodes, whoever gets it owns the chain)."""
        return [self.pieces[s] for s in sorted(self.pieces, reverse=True)]

    @property
    def saved_at(self) -> float:
        return max(e.saved_at for e in self.entries.values())

    @property
    def size(self) -> int:
        return sum(e.size for e in self.entries.values())

    def broken_for(self, now: float | None = None) -> bool:
        """Missing a piece for longer than a restore or a write could take."""
        now = time.time() if now is None else now
        return not self.whole and now - self.saved_at > BROKEN_GRACE_SECONDS


class ChainIndex:
    @staticmethod
    def groups(items: Iterable[tuple[str, StoredEntry]]) -> list[ChainGroup]:
        """The files grouped by chain; a file from before chains is a group of one."""
        groups: dict[str, ChainGroup] = {}
        for name, entry in items:
            key = entry.chain or name
            group = groups.setdefault(key, ChainGroup(key, {}, {}))
            seq = entry.seq if entry.chain else 0
            if seq in group.pieces:  # two files claim one place: keep the newer, the chain decides
                if group.entries[seq].saved_at >= entry.saved_at:
                    continue
            group.pieces[seq] = name
            group.entries[seq] = entry
        return list(groups.values())

    @staticmethod
    def heads(
        items: Iterable[tuple[str, StoredEntry]],
    ) -> list[tuple[ChainGroup, str, StoredEntry]]:
        """Each restorable chat: its group and its last piece (the whole chat's tokens and length).
        A chain cut short at its end (a piece never written) still restores up to its last piece."""
        found = []
        for group in ChainIndex.groups(items):
            head = group.head
            if head is not None:
                found.append((group, head[0], head[1]))
        return found

    @staticmethod
    def whole_on_disk(items: Iterable[tuple[str, StoredEntry]], chain: str, seq: int) -> bool:
        """Is `chain` on disk with every place up to `seq`, so a piece after it can follow?"""
        group = next((g for g in ChainIndex.groups(items) if g.key == chain), None)
        return group is not None and group.head_seq >= seq and max(group.pieces) == seq
