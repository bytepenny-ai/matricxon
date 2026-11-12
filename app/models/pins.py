"""Pinned models: loaded in the background after start and kept (MATRICXON_PINNED_MODELS).

A pin is a wish, not a reservation. Each pin is tried up to RETRIES times, RETRY_SECONDS apart; then
it is `failed` until it is saved again or the server restarts, and its memory is used as usual. An
unload request still unloads a pinned model (`unloaded`); it loads again when saved again or at the
next start. Pinned chat models count toward max_loaded_models; helpers (embedding, rerank) don't.

Each pin may say where it loads (app.models.placement): "tag|ram", "tag|vram", or "tag" (auto: VRAM
when the server uses a GPU and the model fits, else RAM). A pinned model sitting in the other memory
loads again in the right one once its replies are done.
"""

import logging
import threading
import time
from collections.abc import Callable

from app.models.load_limit import LoadLimit
from app.models.manager import ModelManager
from app.models.pin_entries import PinLimitError, PinState, normalize_pins, parse_pins, split_pin
from app.models.placement import ModelPlacement
from app.server.errors import UnknownModelError

__all__ = ["ModelPins", "PinLimitError", "PinState", "normalize_pins", "parse_pins", "split_pin"]

logger = logging.getLogger(__name__)


class ModelPins:
    RETRIES = 3
    RETRY_SECONDS = 60.0

    def __init__(
        self,
        manager: ModelManager,
        limit: LoadLimit,
        installed: Callable[[], dict[str, int]],  # tag -> file bytes
        tags: list[str] | None = None,
        retry_seconds: float = RETRY_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        placement: ModelPlacement | None = None,
        voter_only: bool = False,
    ) -> None:
        self._manager = manager
        self._limit = limit
        self._installed = installed
        self._retry_seconds = retry_seconds
        self._clock = clock
        self._placement = placement
        self._voter_only = voter_only
        self._states: dict[str, PinState] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stopping = False
        self._thread: threading.Thread | None = None
        self.set(tags or [])

    def tags(self) -> list[str]:
        """The pins as saved: "tag" or "tag|ram"."""
        with self._lock:
            states = list(self._states.values())
        return [s.tag if s.memory == "auto" else f"{s.tag}|{s.memory}" for s in states]

    def check(self, entries: list[str], max_loaded: int) -> None:
        """Raises PinLimitError when this node can't take these pins."""
        pins = dict(split_pin(entry) for entry in entries)
        tags = list(pins)
        if pins and self._voter_only:
            raise PinLimitError("this is a voter-only node: it never loads models")
        if (
            "vram" in pins.values()
            and self._placement is not None
            and self._placement.gpu() is None
        ):
            raise PinLimitError("this server has no GPU: pin to RAM (or auto) instead of VRAM")
        # Only new pins must be installed: removing one (e.g. after its model was deleted) is never
        # blocked by another pin whose model is gone too.
        current = {split_pin(entry)[0] for entry in self.tags()}
        missing = [t for t in tags if t not in current and t not in self._installed()]
        if missing:
            raise PinLimitError(f"not installed on this node: {', '.join(missing)}")
        chat = [tag for tag in tags if self._limit.counts(tag)]
        if len(chat) > max_loaded:
            raise PinLimitError(
                f"{len(chat)} chat models pinned, but this node keeps at most {max_loaded} loaded "
                "(max_loaded_models): pin fewer or raise the limit"
            )

    def set(self, entries: list[str]) -> None:
        """New pins load in the background; a removed pin becomes an ordinary model (keep-alive
        applies again); a failed or unloaded pin saved again, or one moved to the other memory, is
        loaded again. Each pin's memory goes to the loader (LoadSettings.model_memory)."""
        pins = dict(split_pin(entry) for entry in entries)
        with self._lock:
            for tag in [t for t in self._states if t not in pins]:
                self._release(self._states.pop(tag))
            for tag, memory in pins.items():
                state = self._states.get(tag)
                if state is None or state.state in ("failed", "unloaded") or state.memory != memory:
                    if state is not None and state.memory != memory:
                        self._release(state)
                    self._states[tag] = PinState(tag, memory)
            self._states = {tag: self._states[tag] for tag in pins}
            wishes = getattr(getattr(self._manager, "settings", None), "model_memory", None)
            if wishes is not None:
                wishes.clear()
                wishes.update({s.tag: s.memory for s in self._states.values()})
        self._wake.set()

    def _sizes(self) -> dict[str, int]:
        try:
            return dict(self._installed())
        except Exception:  # noqa: BLE001 - sizes only order the loads; unknown: 0
            return {}

    def _release(self, state: PinState) -> None:
        """Its model stays loaded as an ordinary one (keep-alive applies again)."""
        if state.handle is not None:
            state.handle.pinned = False
            state.handle.touch(self._manager.now())

    def start(self) -> None:
        """Starts the background loader; the server answers meanwhile."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="model-pins", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stopping = True
        self._wake.set()

    def report(self) -> list[PinState]:
        """Each pin's state; a pinned model unloaded by hand shows as `unloaded`."""
        loaded = {handle.tag: handle for handle in self._manager.list_loaded()}
        with self._lock:
            for state in self._states.values():
                if state.state == "loaded" and loaded.get(state.tag) is not state.handle:
                    state.state, state.handle = "unloaded", None
            return [
                PinState(s.tag, s.memory, s.size, s.state, s.attempt, s.error)
                for s in self._states.values()
            ]

    def _due(self) -> PinState | None:
        """The smallest pin due to load: small helpers (embedding models) are ready in seconds
        instead of waiting behind a big chat model. Sizes are read here, in the loader thread: the
        models folder can take a minute to read on a slow disk; the server must start at once."""
        now = self._clock()
        with self._lock:
            due = [s for s in self._states.values() if s.state == "loading" and s.next_try <= now]
        if any(s.size < 0 for s in due):
            sizes = self._sizes()
            for state in due:
                if state.size < 0:
                    state.size = sizes.get(state.tag, 0)
        return min(due, key=lambda s: s.size, default=None)

    def _wait_seconds(self) -> float | None:
        with self._lock:
            waits = [s.next_try for s in self._states.values() if s.state == "loading"]
        return max(0.0, min(waits) - self._clock()) if waits else None

    def _run(self) -> None:
        while not self._stopping:
            state = self._due()
            if state is None:
                self._wake.wait(self._wait_seconds())
                self._wake.clear()
                continue
            self.load_one(state)

    def _misplaced(self, state: PinState) -> bool:
        """True while the model sits in the other memory and is busy (checked again soon); when it
        is idle it is unloaded here, so the attempt loads it in the right place."""
        handle = next((h for h in self._manager.list_loaded() if h.tag == state.tag), None)
        # Loaded before it was pinned, so mapped from disk: load it again, copied into RAM.
        mapped = handle is not None and not handle.on_gpu and not handle.resident
        if handle is None or not (ModelPlacement.misplaced(state.memory, handle.on_gpu) or mapped):
            return False
        if handle.worker.is_busy():
            state.next_try = self._clock() + 5.0
            return True
        where = "RAM to stay" if mapped else state.memory.upper()
        logger.info("moving pinned model %s to %s", state.tag, where)
        self._manager.unload(state.tag)
        return False

    def _over_the_limit(self, state: PinState) -> bool:
        """A chat pin past max_loaded_models (a hand-edited .env; the API refuses it): failed at
        once, no retries, so pinned models never push the node past its limit."""
        if not self._limit.counts(state.tag):
            return False
        held = sum(1 for h in self._manager.list_loaded() if h.pinned and h.counts_toward_limit)
        limit = self._manager.settings.max_loaded
        if held < limit:
            return False
        with self._lock:
            state.state, state.error = (
                "failed",
                (
                    f"over the limit: {limit} chat models pinned and loaded already "
                    "(max_loaded_models)"
                ),
            )
        logger.warning("pinned model %s not loaded: %s", state.tag, state.error)
        return True

    def load_one(self, state: PinState) -> None:
        """One attempt for one pin, one at a time (the background thread; tests call it)."""
        if self._misplaced(state) or self._over_the_limit(state):
            return
        state.attempt += 1
        logger.info(
            "loading pinned model %s (attempt %d of %d)", state.tag, state.attempt, self.RETRIES
        )
        try:
            handle = self._manager.get_or_load(state.tag)
        except Exception as error:  # noqa: BLE001 - any failure is a failed attempt, never a crash
            with self._lock:
                if self._states.get(state.tag) is not state:
                    return  # unpinned meanwhile
                state.error = str(error) or type(error).__name__
                if isinstance(error, UnknownModelError):
                    state.error = f"{state.error}: not installed any more (deleted?), unpin it"
                if state.attempt >= self.RETRIES or isinstance(error, UnknownModelError):
                    state.state = "failed"
                    logger.warning(
                        "pinned model %s not loaded, gave up: %s", state.tag, state.error
                    )
                else:
                    state.next_try = self._clock() + self._retry_seconds
            return
        with self._lock:
            unpinned = self._states.get(state.tag) is not state
            if not unpinned:
                handle.pinned = True
                state.state, state.handle, state.error = "loaded", handle, None
        if unpinned:
            # An ordinary model now; but if it was unpinned because it was deleted, don't keep it.
            if state.tag not in self._sizes():
                logger.info(
                    "unloading %s: unpinned while loading, and no longer installed", state.tag
                )
                self._manager.unload(state.tag)
            return
        logger.info("pinned model %s loaded", state.tag)
