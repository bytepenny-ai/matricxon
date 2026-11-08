"""Pin entries as saved (MATRICXON_PINNED_MODELS, PUT /api/runtime-settings): "tag", "tag|ram" or
"tag|vram" (where it loads, app.models.placement; plain: auto), and one pin's state (PinState)."""

import logging
from dataclasses import dataclass

from app.models.handle import ModelHandle
from app.models.placement import MEMORY_WISHES

logger = logging.getLogger(__name__)


class PinLimitError(ValueError):
    """A pin list this node refuses (too many chat models, or a model not installed)."""


def parse_pins(text: str) -> list[str]:
    """MATRICXON_PINNED_MODELS: pins separated by commas, blanks dropped. A bad entry is logged
    and skipped, so a typo in .env never stops the server (the API refuses one with 409)."""
    good = []
    for entry in text.split(","):
        try:
            good += normalize_pins([entry])
        except PinLimitError as error:
            logger.warning("MATRICXON_PINNED_MODELS: skipped %r: %s", entry.strip(), error)
    return normalize_pins(good)


def split_pin(entry: str) -> tuple[str, str]:
    """ "tag|vram" -> ("tag", "vram"); no memory given: "auto". Raises PinLimitError."""
    tag, _sep, memory = entry.strip().partition("|")
    memory = memory.strip() or "auto"
    if memory not in MEMORY_WISHES:
        raise PinLimitError(f"{tag.strip()}: memory must be auto, ram or vram, not {memory!r}")
    return tag.strip(), memory


def normalize_pins(entries: list[str]) -> list[str]:
    """One entry per tag (in first-seen order, the last memory wins): "tag" or "tag|ram"."""
    pins: dict[str, str] = {}
    for entry in entries:
        if entry.strip():
            tag, memory = split_pin(entry)
            pins[tag] = memory
    return [tag if memory == "auto" else f"{tag}|{memory}" for tag, memory in pins.items()]


@dataclass
class PinState:
    tag: str
    memory: str = "auto"  # where it loads: auto, ram or vram (app.models.placement)
    size: int = (
        -1
    )  # its file's bytes, smaller pins load first; -1: not read yet (the loader reads it)
    state: str = "loading"  # loading -> loaded; or failed (after the retries) or unloaded (by hand)
    attempt: int = 0
    error: str | None = None
    next_try: float = 0.0
    handle: ModelHandle | None = None
