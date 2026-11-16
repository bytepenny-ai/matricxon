"""Reports progress while a long prompt is read, before the first token (a CPU can take minutes):
the chat stream sends a ChatProgressChunk each time the read moved on, so a client's timeout can
tell a slow read from a stuck one (pAIring's reply timeout waits while progress arrives). When the
cache is being compacted (a trimmed chat, LiveRequest.compacting) the first chunk goes out at once,
marked `compacting`, so a client can say why this reply takes longer."""

import asyncio
import time

from app.runtime.live_requests import LiveRequest
from app.schemas.chat import ChatProgressChunk, ChatReadProgress

# How often, while no token has come, the read is checked for progress to report.
PROGRESS_SECONDS = 10.0
# How soon a compacting cache is reported.
COMPACTING_SECONDS = 1.0


class ReadProgressQueue:
    def __init__(
        self,
        results: "asyncio.Queue[object]",
        live: LiveRequest,
        total: int,
        interval: float = PROGRESS_SECONDS,
    ) -> None:
        self._results = results
        self._live = live
        self._total = total
        self._interval = interval
        self._last = live.read_progress()
        self._last_sent = time.monotonic()
        self._reading = True
        self._told_compacting = False

    async def get(self) -> object:
        """The next item of `results` (a token id, None at the end, or an exception), or, while
        the prompt is still being read, a progress chunk (a dict) each time the read moved on."""
        while True:
            try:
                wait = min(self._interval, COMPACTING_SECONDS)
                item = await asyncio.wait_for(self._results.get(), timeout=wait)
            except TimeoutError:
                chunk = self._progress() if self._reading else None
                if chunk is not None:
                    return chunk
                continue
            self._reading = False
            return item

    def _progress(self) -> dict | None:
        progress = self._live.read_progress()
        compacting = bool(getattr(self._live, "compacting", False))
        tell = compacting and not self._told_compacting
        due = time.monotonic() - self._last_sent >= self._interval
        if not tell and (progress == self._last or not due):
            return None
        self._last, self._last_sent = progress, time.monotonic()
        self._told_compacting = self._told_compacting or compacting
        cached, read, layer = progress
        return ChatProgressChunk(
            prompt_progress=ChatReadProgress(
                cached=cached,
                read=read,
                total=self._total,
                layer=layer,
                compacting=True if compacting else None,
            )
        ).to_ndjson_dict()
