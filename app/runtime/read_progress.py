"""Reports progress while a long prompt is read, before the first token (a CPU can take minutes):
the chat stream sends a ChatProgressChunk each time the read moved on, so a client's timeout can
tell a slow read from a stuck one (pAIring's reply timeout waits while progress arrives)."""

import asyncio

from app.runtime.live_requests import LiveRequest
from app.schemas.chat import ChatProgressChunk, ChatReadProgress

# How often, while no token has come, the read is checked for progress to report.
PROGRESS_SECONDS = 10.0


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
        self._reading = True

    async def get(self) -> object:
        """The next item of `results` (a token id, None at the end, or an exception), or, while
        the prompt is still being read, a progress chunk (a dict) each time the read moved on."""
        while True:
            try:
                item = await asyncio.wait_for(self._results.get(), timeout=self._interval)
            except TimeoutError:
                progress = self._live.read_progress()
                if self._reading and progress != self._last:
                    self._last = progress
                    cached, read, layer = progress
                    return ChatProgressChunk(
                        prompt_progress=ChatReadProgress(
                            cached=cached, read=read, total=self._total, layer=layer
                        )
                    ).to_ndjson_dict()
                continue
            self._reading = False
            return item
