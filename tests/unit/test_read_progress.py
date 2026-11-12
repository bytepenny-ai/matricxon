"""read_progress.py: while a long prompt is read, the chat stream reports each step of the read
(so a client's timeout can wait for a slow read), and stops once tokens flow."""

import asyncio

from app.runtime.read_progress import ReadProgressQueue
from app.schemas.chat import ChatDoneChunk


class FakeLive:
    def __init__(self) -> None:
        self.progress = (100, 0, None)

    def read_progress(self) -> tuple[int, int, int | None]:
        return self.progress


def test_progress_is_reported_while_reading_and_only_when_it_moved() -> None:
    async def run() -> list[object]:
        results: asyncio.Queue[object] = asyncio.Queue()
        live = FakeLive()
        queue = ReadProgressQueue(results, live, total=900, interval=0.02)
        seen = []

        async def reader() -> None:
            await asyncio.sleep(0.15)
            live.progress = (100, 256, None)
            await asyncio.sleep(0.2)
            live.progress = (100, 256, 7)
            await asyncio.sleep(0.2)
            await results.put(42)
            live.progress = (100, 800, None)  # moved on, but tokens flow now: no more reports
            await asyncio.sleep(0.2)
            await results.put(None)

        task = asyncio.create_task(reader())
        while (item := await queue.get()) is not None:
            seen.append(item)
        await task
        return seen

    seen = asyncio.run(run())
    progress = [item["prompt_progress"] for item in seen if isinstance(item, dict)]
    assert progress == [
        {"cached": 100, "read": 256, "total": 900},
        {"cached": 100, "read": 256, "total": 900, "layer": 7},
    ]
    assert seen[-1] == 42 and all(item["done"] is False for item in seen if isinstance(item, dict))


def test_the_last_chunk_says_how_much_came_from_the_cache() -> None:
    chunk = ChatDoneChunk(prompt_eval_count=900, prompt_cached_count=100).to_ndjson_dict()
    assert chunk["prompt_cached_count"] == 100
