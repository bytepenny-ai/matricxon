"""A reply whose client left (the stream closed before its end) stops its work on the worker: a
long prompt kept being read for minutes for a chat already deleted. Every reply gets an id to stop
it by."""

import asyncio
from types import SimpleNamespace

from app.routers.chat_router import ChatRequestHandler
from app.schemas.chat import ChatRequest


class FakeWorker:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    def cancel_job(self, request_id: str) -> bool:
        self.cancelled.append(request_id)
        return True


class FakeLive:
    def queued(self, _count: int) -> None:
        pass

    def finish(self) -> None:
        pass


def _handler(chunks: int) -> ChatRequestHandler:
    handler = ChatRequestHandler.__new__(ChatRequestHandler)

    async def tokens(*_args):
        for i in range(chunks):
            yield {"n": i}

    handler._stream_tokens = tokens
    return handler


def _request() -> ChatRequest:
    return ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}], request_id="r1")


def test_a_client_leaving_mid_stream_stops_the_reply() -> None:
    async def run() -> list[str]:
        worker = FakeWorker()
        handle = SimpleNamespace(worker=worker)
        stream = _handler(5)._stream(handle, None, 10, 0.0, _request(), FakeLive())
        await stream.__anext__()
        await stream.aclose()  # the client disconnected
        return worker.cancelled

    assert asyncio.run(run()) == ["r1"]


def test_a_finished_stream_stops_nothing() -> None:
    async def run() -> list[str]:
        worker = FakeWorker()
        handle = SimpleNamespace(worker=worker)
        async for _chunk in _handler(3)._stream(handle, None, 10, 0.0, _request(), FakeLive()):
            pass
        return worker.cancelled

    assert asyncio.run(run()) == []


def test_every_reply_gets_an_id_to_stop_it_by() -> None:
    request = ChatRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    assert ChatRequestHandler._prepared(request).request_id.startswith("req-")
    assert ChatRequestHandler._prepared(_request()).request_id == "r1"
