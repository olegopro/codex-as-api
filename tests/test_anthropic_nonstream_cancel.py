from __future__ import annotations

import threading

import anyio
import pytest
from starlette.requests import Request

from codex_as_api import server
from codex_as_api.auth import ChatGPTOAuthProtocolError


def test_nonstream_disconnect_cancels_only_its_worker():
    released = threading.Event()
    cancelled = []

    async def exercise():
        entered = anyio.Event()

        def call():
            anyio.from_thread.run_sync(entered.set)
            assert released.wait(1), "worker was not cancelled after disconnect"
            return "finished"

        def cancel():
            cancelled.append(True)
            released.set()

        async def receive():
            await entered.wait()
            return {"type": "http.disconnect"}

        request = Request({"type": "http", "headers": []}, receive=receive)
        with anyio.fail_after(2):
            assert await server._anthropic_nonstream_call(request, call, cancel) == "finished"

    try:
        anyio.run(exercise)
    finally:
        released.set()
    assert cancelled == [True]


def test_nonstream_completed_worker_keeps_warm_channel():
    cancelled = []

    async def exercise():
        async def receive():
            await anyio.sleep_forever()

        request = Request({"type": "http", "headers": []}, receive=receive)
        result = await server._anthropic_nonstream_call(request, lambda: "completed", lambda: cancelled.append(True))
        assert result == "completed"

    anyio.run(exercise)
    assert cancelled == []


def test_nonstream_worker_error_keeps_original_error_type():
    async def exercise():
        async def receive():
            await anyio.sleep_forever()

        def fail():
            raise ChatGPTOAuthProtocolError("synthetic error")

        request = Request({"type": "http", "headers": []}, receive=receive)
        with pytest.raises(ChatGPTOAuthProtocolError):
            await server._anthropic_nonstream_call(request, fail, lambda: None)

    anyio.run(exercise)
