from __future__ import annotations

import threading
from contextlib import aclosing
from types import SimpleNamespace

import anyio

from codex_as_api import server


def test_closing_heartbeat_closes_its_source():
    closed = threading.Event()

    def chunks():
        try:
            yield "first"
            yield "second"
        finally:
            closed.set()

    async def exercise():
        async with aclosing(server._anthropic_heartbeat(chunks())) as stream:
            assert await anext(stream) == "first"
        assert closed.is_set()

    anyio.run(exercise)


def test_heartbeat_does_not_advance_or_cancel_a_waiting_source(monkeypatch):
    release = threading.Event()
    closed = threading.Event()
    calls = 0

    def chunks():
        nonlocal calls
        calls += 1
        try:
            if not release.wait(2):
                raise AssertionError("test did not release upstream")
            yield "done"
        finally:
            closed.set()

    async def exercise():
        async with aclosing(server._anthropic_heartbeat(chunks())) as stream:
            assert await anext(stream) == 'event: ping\ndata: {"type":"ping"}\n\n'
            assert calls == 1 and not closed.is_set()
            release.set()
            assert [chunk async for chunk in stream] == ["done"]
        assert closed.is_set() and calls == 1

    monkeypatch.setattr(server, "ANTHROPIC_HEARTBEAT_SECONDS", 0.01)
    try:
        anyio.run(exercise)
    finally:
        release.set()


def test_heartbeat_cancels_only_an_unfinished_request():
    cancelled = []
    control = SimpleNamespace(cancel=lambda: cancelled.append(True))

    async def exercise():
        async with aclosing(server._anthropic_heartbeat(iter(["first", "second"]), cancel=control)) as stream:
            assert await anext(stream) == "first"
        assert cancelled == [True]
        cancelled.clear()
        assert [chunk async for chunk in server._anthropic_heartbeat(iter(["done"]), cancel=control)] == ["done"]
        assert cancelled == []

    anyio.run(exercise)
