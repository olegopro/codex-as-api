from __future__ import annotations

import threading
import time
from contextlib import aclosing
from types import SimpleNamespace

import anyio
import pytest

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


def test_cancellation_bypasses_saturated_shared_workers():
    stream_release, prep_release = threading.Event(), threading.Event()
    cancellation_threads = []

    async def exercise():
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 2
        stream_waiting, prep_waiting, done = anyio.Event(), anyio.Event(), anyio.Event()
        scopes = []

        def cancel():
            cancellation_threads.append(threading.get_ident())
            stream_release.set()

        def chunks():
            yield "first"
            anyio.from_thread.run_sync(stream_waiting.set)
            assert stream_release.wait(3), "cancel did not release upstream"

        def prepare():
            anyio.from_thread.run_sync(prep_waiting.set)
            assert prep_release.wait(3), "test did not release preparation"

        async def consume():
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                async for _ in server._anthropic_heartbeat(chunks(), cancel=SimpleNamespace(cancel=cancel)):
                    pass
            done.set()

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(consume)
            await stream_waiting.wait()
            tasks.start_soon(anyio.to_thread.run_sync, prepare)
            await prep_waiting.wait()
            assert limiter.borrowed_tokens == 2
            scopes[0].cancel()
            try:
                with anyio.fail_after(1):
                    await done.wait()
                assert cancellation_threads and cancellation_threads[0] != threading.get_ident()
                assert not prep_release.is_set()
            finally:
                prep_release.set()
                stream_release.set()

    anyio.run(exercise)


@pytest.mark.parametrize("terminal", ["event: error\n\n", "event: message_stop\n\n"])
def test_idle_timer_never_loses_data_or_terminal_frames(monkeypatch, terminal):
    expected = [f"chunk-{i}" for i in range(100)] + [terminal]

    def chunks():
        for chunk in expected:
            time.sleep(0.001)
            yield chunk

    async def exercise():
        received = [chunk async for chunk in server._anthropic_heartbeat(chunks())]
        received = [chunk for chunk in received if not chunk.startswith("event: ping")]
        assert received == expected

    monkeypatch.setattr(server, "ANTHROPIC_HEARTBEAT_SECONDS", 0.001)
    anyio.run(exercise)
