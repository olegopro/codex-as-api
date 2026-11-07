"""Contention, cancellation and full-input replay; no model requests."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect
from websockets.sync.server import serve

from codex_as_api import siwc_transport
from codex_as_api.auth import ChatGPTOAuthUpstreamError
from codex_as_api.provider import _RequestCancellation
from codex_as_api.siwc_transport import SIWCWebSocketPool

from .test_diagnostics import diagnostic_log as diagnostic_log
from .test_diagnostics import records
from .test_siwc_provider import _completed, _text_item
from .test_siwc_websocket import FakeSocket

ROOT = [{"role": "user", "content": "root"}]
ANSWER = {"role": "assistant", "content": "answer"}
SIBLING = [*ROOT, ANSWER, {"role": "user", "content": "summary"}]
MAIN = [*ROOT, ANSWER, {"role": "user", "content": "continue"}]
OPTIONS = {"account_id": "account", "access_token": "synthetic", "timeout": 2}


def completed(tokens=64000):
    event = _completed([_text_item("answer")])
    event["response"]["usage"]["input_tokens"] = tokens
    return event


@pytest.fixture()
def contention(monkeypatch):
    sockets, observations = [], []
    entered = threading.Event()

    def emit(stage, **fields):
        observations.append({"stage": stage, **fields})
        if stage == "channel_wait_started":
            entered.set()

    def create(*, tokens=64000, **kwargs):
        def make_socket(*_args, **_kwargs):
            sock = FakeSocket([{"type": "response.created"}, completed(tokens)])
            sockets.append(sock)
            return sock

        monkeypatch.setattr(siwc_transport, "connect", make_socket)
        pool = SIWCWebSocketPool(warm_wait_seconds=1, **kwargs)
        list(pool.events({"input": ROOT}, **OPTIONS))
        sibling = pool.events({"input": SIBLING}, **OPTIONS)
        next(sibling)
        return pool, sibling

    monkeypatch.setattr(siwc_transport, "diagnostic_emit", emit)
    return create, sockets, observations, entered


def test_continuation_waits_for_sibling_and_replays_full_input(contention):
    create, sockets, events, entered = contention
    pool, sibling = create()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: list(pool.events({"input": MAIN}, **OPTIONS)))
            assert entered.wait(1) and not future.done()
            assert pool.active_requests == 2
            list(sibling)
            assert future.result(timeout=2)[-1]["type"] == "response.completed"
        assert len(sockets) == 1 and [p["input"] for p in sockets[0].sent] == [ROOT, SIBLING, MAIN]
        assert all("previous_response_id" not in p for p in sockets[0].sent)
        assert [e for e in events if e["stage"] == "channel_wait_finished"][-1]["wait_outcome"] == "acquired"
        assert pool.active_requests == 0 and pool._waiters == 0
    finally:
        sibling.close()
        pool.close()


def test_wait_times_out_once_and_opens_an_independent_channel(contention):
    create, sockets, events, entered = contention
    pool, sibling = create()
    pool._warm_wait_seconds = .03
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: list(pool.events({"input": MAIN}, **OPTIONS)))
            assert entered.wait(1)
            assert future.result(timeout=1)[-1]["type"] == "response.completed"
        assert len(sockets) == 2 and sockets[1].sent[0]["input"] == MAIN
        assert [e for e in events if e["stage"] == "channel_wait_finished"][-1]["wait_outcome"] == "timeout"
        assert pool._waiters == 0
        assert not sockets[0].closed and pool.active_requests == 1
    finally:
        sibling.close()
        pool.close()


@pytest.mark.parametrize("release_after,expected", [(15, "acquired"), (25, "timeout")])
def test_default_wait_covers_fifteen_seconds_but_not_twenty_five(monkeypatch, release_after, expected):
    now = [1000.0]
    events, waits = [], []
    monkeypatch.setattr(siwc_transport.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(siwc_transport, "diagnostic_emit", lambda stage, **fields: events.append(
        {"stage": stage, **fields},
    ))
    monkeypatch.setattr(siwc_transport, "connect", lambda *_a, **_k: FakeSocket([completed()]))
    pool = SIWCWebSocketPool()
    try:
        list(pool.events({"input": ROOT}, **OPTIONS))
        key, entry = next(iter(pool._connections.items()))
        entry.users = 1

        def wait(timeout):
            waits.append(timeout)
            now[0] += min(timeout, release_after)
            if release_after <= timeout:
                entry.users = 0

        monkeypatch.setattr(pool._condition, "wait", wait)
        selected = pool._reserve(key[:4], MAIN, "continuation", timeout=60)
        assert waits == [20]
        assert (selected is entry) == (expected == "acquired")
        assert events[-2]["wait_outcome"] == expected
        assert pool._waiters == 0
        entry.users = selected.users = 0
    finally:
        pool.close()


def test_default_pool_keeps_forty_minute_idle_channel_but_preserves_absolute_age(monkeypatch):
    now, sockets = [1000.0], []
    monkeypatch.setattr(siwc_transport.time, "monotonic", lambda: now[0])

    def connect(*_args, **_kwargs):
        socket = FakeSocket([completed()])
        sockets.append(socket)
        return socket

    monkeypatch.setattr(siwc_transport, "connect", connect)
    pool = SIWCWebSocketPool()
    try:
        list(pool.events({"input": ROOT}, **OPTIONS))
        now[0] += 40 * 60
        list(pool.events({"input": MAIN}, **OPTIONS))
        assert len(sockets) == 1
        now[0] = 1000 + 55 * 60 + 1
        tail = [*MAIN, ANSWER, {"role": "user", "content": "again"}]
        list(pool.events({"input": tail}, **OPTIONS))
        assert len(sockets) == 2 and sockets[0].closed
    finally:
        pool.close()


@pytest.mark.parametrize("stop", ["cancel", "close_pool"])
def test_wait_is_woken_by_cancel_or_pool_shutdown(contention, stop):
    create, sockets, events, entered = contention
    pool, sibling = create()
    cancel = _RequestCancellation()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: list(pool.events({"input": MAIN}, cancel=cancel, **OPTIONS)))
            assert entered.wait(1)
            if stop == "cancel":
                cancel.cancel()
            else:
                pool.close()
            with pytest.raises(OSError if stop == "cancel" else ChatGPTOAuthUpstreamError):
                future.result(timeout=.5)
        assert pool._waiters == 0 and len(sockets) == 1
        assert events[-1]["wait_outcome"] == ("cancelled" if stop == "cancel" else "error")
        if stop == "cancel":
            assert not sockets[0].closed
    finally:
        sibling.close()
        pool.close()


def test_refreshed_token_still_waits_for_the_warm_sibling(contention):
    create, sockets, events, entered = contention
    pool, sibling = create()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: list(pool.events({"input": MAIN},
                                                              **{**OPTIONS, "access_token": "refreshed"})))
            assert entered.wait(1) and not future.done()
            list(sibling)
            assert future.result(timeout=2)[-1]["type"] == "response.completed"
        assert len(sockets) == 1 and [p["input"] for p in sockets[0].sent] == [ROOT, SIBLING, MAIN]
        started = [e for e in events if e["stage"] == "upstream_start" and "reused" in e][-1]
        assert started["reused"] and started["token_rotated"] and started["reason"] == "reused"
    finally:
        sibling.close()
        pool.close()


@pytest.mark.parametrize("why", ["small", "capacity", "expired", "closed"])
def test_unhelpful_or_unavailable_wait_never_delays_a_request(contention, why):
    create, sockets, events, entered = contention
    pool, sibling = create(tokens=20 if why == "small" else 64000,
                           max_waiters=0 if why == "capacity" else 8)
    options = dict(OPTIONS)
    connection = next(iter(pool._connections.values()))
    if why == "expired":
        connection.opened_at -= 3301
    elif why == "closed":
        sockets[0].close()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(lambda: list(pool.events({"input": MAIN}, **options))).result(timeout=.5)
        assert not entered.is_set() and len(sockets) == 2 and pool._waiters == 0
        selected = [e for e in events if e["stage"] == "channel_selected"][-1]
        assert selected["wait_outcome"] == ("capacity" if why == "capacity" else "not_needed")
    finally:
        sibling.close()
        pool.close()


def test_pre_cancelled_request_does_not_reserve_or_open_a_channel(monkeypatch):
    monkeypatch.setattr(siwc_transport, "connect", lambda *_a, **_k: pytest.fail("unexpected connect"))
    pool, cancel = SIWCWebSocketPool(), _RequestCancellation()
    cancel.cancel()
    try:
        with pytest.raises(OSError):
            list(pool.events({"input": ROOT}, cancel=cancel, **OPTIONS))
        assert not pool._connections and pool.active_requests == 0
    finally:
        pool.close()


def test_waiter_capacity_preserves_progress_for_other_requests(contention):
    create, sockets, events, entered = contention
    pool, sibling = create(max_waiters=1)
    other = [*ROOT, ANSWER, {"role": "user", "content": "other continuation"}]
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            waiting = executor.submit(lambda: list(pool.events({"input": MAIN}, **OPTIONS)))
            assert entered.wait(1)
            assert executor.submit(lambda: list(pool.events({"input": other}, **OPTIONS))).result(timeout=.5)
            assert any(e.get("wait_outcome") == "capacity" for e in events)
            assert pool._waiters == 1 and len(sockets) == 2
            list(sibling)
            waiting.result(timeout=2)
        assert pool.active_requests == 0 and pool._waiters == 0
    finally:
        sibling.close()
        pool.close()


def test_failed_sibling_loses_affinity_without_leaking_a_waiter(contention):
    create, sockets, events, entered = contention
    pool, sibling = create()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: list(pool.events({"input": MAIN}, **OPTIONS)))
            assert entered.wait(1)
            sibling.close()
            future.result(timeout=2)
        assert len(sockets) == 2 and sockets[0].closed and pool._waiters == 0
        assert [e for e in events if e["stage"] == "channel_wait_finished"][-1]["wait_outcome"] == "affinity_lost"
    finally:
        sibling.close()
        pool.close()


def test_real_loopback_keeps_siblings_on_one_exclusive_socket(monkeypatch, diagnostic_log):
    requests, connections = [], []
    waiting = threading.Event()
    original_emit = siwc_transport.diagnostic_emit

    def emit(stage, **fields):
        original_emit(stage, **fields)
        if stage == "channel_wait_started":
            waiting.set()

    def handler(socket):
        connections.append(socket)
        try:
            while True:
                requests.append(json.loads(socket.recv(timeout=5)))
                socket.send('{"type":"response.created"}')
                socket.send(json.dumps(completed()))
        except ConnectionClosed:
            pass

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        monkeypatch.setattr(siwc_transport, "connect", lambda _url, **kw:
                            connect(f"ws://127.0.0.1:{port}", **{**kw, "proxy": None}))
        monkeypatch.setattr(siwc_transport, "diagnostic_emit", emit)
        pool = SIWCWebSocketPool(warm_wait_seconds=1)
        sibling = None
        try:
            list(pool.events({"input": ROOT}, **OPTIONS))
            sibling = pool.events({"input": SIBLING}, **OPTIONS)
            next(sibling)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(lambda: list(pool.events({"input": MAIN}, **OPTIONS)))
                assert waiting.wait(1)
                list(sibling)
                future.result(timeout=2)
            assert len(connections) == 1 and [p["input"] for p in requests] == [ROOT, SIBLING, MAIN]
            assert all("previous_response_id" not in p for p in requests)
            assert any(e.get("wait_outcome") == "acquired" for e in records(diagnostic_log))
        finally:
            if sibling is not None:
                sibling.close()
            pool.close()
        server.shutdown()
        thread.join(timeout=2)
