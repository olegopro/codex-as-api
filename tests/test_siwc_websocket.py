from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response
from websockets.protocol import State

from codex_as_api import provider as provider_module
from codex_as_api import siwc_transport
from codex_as_api.auth import ChatGPTOAuthProtocolError, ChatGPTOAuthUpstreamError
from codex_as_api.messages import AssistantResponse, Message, MessageRole, ToolCall, ToolSchema, Usage
from codex_as_api.provider import ChatGPTOAuthProvider
from codex_as_api.siwc_transport import SIWCWebSocketPool

from . import test_siwc_provider

siwc_provider = test_siwc_provider.siwc_provider


class FakeSocket:
    def __init__(self, events):
        self.events = events
        self.pending = deque()
        self.sent = []
        self.closed = False
        self.state = State.OPEN

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()

    def close(self):
        self.closed = True
        self.state = State.CLOSED

    def send(self, value):
        assert not self.closed
        self.sent.append(json.loads(value))
        self.pending.extend(self.events)

    def recv(self, *, timeout):
        assert timeout > 0
        value = self.pending.popleft()
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, str) else json.dumps(value)


@pytest.fixture()
def websocket_provider(siwc_provider, monkeypatch):
    instances = []
    calls = []
    providers = []

    def create(*, events=None, handshake=None):
        http_provider, requests = siwc_provider()
        provider = ChatGPTOAuthProvider(backend="siwc", http_client=http_provider._http_client)
        providers.append(provider)

        def connect(url, **kwargs):
            calls.append((url, kwargs))
            if handshake is not None:
                handshake(len(calls))
            sock = FakeSocket([test_siwc_provider._completed()] if events is None else events)
            instances.append(sock)
            return sock

        monkeypatch.setattr(siwc_transport, "connect", connect)
        return provider, instances, calls, requests

    yield create
    for provider in providers:
        provider.close()


def test_siwc_default_websocket_reuses_session_and_sends_full_stateless_input(websocket_provider, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8110")
    monkeypatch.delenv("https_proxy", raising=False)
    provider, sockets, calls, requests = websocket_provider()
    for _ in range(2):
        result = provider.chat(test_siwc_provider._messages(), prompt_cache_key="session-a")
        assert result.content == "Ответ" and result.usage.cached_tokens == 12
    assert len(calls) == 1 and len(sockets[0].sent) == 2
    url, options = calls[0]
    assert url == "wss://api.openai.com/v1/responses"
    assert options["additional_headers"] == {"Authorization": "Bearer synthetic-access"}
    assert options["proxy"] == "http://127.0.0.1:8110"
    assert options["user_agent_header"] is None
    assert options["max_size"] == 64 * 1024 * 1024
    assert options["ping_timeout"] == 60.0
    assert "max_queue" not in options and "ping_interval" not in options
    for payload in sockets[0].sent:
        assert payload["type"] == "response.create" and payload["store"] is False
        assert payload["input"][-1]["role"] == "user"
        assert "stream" not in payload and "previous_response_id" not in payload
    assert [request.method for request in requests] == ["GET"]


def test_siwc_websocket_sessions_do_not_share_a_channel(websocket_provider):
    provider, sockets, calls, _ = websocket_provider()
    for key in ("one", "two", "one"):
        provider.chat(test_siwc_provider._messages(), prompt_cache_key=key)
    assert len(calls) == 2
    assert [len(sock.sent) for sock in sockets] == [2, 1]


def test_claude_code_late_system_notes_keep_the_same_channel_and_history_prefix(websocket_provider):
    provider, sockets, calls, _ = websocket_provider(events=[
        test_siwc_provider._completed([test_siwc_provider._text_item("Earlier answer")]),
    ])
    first = [
        Message(role=MessageRole.SYSTEM, content="Stable system prompt"),
        Message(role=MessageRole.USER, content="Read the fixture"),
        Message(role=MessageRole.SYSTEM, content="Environment and initial budget"),
    ]
    second = [*first,
        Message(role=MessageRole.ASSISTANT, content="Earlier answer"),
        Message(role=MessageRole.USER, content="Tool result"),
        Message(role=MessageRole.SYSTEM, content="<total_tokens>14999985 tokens left</total_tokens>"),
    ]
    for messages in (first, second):
        provider.chat(messages, prompt_cache_key="same-claude-session")
    assert len(calls) == 1
    initial, continued = sockets[0].sent
    assert initial["instructions"] == continued["instructions"] == "Stable system prompt"
    assert continued["input"][:len(initial["input"])] == initial["input"]
    assert initial["input"][1]["role"] == "developer"
    assert initial["input"][1]["content"][0]["text"] == first[2].content
    assert continued["input"][-1]["role"] == "developer"
    assert continued["input"][-1]["content"][0]["text"] == second[-1].content


def test_completed_auxiliary_fork_preserves_main_anchor(monkeypatch):
    sockets = []

    def connect(*_args, **_kwargs):
        socket = FakeSocket([test_siwc_provider._completed([test_siwc_provider._text_item("answer")])])
        sockets.append(socket)
        return socket

    monkeypatch.setattr(siwc_transport, "connect", connect)
    pool = SIWCWebSocketPool()
    options = {"account_id": "account", "access_token": "synthetic", "timeout": 1}
    main = [{"role": "user", "content": "root"}]
    replay = {"role": "assistant", "content": "answer"}
    fork = [*main, replay, {"role": "user", "content": "summary"}]
    continued = [*main, replay, {"role": "user", "content": "continue"}]
    try:
        for history in (main, fork, continued):
            list(pool.events({"input": history}, **options))
        assert len(sockets) == 1
        assert [request["input"] for request in sockets[0].sent] == [main, fork, continued]
    finally:
        pool.close()


def test_multiple_forks_remain_replayable_and_anchors_are_bounded(monkeypatch):
    sockets = []

    def connect(*_args, **_kwargs):
        socket = FakeSocket([test_siwc_provider._completed([test_siwc_provider._text_item("answer")])])
        sockets.append(socket)
        return socket

    monkeypatch.setattr(siwc_transport, "connect", connect)
    pool = SIWCWebSocketPool()
    options = {"account_id": "account", "access_token": "synthetic", "timeout": 1}
    root = [{"role": "user", "content": "root"}]
    answer = {"role": "assistant", "content": "answer"}
    try:
        list(pool.events({"input": root}, **options))
        for label in ("summary", "suggestion", "summary"):
            list(pool.events({"input": [*root, answer, {"role": "user", "content": label}]}, **options))
        list(pool.events({"input": [*root, answer, {"role": "user", "content": "continue"}]}, **options))
        assert len(sockets) == 1
        connection = next(iter(pool._connections.values()))
        assert len(connection.anchors) == 4
        history = [*root, answer, {"role": "user", "content": "continue"}]
        for index in range(20):
            history = [*history, answer, {"role": "user", "content": str(index)}]
            list(pool.events({"input": history}, **options))
        assert len(connection.anchors) == 16
        connection.close()
        assert not connection.anchors
    finally:
        pool.close()


def test_longest_valid_anchor_wins_without_waiting(monkeypatch):
    sockets = []

    def connect(*_args, **_kwargs):
        socket = FakeSocket([{"type": "response.created"},
                             test_siwc_provider._completed([test_siwc_provider._text_item("answer")])])
        sockets.append(socket)
        return socket

    monkeypatch.setattr(siwc_transport, "connect", connect)
    pool = SIWCWebSocketPool()
    options = {"account_id": "account", "access_token": "synthetic", "timeout": 1}
    root = [{"role": "user", "content": "root"}]
    answer = {"role": "assistant", "content": "answer"}
    longer = [*root, answer, {"role": "user", "content": "branch"}]
    streams = [pool.events({"input": history}, **options) for history in (root, longer)]
    try:
        for stream in streams:
            next(stream)
        for stream in streams:
            list(stream)
        continued = [*longer, answer, {"role": "user", "content": "continue"}]
        list(pool.events({"input": continued}, **options))
        assert len(sockets) == 2
        assert [len(socket.sent) for socket in sockets] == [1, 2]
        assert sockets[1].sent[-1]["input"] == continued
    finally:
        for stream in streams:
            stream.close()
        pool.close()


@pytest.mark.parametrize("failure", ["error", "cancel"])
def test_failed_or_cancelled_fork_clears_socket_anchors(monkeypatch, failure):
    socket = FakeSocket([test_siwc_provider._completed([test_siwc_provider._text_item("answer")])])
    monkeypatch.setattr(siwc_transport, "connect", lambda *args, **kwargs: socket)
    pool = SIWCWebSocketPool()
    options = {"account_id": "account", "access_token": "synthetic", "timeout": 1}
    root = [{"role": "user", "content": "root"}]
    try:
        list(pool.events({"input": root}, **options))
        connection = next(iter(pool._connections.values()))
        assert len(connection.anchors) == 1
        socket.events = ([{"type": "error"}] if failure == "error"
                         else [{"type": "response.created"}, {"type": "response.completed"}])
        fork = pool.events({"input": [*root, {"role": "assistant", "content": "answer"},
                                      {"role": "user", "content": "summary"}]}, **options)
        if failure == "error":
            list(fork)
        else:
            next(fork)
            fork.close()
        assert socket.closed and not connection.anchors and connection.users == 0
    finally:
        pool.close()


def test_background_model_does_not_replace_the_main_models_channel(websocket_provider):
    provider, sockets, calls, _ = websocket_provider()
    for model in ("synthetic-first", "synthetic-second", "synthetic-first"):
        provider.chat(test_siwc_provider._messages(), model=model, prompt_cache_key="same-session")
    assert len(calls) == 2
    assert [len(sock.sent) for sock in sockets] == [2, 1]


def test_sibling_histories_keep_separate_warm_channels(monkeypatch):
    sockets = []

    def connect(*_args, **_kwargs):
        sock = FakeSocket([test_siwc_provider._completed([test_siwc_provider._text_item("Earlier answer")])])
        sockets.append(sock)
        return sock

    monkeypatch.setattr(siwc_transport, "connect", connect)
    pool = SIWCWebSocketPool()
    try:
        for task in ("main", "child", "main", "child"):
            payload = {"prompt_cache_key": "same-parent", "model": "same-model",
                       "instructions": "Stable instructions", "input": [{"role": "user", "content": task}]}
            if task == "main" and sockets:
                payload["input"].append({"role": "assistant", "content": "Earlier answer"})
            list(pool.events(payload, account_id="account", access_token="synthetic", timeout=1))
        assert len(sockets) == 2
        assert [len(sock.sent) for sock in sockets] == [2, 2]
    finally:
        pool.close()


@pytest.mark.parametrize("identical_prompts", [False, True])
@pytest.mark.parametrize("with_tools", [False, True])
def test_diverged_full_histories_reuse_their_own_warm_sockets(monkeypatch, identical_prompts, with_tools):
    sockets, outputs = [], []

    def connect(*_args, **_kwargs):
        label = str(len(sockets))
        output = [test_siwc_provider._text_item("answer " + label)]
        if with_tools:
            output.insert(0, {"type": "reasoning", "id": "rs_" + label,
                              "encrypted_content": "cipher_" + label, "summary": [], "status": "completed"})
            output.append({"type": "function_call", "id": "fc_" + label, "call_id": "call_" + label,
                           "name": "codex_as_api.read", "arguments": '{"b": 2, "a": 1}', "status": "completed"})
        outputs.append(output)
        events = [{"type": "response.created"}]
        events.extend({"type": "response.output_item.done", "item": item} for item in output)
        events.append(test_siwc_provider._completed([]))
        socket = FakeSocket(events)
        sockets.append(socket)
        return socket

    monkeypatch.setattr(siwc_transport, "connect", connect)
    pool = SIWCWebSocketPool(limit=2)
    options = {"account_id": "account", "access_token": "synthetic", "timeout": 1}
    base = {"prompt_cache_key": "same-session", "model": "same-model", "instructions": "stable"}
    histories = [
        [{"role": "user", "content": "root"}, {"role": "assistant", "content": "shared"},
         {"role": "user", "content": "same" if identical_prompts else label}]
        for label in ("A", "B")
    ]
    try:
        streams = [pool.events({**base, "input": history}, **options) for history in histories]
        for stream in streams:
            next(stream)
        assert pool.active_requests == 2
        for stream in streams:
            list(stream)
        for _ in range(2):
            for index, history in enumerate(histories):
                replay = []
                for item in outputs[index]:
                    copied = {key: value for key, value in item.items() if key != "status"}
                    if item["type"] == "function_call":
                        copied.pop("id")
                        copied.update(name="read", namespace="codex_as_api", arguments='{"a":1,"b":2}')
                    replay.append(copied)
                history.extend(replay)
                history.append({"role": "user", "content": "continue"})
                list(pool.events({**base, "input": history}, **options))
                assert sockets[index].sent[-1]["input"] == history
        assert len(sockets) == 2
        assert [len(socket.sent) for socket in sockets] == [3, 3]
        for connection in pool._connections.values():
            assert all(isinstance(part, bytes) and len(part) == 32 for part in connection.history)
            assert isinstance(connection.output_key, bytes)
    finally:
        pool.close()


def test_affinity_survives_anthropic_reasoning_and_tool_replay(websocket_provider, monkeypatch):
    provider, _, _, _ = websocket_provider()
    sockets, outputs = [], []
    original_name = "synthetic.read-file"
    wire_name = provider_module._siwc_wire_name(original_name)
    tools = [ToolSchema(name=original_name, description="", parameters={})]

    def connect(*_args, **_kwargs):
        label = str(len(sockets))
        reasoning = {"type": "reasoning", "id": "rs_" + label, "encrypted_content": "cipher_" + label, "summary": []}
        call = {"type": "function_call", "name": "codex_as_api." + wire_name,
                "call_id": "call_" + label, "arguments": '{"path": "fixture"}', "status": "completed"}
        output = [reasoning, test_siwc_provider._text_item("answer " + label), call]
        outputs.append(output)
        socket = FakeSocket([*[{"type": "response.output_item.done", "item": item} for item in output],
                             test_siwc_provider._completed([])])
        sockets.append(socket)
        return socket

    monkeypatch.setattr(siwc_transport, "connect", connect)
    streams = [provider.chat_stream(test_siwc_provider._messages(), tools=tools, prompt_cache_key="same-session")
               for _ in range(2)]
    for stream in streams:
        assert next(stream)["type"] == "reasoning_item"
    for stream in streams:
        list(stream)
    for index, output in enumerate(outputs[:]):
        call = output[-1]
        response = AssistantResponse(
            content="answer " + str(index), finish_reason="tool_calls", reasoning_items=(output[0],),
            usage=Usage(prompt_tokens=20, completion_tokens=3, total_tokens=23),
            tool_calls=(ToolCall(id=call["call_id"], name=original_name, arguments=call["arguments"]),),
        )
        converted = test_siwc_provider.internal_response_to_anthropic(
            response, "claude-synthetic", "msg_fixture", backend_model="synthetic-first",
        )
        messages, replay_tools, *_ = test_siwc_provider.anthropic_request_to_internal(
            model="claude-synthetic", system="Правила", backend_model="synthetic-first",
            messages=[{"role": "user", "content": "Вопрос"},
                      {"role": "assistant", "content": converted["content"]},
                      {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["call_id"],
                                                    "content": "done"}]}],
            tools=[{"name": original_name, "input_schema": {}}],
        )
        provider.chat(messages, tools=replay_tools, prompt_cache_key="same-session")
        assert len(sockets) == 2 and len(sockets[index].sent) == 2


def test_rewritten_output_does_not_reuse_an_incompatible_warm_channel(monkeypatch):
    sockets = []

    def connect(*_args, **_kwargs):
        socket = FakeSocket([test_siwc_provider._completed([test_siwc_provider._text_item("original")])])
        sockets.append(socket)
        return socket

    monkeypatch.setattr(siwc_transport, "connect", connect)
    pool = SIWCWebSocketPool()
    options = {"account_id": "account", "access_token": "synthetic", "timeout": 1}
    original = {"input": [{"role": "user", "content": "root"}]}
    try:
        list(pool.events(original, **options))
        changed = {"input": [*original["input"], {"role": "assistant", "content": "edited"},
                             {"role": "user", "content": "continue"}]}
        list(pool.events(changed, **options))
        assert len(sockets) == 2
        list(pool.events(original, **options))
        assert len(sockets) == 2 and len(sockets[0].sent) == 2
    finally:
        pool.close()


@pytest.mark.parametrize("identical_prompts", [False, True])
@pytest.mark.parametrize("limit,concurrency", [(100, 2), (100, 20), (100, 32), (100, 100), (None, 128)])
def test_same_parent_model_requests_reach_upstream_concurrently(monkeypatch, identical_prompts, limit, concurrency):
    sockets = []
    observed_active = []
    pool = SIWCWebSocketPool() if limit is None else SIWCWebSocketPool(limit=limit)
    assert pool.max_connections == limit

    def at_peak():
        observed_active.append(pool.active_requests)
        if concurrency == pool.max_connections:
            with pytest.raises(ChatGPTOAuthUpstreamError) as caught:
                list(pool.events({"prompt_cache_key": "overflow"}, account_id="account",
                                 access_token="synthetic", timeout=1))
            assert caught.value.status == 429

    rendezvous = threading.Barrier(concurrency, action=at_peak)

    class ConcurrentSocket(FakeSocket):
        def recv(self, *, timeout):
            # Every independent generation must enter recv before any finishes.
            rendezvous.wait(timeout=5)
            return super().recv(timeout=timeout)

    def connect(*_args, **_kwargs):
        sock = ConcurrentSocket([{"type": "response.completed"}])
        sockets.append(sock)
        return sock

    def request(index):
        payload = {"prompt_cache_key": "parent", "model": "same-model",
                   "input": [{"role": "user", "content": "same" if identical_prompts else str(index)}]}
        return list(pool.events(payload, account_id="account", access_token="synthetic", timeout=3))

    monkeypatch.setattr(siwc_transport, "connect", connect)
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(request, index) for index in range(concurrency)]
            assert all(future.result(timeout=10)[-1]["type"] == "response.completed" for future in futures)
        assert len(sockets) == concurrency and pool.active_requests == 0 and observed_active == [concurrency]
    finally:
        pool.close()


def test_saturated_same_branch_fails_explicitly_instead_of_waiting(monkeypatch):
    monkeypatch.setattr(siwc_transport, "connect", lambda *_a, **_k:
                        FakeSocket([{"type": "response.created"}, {"type": "response.completed"}]))
    pool = SIWCWebSocketPool(limit=1)

    def events():
        return pool.events({"prompt_cache_key": "same", "model": "same"}, account_id="account",
                           access_token="synthetic", timeout=1)

    first = events()
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            next(first)
            future = executor.submit(lambda: list(events()))
            with pytest.raises(ChatGPTOAuthUpstreamError) as caught:
                future.result(timeout=2)
            assert caught.value.status == 429
        finally:
            first.close()
            pool.close()


@pytest.mark.parametrize("expire", [False, True])
def test_many_idle_channels_close_concurrently(monkeypatch, expire):
    rendezvous = threading.Barrier(8)
    sockets = []

    class ClosingSocket(FakeSocket):
        def close(self):
            if not self.closed and self in sockets[:8]:
                rendezvous.wait(timeout=3)
            super().close()

    def connect(*_args, **_kwargs):
        socket = ClosingSocket([{"type": "response.completed"}])
        sockets.append(socket)
        return socket

    monkeypatch.setattr(siwc_transport, "connect", connect)
    monkeypatch.setattr(siwc_transport.time, "monotonic", lambda: 100)
    pool = SIWCWebSocketPool(idle_timeout=10)
    try:
        for index in range(8):
            list(pool.events({"prompt_cache_key": str(index)}, account_id="account",
                             access_token="synthetic", timeout=1))
        if expire:
            monkeypatch.setattr(siwc_transport.time, "monotonic", lambda: 111)
            list(pool.events({"prompt_cache_key": "new"}, account_id="account",
                             access_token="synthetic", timeout=1))
        pool.close()
        assert all(socket.closed for socket in sockets[:8])
        assert pool.active_requests == 0
    finally:
        pool.close()


def test_expired_cleanup_does_not_delay_its_request_or_other_leases(monkeypatch):
    sockets = []
    closing, release = threading.Event(), threading.Event()
    close_lock = threading.Lock()
    active_closes = maximum_closes = 0

    class ClosingSocket(FakeSocket):
        def close(self):
            nonlocal active_closes, maximum_closes
            if not self.closed and self in sockets[:100]:
                with close_lock:
                    active_closes += 1
                    maximum_closes = max(maximum_closes, active_closes)
                    if active_closes == 8:
                        closing.set()
                try:
                    assert release.wait(3), "test did not release idle cleanup"
                finally:
                    with close_lock:
                        active_closes -= 1
            super().close()

    def connect(*_args, **_kwargs):
        socket = ClosingSocket([{"type": "response.completed"}])
        sockets.append(socket)
        return socket

    def events(session):
        return list(pool.events({"prompt_cache_key": session}, account_id="account",
                                access_token="synthetic", timeout=1))

    monkeypatch.setattr(siwc_transport, "connect", connect)
    monkeypatch.setattr(siwc_transport.time, "monotonic", lambda: 100)
    pool = SIWCWebSocketPool(idle_timeout=10)
    try:
        for index in range(100):
            events(str(index))
        monkeypatch.setattr(siwc_transport.time, "monotonic", lambda: 111)
        with ThreadPoolExecutor(max_workers=2) as workers:
            triggering_request = workers.submit(events, "trigger")
            try:
                assert closing.wait(1)
                assert triggering_request.result(timeout=0.5)
                assert workers.submit(events, "unrelated").result(timeout=0.5)
                assert pool.active_requests == 0
                assert not any(socket.closed for socket in sockets[:100])
            finally:
                release.set()
        pool.close()
        assert maximum_closes == 8
        assert all(socket.closed for socket in sockets[:100])
        assert pool.active_requests == 0
    finally:
        release.set()
        pool.close()


def test_overlapping_cleanup_is_bounded_joined_and_keeps_active_channels(monkeypatch):
    sockets, retired, closer_threads = [], [], set()
    closing, release, stopping = threading.Event(), threading.Event(), threading.Event()
    close_lock = threading.Lock()
    active_closes = maximum_closes = 0
    attempts = {}

    class ClosingSocket(FakeSocket):
        def close(self):
            nonlocal active_closes, maximum_closes
            if self in retired:
                with close_lock:
                    attempts[self] = attempts.get(self, 0) + 1
                    closer_threads.add(threading.current_thread())
                    active_closes += 1
                    maximum_closes = max(maximum_closes, active_closes)
                    if active_closes == 8:
                        closing.set()
                try:
                    assert release.wait(3), "Очистка не была освобождена тестом"
                finally:
                    with close_lock:
                        active_closes -= 1
            super().close()

    def connect(*_args, **_kwargs):
        socket = ClosingSocket([{"type": "response.created"}, {"type": "response.completed"}])
        sockets.append(socket)
        return socket

    def events(session):
        return pool.events({"prompt_cache_key": session}, account_id="account",
                           access_token="synthetic", timeout=1)

    monkeypatch.setattr(siwc_transport, "connect", connect)
    monkeypatch.setattr(siwc_transport.time, "monotonic", lambda: 100)
    pool = SIWCWebSocketPool(idle_timeout=10)
    active = events("active")
    try:
        next(active)
        active_socket = sockets[0]
        for index in range(8):
            list(events(str(index)))
        retired.extend(sockets[1:])
        monkeypatch.setattr(siwc_transport.time, "monotonic", lambda: 111)
        for index in range(8, 16):
            list(events(str(index)))
        assert closing.wait(1)
        retired.extend(sockets[9:])
        monkeypatch.setattr(siwc_transport.time, "monotonic", lambda: 122)
        with ThreadPoolExecutor(max_workers=1) as worker:
            assert worker.submit(lambda: list(events("trigger"))).result(timeout=0.5)
            assert pool.active_requests == 1 and not active_socket.closed
            assert list(active)[-1]["type"] == "response.completed"

            def stop():
                stopping.set()
                pool.close()

            shutdown = worker.submit(stop)
            try:
                assert stopping.wait(1) and not shutdown.done()
            finally:
                release.set()
            shutdown.result(timeout=3)
        assert maximum_closes == 8 and len(attempts) == 16
        assert set(attempts.values()) == {1}
        assert all(socket.closed for socket in sockets)
        for thread in closer_threads:
            thread.join(timeout=1)
            assert not thread.is_alive()
        with pytest.raises(ChatGPTOAuthUpstreamError) as caught:
            list(events("after-close"))
        assert caught.value.status == 503
    finally:
        release.set()
        active.close()
        pool.close()


@pytest.mark.parametrize("failure", ["invalid-json", OSError("synthetic disconnect"),
                                    {"type": "error", "message": "synthetic-access"}])
def test_siwc_websocket_failure_closes_channel_without_replaying_request(websocket_provider, failure):
    provider, sockets, calls, _ = websocket_provider(events=[failure])
    with pytest.raises((ChatGPTOAuthProtocolError, ChatGPTOAuthUpstreamError)) as caught:
        list(provider.chat_stream(test_siwc_provider._messages()))
    assert "synthetic-access" not in str(caught.value)
    assert len(calls) == 1 and len(sockets[0].sent) == 1 and sockets[0].closed


def test_siwc_websocket_refreshes_only_rejected_handshake(websocket_provider, monkeypatch):
    def handshake(attempt):
        if attempt == 1:
            raise InvalidStatus(Response(401, "Unauthorized", Headers(), body=b"rejected"))

    provider, sockets, calls, _ = websocket_provider(handshake=handshake)
    refreshed = []
    monkeypatch.setattr(provider, "_refresh_after_unauthorized", lambda token: refreshed.append(token))
    assert provider.chat(test_siwc_provider._messages()).content == "Ответ"
    assert len(refreshed) == 1 and len(calls) == 2 and len(sockets[0].sent) == 1


def _rotate_token(monkeypatch):
    replacement = test_siwc_provider._token()
    replacement.access_token = "synthetic-new-access"
    monkeypatch.setattr(provider_module, "token_for_request", lambda *_args, **_kwargs: replacement)


def test_siwc_websocket_rotated_token_keeps_open_connection(websocket_provider, monkeypatch):
    provider, sockets, calls, _ = websocket_provider()
    provider.chat(test_siwc_provider._messages())
    _rotate_token(monkeypatch)
    provider.chat(test_siwc_provider._messages())
    assert len(calls) == 1 and not sockets[0].closed and len(sockets[0].sent) == 2
    sockets[0].close()
    provider.chat(test_siwc_provider._messages())
    assert len(calls) == 2
    assert calls[-1][1]["additional_headers"]["Authorization"] == "Bearer synthetic-new-access"


@pytest.mark.parametrize("rejection", [{"type": "error", "error": {"message": "synthetic-access"}},
                                       ConnectionClosedError(Close(1008, ""), None)])
def test_rotated_channel_rejection_is_replayed_once_on_a_new_socket(websocket_provider, monkeypatch, rejection):
    provider, sockets, calls, _ = websocket_provider()
    provider.chat(test_siwc_provider._messages())
    sockets[0].events = [rejection]
    _rotate_token(monkeypatch)
    assert provider.chat(test_siwc_provider._messages()).content == "Ответ"
    assert len(calls) == 2 and sockets[0].closed and len(sockets[1].sent) == 1
    assert calls[-1][1]["additional_headers"]["Authorization"] == "Bearer synthetic-new-access"


def test_refused_connect_is_retried_within_the_request(websocket_provider, monkeypatch):
    delays = []
    monkeypatch.setattr(siwc_transport.time, "sleep", delays.append)

    def handshake(attempt):
        if attempt < 3:
            raise ConnectionRefusedError("synthetic proxy restart")

    provider, sockets, calls, _ = websocket_provider(handshake=handshake)
    assert provider.chat(test_siwc_provider._messages()).content == "Ответ"
    assert len(calls) == 3 and delays == [0.5, 1.0] and len(sockets) == 1 and len(sockets[0].sent) == 1


def test_persistent_connect_failure_returns_after_the_retry_window(websocket_provider, monkeypatch):
    clock, events = [1000.0], []
    monkeypatch.setattr(siwc_transport.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(siwc_transport.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(siwc_transport, "diagnostic_emit",
                        lambda stage, **fields: events.append({"stage": stage, **fields}))

    def handshake(_attempt):
        raise ConnectionRefusedError("synthetic proxy down")

    provider, sockets, calls, _ = websocket_provider(handshake=handshake)
    with pytest.raises(ChatGPTOAuthUpstreamError) as caught:
        provider.chat(test_siwc_provider._messages())
    assert caught.value.status == 502 and not sockets and len(calls) == 8 and clock[0] == 1019.5
    assert [e["reason"] for e in events if e["stage"] == "upstream_error"] == ["connect_failed"]


def test_same_token_channel_rejection_is_not_replayed(websocket_provider):
    provider, sockets, calls, _ = websocket_provider()
    provider.chat(test_siwc_provider._messages())
    sockets[0].events = [{"type": "error", "error": {"message": "synthetic-access"}}]
    with pytest.raises((ChatGPTOAuthProtocolError, ChatGPTOAuthUpstreamError)):
        provider.chat(test_siwc_provider._messages())
    assert len(calls) == 1 and sockets[0].closed and len(sockets[0].sent) == 2


def test_closed_idle_socket_is_reopened_before_sending_a_new_request(websocket_provider):
    provider, sockets, calls, _ = websocket_provider()
    provider.chat(test_siwc_provider._messages())
    sockets[0].close()
    assert provider.chat(test_siwc_provider._messages()).content == "Ответ"
    assert len(calls) == 2 and len(sockets[1].sent) == 1


def test_pool_does_not_evict_an_active_response_and_releases_abandoned_stream(monkeypatch):
    sockets = []

    def connect(*_args, **_kwargs):
        sock = FakeSocket([{"type": "response.created"}, {"type": "response.completed"}])
        sockets.append(sock)
        return sock

    monkeypatch.setattr(siwc_transport, "connect", connect)
    pool = SIWCWebSocketPool(limit=1)

    def events(key):
        return pool.events({"prompt_cache_key": key, "input": []}, account_id="account",
                           access_token="synthetic", timeout=1)

    first = events("a")
    try:
        assert next(first)["type"] == "response.created"
        with pytest.raises(ChatGPTOAuthUpstreamError) as caught:
            list(events("b"))
        assert caught.value.status == 429
        assert len(sockets) == 1 and not sockets[0].closed
    finally:
        first.close()
    assert sockets[0].closed
    assert len(list(events("b"))) == 2
    pool.close()
    assert sockets[-1].closed


def test_pool_lru_evicts_only_idle_channels_and_never_reuses_across_accounts(monkeypatch):
    sockets = []

    def connect(*_args, **_kwargs):
        sock = FakeSocket([{"type": "response.completed"}])
        sockets.append(sock)
        return sock

    monkeypatch.setattr(siwc_transport, "connect", connect)
    pool = SIWCWebSocketPool(limit=1)
    try:
        for account in ("account-a", "account-b"):
            list(pool.events({"prompt_cache_key": "same"}, account_id=account,
                             access_token="synthetic", timeout=1))
        assert len(sockets) == 2 and not sockets[1].closed
        pool.close()
        assert sockets[0].closed
    finally:
        pool.close()


@pytest.mark.parametrize("same_branch", [False, True])
def test_cancellation_closes_only_its_channel_and_completed_requests_unbind(monkeypatch, same_branch):
    sockets = []

    def connect(*_args, **_kwargs):
        sock = FakeSocket([{"type": "response.created"}, {"type": "response.completed"}])
        sockets.append(sock)
        return sock

    monkeypatch.setattr(siwc_transport, "connect", connect)
    pool = SIWCWebSocketPool()
    first_cancel = provider_module._RequestCancellation()
    second_cancel = provider_module._RequestCancellation()
    first = pool.events({"prompt_cache_key": "one"}, account_id="account", access_token="synthetic",
                        timeout=1, cancel=first_cancel)
    try:
        next(first)
        list(pool.events({"prompt_cache_key": "one" if same_branch else "two"}, account_id="account",
                         access_token="synthetic", timeout=1, cancel=second_cancel))
        second_cancel.cancel()
        assert not sockets[1].closed
        first_cancel.cancel()
        assert sockets[0].closed and not sockets[1].closed
    finally:
        first.close()
        pool.close()


def test_timing_logs_do_not_contain_prompt_or_auth(websocket_provider, caplog):
    caplog.set_level(logging.INFO, logger="codex_as_api")
    provider, _, _, _ = websocket_provider()
    provider.chat(test_siwc_provider._messages(), prompt_cache_key="private-session-marker")
    assert "SIWC WS start" in caplog.text and "elapsed_ms=" in caplog.text
    assert "first_event_ms=" in caplog.text and "first_delta_ms=" in caplog.text
    assert "private-session-marker" not in caplog.text and "synthetic-access" not in caplog.text


def test_cancel_after_completed_event_keeps_the_warm_channel(monkeypatch):
    sockets = []

    def connect(*_args, **_kwargs):
        sock = FakeSocket([{"type": "response.completed"}])
        sockets.append(sock)
        return sock

    monkeypatch.setattr(siwc_transport, "connect", connect)
    pool = SIWCWebSocketPool()
    cancel = provider_module._RequestCancellation()
    events = pool.events({"prompt_cache_key": "one"}, account_id="account", access_token="synthetic",
                         timeout=1, cancel=cancel)
    try:
        assert next(events)["type"] == "response.completed"
        # The SSE consumer may disconnect before resuming the upstream generator.
        cancel.cancel()
        assert not sockets[0].closed
    finally:
        events.close()
        pool.close()


LARGE_INPUT = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x" * (130 * 1024)}]}]
AFTER_INTERRUPT = LARGE_INPUT + [
    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "partial"}]},
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "[Request interrupted by user]"}]},
]


class SilentSocket(FakeSocket):
    def __init__(self, events):
        super().__init__(events)
        self.closing = threading.Event()

    def close(self):
        super().close()
        self.closing.set()

    def recv(self, *, timeout):
        if self.pending:
            return super().recv(timeout=timeout)
        if self.closing.wait(timeout):
            raise ConnectionClosedError(None, None)
        raise TimeoutError


def _cancelled_pool(monkeypatch, tail, socket_type=FakeSocket):
    sockets = []

    def connect(*_args, **_kwargs):
        sock = socket_type([{"type": "response.created"}, *tail])
        sockets.append(sock)
        return sock

    monkeypatch.setattr(siwc_transport, "connect", connect)
    pool = SIWCWebSocketPool()
    cancel = provider_module._RequestCancellation()
    events = pool.events({"prompt_cache_key": "one", "input": LARGE_INPUT}, account_id="account",
                         access_token="synthetic", timeout=1, cancel=cancel)
    assert next(events)["type"] == "response.created"
    cancel.cancel()
    events.close()
    deadline = time.monotonic() + 2
    while pool.active_requests and time.monotonic() < deadline:
        time.sleep(.01)
    assert pool.active_requests == 0
    return pool, sockets


def test_cancelled_large_answer_is_drained_and_keeps_the_warm_channel(monkeypatch):
    completed = {"type": "response.completed", "response": {"usage": {"input_tokens": 40000}}}
    pool, sockets = _cancelled_pool(monkeypatch, [{"type": "response.output_text.delta", "delta": "a"}, completed])
    try:
        assert not sockets[0].closed
        anchor = next(iter(pool._connections.values())).anchors[-1]
        assert anchor.drained and anchor.input_tokens == 40000
        events = list(pool.events({"prompt_cache_key": "one", "input": AFTER_INTERRUPT}, account_id="account",
                                  access_token="synthetic", timeout=1))
        assert events[-1]["type"] == "response.completed"
        assert len(sockets) == 1 and [len(p["input"]) for p in sockets[0].sent] == [1, 3]
    finally:
        pool.close()


@pytest.mark.parametrize("tail", [[{"type": "response.failed", "response": {"error": {"message": "synthetic"}}}],
                                  "deadline"])
def test_unfinished_or_failed_drain_closes_the_channel(monkeypatch, tail):
    monkeypatch.setattr(siwc_transport, "_DRAIN_SECONDS", .05)
    pool, sockets = _cancelled_pool(monkeypatch, [] if tail == "deadline" else tail,
                                    SilentSocket if tail == "deadline" else FakeSocket)
    try:
        assert sockets[0].closed
        assert not next(iter(pool._connections.values())).anchors
    finally:
        pool.close()



def test_drain_request_racing_a_normal_completion_keeps_the_socket(monkeypatch):
    monkeypatch.setattr(siwc_transport, "_DRAIN_SECONDS", .05)
    gate, sockets = threading.Event(), []

    class GateSocket(FakeSocket):
        def recv(self, *, timeout):
            if len(self.pending) == 1:
                assert gate.wait(timeout)
            return super().recv(timeout=timeout)

    def connect(*_args, **_kwargs):
        sockets.append(GateSocket([{"type": "response.created"}, {"type": "response.completed"}]))
        return sockets[-1]

    monkeypatch.setattr(siwc_transport, "connect", connect)
    pool = SIWCWebSocketPool()
    cancel = provider_module._RequestCancellation()
    events = pool.events({"prompt_cache_key": "one", "input": LARGE_INPUT}, account_id="account",
                         access_token="synthetic", timeout=1, cancel=cancel)
    try:
        assert next(events)["type"] == "response.created"
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(next, events)
            time.sleep(.02)
            cancel.cancel()
            gate.set()
            assert pending.result(timeout=1)["type"] == "response.completed"
        time.sleep(.1)  # The drain deadline passes while the generator waits at its final yield.
        assert not sockets[0].closed
        assert next(events, None) is None
        assert not sockets[0].closed and pool.active_requests == 0
    finally:
        events.close()
        pool.close()
