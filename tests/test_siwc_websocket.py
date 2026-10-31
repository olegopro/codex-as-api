from __future__ import annotations

import json
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
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
@pytest.mark.parametrize("concurrency", [2, 20, 32, 100])
def test_same_parent_model_requests_reach_upstream_concurrently(monkeypatch, identical_prompts, concurrency):
    sockets = []
    observed_active = []
    pool = SIWCWebSocketPool()
    assert pool.max_connections == 100

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
        else:
            pool.close()
        assert all(socket.closed for socket in sockets[:8])
        assert pool.active_requests == 0
    finally:
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


def test_siwc_websocket_rotated_token_reopens_connection(websocket_provider, monkeypatch):
    provider, sockets, calls, _ = websocket_provider()
    provider.chat(test_siwc_provider._messages())
    replacement = test_siwc_provider._token()
    replacement.access_token = "synthetic-new-access"
    monkeypatch.setattr(provider_module, "token_for_request", lambda *_args, **_kwargs: replacement)
    provider.chat(test_siwc_provider._messages())
    assert len(calls) == 2 and sockets[0].closed
    assert calls[-1][1]["additional_headers"]["Authorization"] == "Bearer synthetic-new-access"


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
        assert len(sockets) == 2 and sockets[0].closed and not sockets[1].closed
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
