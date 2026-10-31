from __future__ import annotations

import json
from collections import deque

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response
from websockets.protocol import State

from codex_as_api import provider as provider_module
from codex_as_api import siwc_transport
from codex_as_api.auth import ChatGPTOAuthProtocolError, ChatGPTOAuthUpstreamError
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


def test_background_model_does_not_replace_the_main_models_channel(websocket_provider):
    provider, sockets, calls, _ = websocket_provider()
    for model in ("synthetic-first", "synthetic-second", "synthetic-first"):
        provider.chat(test_siwc_provider._messages(), model=model, prompt_cache_key="same-session")
    assert len(calls) == 2
    assert [len(sock.sent) for sock in sockets] == [2, 1]


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


def test_cancellation_closes_only_its_channel_and_completed_requests_unbind(monkeypatch):
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
        list(pool.events({"prompt_cache_key": "two"}, account_id="account", access_token="synthetic",
                         timeout=1, cancel=second_cancel))
        second_cancel.cancel()
        assert not sockets[1].closed
        first_cancel.cancel()
        assert sockets[0].closed and not sockets[1].closed
    finally:
        first.close()
        pool.close()
