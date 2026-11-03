from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import anyio
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from codex_as_api import server
from codex_as_api.auth import ChatGPTOAuthMissingError, ChatGPTOAuthProtocolError
from codex_as_api.codex_config import CodexConfig
from codex_as_api.provider import SIWC_BASE_URL
from codex_as_api.siwc_transport import SIWCWebSocketPool

from . import test_siwc_provider, test_siwc_websocket

siwc_provider = test_siwc_provider.siwc_provider


def test_session_metadata_keeps_channel_affinity_without_a_header():
    body = {"metadata": {"user_id": json.dumps({"session_id": "session-a"})}}
    assert server._anthropic_prompt_cache_key(body, None) == server._anthropic_prompt_cache_key({}, "session-a")
    expected = server._anthropic_prompt_cache_key({}, "header-wins")
    assert server._anthropic_prompt_cache_key(body, "header-wins") == expected


@pytest.mark.parametrize("fail", [False, True])
def test_siwc_heartbeat_keeps_one_upstream_stream_and_preserves_completion(siwc_client, monkeypatch, fail):
    client, provider, _ = siwc_client
    original = provider.chat_stream
    calls = 0

    def delayed(*args, **kwargs):
        nonlocal calls
        calls += 1
        time.sleep(0.08)
        if fail:
            raise ChatGPTOAuthProtocolError("synthetic interrupted stream")
        yield from original(*args, **kwargs)

    monkeypatch.setattr(provider, "chat_stream", delayed)
    monkeypatch.setattr(server, "ANTHROPIC_HEARTBEAT_SECONDS", 0.01)
    response = client.post("/v1/messages", json=_body(stream=True))
    assert response.status_code == 200
    assert calls == 1
    assert 'event: ping\ndata: {"type":"ping"}\n\n' in response.text
    if fail:
        assert "event: error" in response.text
        assert "event: message_stop" not in response.text
    else:
        assert "event: message_stop" in response.text
        assert "event: error" not in response.text
    assert response.text.index("event: ping") < response.text.index("event: error" if fail else "event: message_stop")


def test_siwc_server_upstream_timeout_exceeds_the_client_idle_watchdog(monkeypatch):
    monkeypatch.setattr(server, "_provider", None)
    monkeypatch.setattr(server, "BACKEND", "siwc")
    arguments = {}

    def create(**kwargs):
        arguments.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(server, "ChatGPTOAuthProvider", create)
    server._get_provider()
    assert arguments["timeout"] == 600.0


@pytest.mark.parametrize("backend", ["siwc", "codex"])
@pytest.mark.parametrize("transport,limit", [("websocket", None), ("websocket", 100), ("http", None)])
def test_thread_capacity_matches_channels_without_affecting_legacy(monkeypatch, backend, transport, limit):
    pool = SIWCWebSocketPool(limit=limit) if transport == "websocket" else None
    monkeypatch.setattr(server, "BACKEND", backend)
    monkeypatch.setattr(server, "_provider", SimpleNamespace(_siwc_websockets=pool))

    async def exercise():
        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = 40
        await server._configure_siwc_thread_capacity()
        expected = 40 if backend == "codex" else math.inf if limit is None else 108
        assert limiter.total_tokens == expected
        assert server._anthropic_cancel_limiter.total_tokens == 8

    anyio.run(exercise)


@pytest.fixture()
def siwc_client(monkeypatch, siwc_provider):
    provider, requests = siwc_provider()
    monkeypatch.setattr(server, "BACKEND", "siwc")
    monkeypatch.setattr(server, "MODEL", None)
    monkeypatch.setattr(server, "CODEX_CONFIG", CodexConfig(codex_home="", config_path=""))
    monkeypatch.setattr(server, "_provider", provider)
    return TestClient(server.app, raise_server_exceptions=False), provider, requests


def _body(*, stream: bool = False) -> dict:
    return {
        "model": "claude-synthetic",
        "max_tokens": 1000,
        "system": "Основные инструкции",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Окружение"}]},
            {"role": "user", "content": "Вопрос"},
        ],
        "stream": stream,
    }


@pytest.mark.parametrize("stream", [False, True])
def test_anthropic_tool_image_reaches_siwc_as_ordered_function_output(siwc_client, stream):
    client, _, requests = siwc_client
    body = _body(stream=stream)
    body["tools"] = [{"name": "inspect", "description": "Fixture", "input_schema": {
        "type": "object", "properties": {}, "additionalProperties": False}}]
    body["messages"] += [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "call_image",
                                             "name": "inspect", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_image", "content": [
            {"type": "text", "text": "BEFORE"},
            {"type": "image", "source": {"type": "url", "url": "https://example.com/tool.png"}},
            {"type": "text", "text": "AFTER"},
        ]}]},
    ]
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 200, response.text
    payload = json.loads(requests[-1].content)
    assert payload["input"][-1] == {
        "type": "function_call_output", "call_id": "call_image", "output": [
            {"type": "input_text", "text": "BEFORE"},
            {"type": "input_image", "image_url": "https://example.com/tool.png"},
            {"type": "input_text", "text": "AFTER"},
        ],
    }


@pytest.mark.parametrize("limit,concurrency", [(100, 100), (None, 128)])
def test_http_stream_capacity_preserves_health_and_explicit_limits(siwc_client, monkeypatch, limit, concurrency):
    _, provider, _ = siwc_client
    pool = SIWCWebSocketPool() if limit is None else SIWCWebSocketPool(limit=limit)
    monkeypatch.setattr(provider, "_siwc_websockets", pool)
    monkeypatch.setattr(provider, "siwc_transport", "websocket")
    all_active = threading.Event()
    release = threading.Event()
    observed = []

    class WaitingSocket(test_siwc_websocket.FakeSocket):
        def recv(self, *, timeout):
            observed.append(pool.active_requests)
            if pool.active_requests == concurrency:
                all_active.set()
            assert release.wait(timeout=10)
            return super().recv(timeout=timeout)

    monkeypatch.setattr(test_siwc_websocket.siwc_transport, "connect", lambda *_args, **_kwargs:
                        WaitingSocket([test_siwc_provider._completed()]))
    with TestClient(server.app) as client, ThreadPoolExecutor(max_workers=concurrency) as workers:
        futures = [workers.submit(client.post, "/v1/messages", json=_body(stream=True)) for _ in range(concurrency)]
        try:
            ready = all_active.wait(timeout=5)
            if ready:
                health = client.get("/health").json()
                assert health["upstream_active_requests"] == concurrency
                assert health["upstream_max_connections"] == limit
                if limit is not None:
                    overflow = client.post("/v1/messages", json=_body(stream=True))
                    assert "event: error" in overflow.text
                    assert '"rate_limit_error"' in overflow.text
        finally:
            release.set()
        responses = [future.result(timeout=10) for future in futures]
        assert ready, f"Only {max(observed, default=0)} upstream receivers ran concurrently"
        assert all(response.status_code == 200 and "event: message_stop" in response.text for response in responses)


@pytest.mark.parametrize("facade", ["anthropic", "openai"])
@pytest.mark.parametrize("limit", [None, 1])
def test_siwc_stream_disconnect_releases_its_websocket(siwc_client, monkeypatch, facade, limit):
    _, provider, _ = siwc_client
    pool = SIWCWebSocketPool(limit=limit)
    release = threading.Event()
    monkeypatch.setattr(provider, "_siwc_websockets", pool)
    monkeypatch.setattr(provider, "siwc_transport", "websocket")

    async def exercise():
        entered, finished = anyio.Event(), anyio.Event()
        scopes = []

        class WaitingSocket(test_siwc_websocket.FakeSocket):
            def recv(self, *, timeout):
                anyio.from_thread.run_sync(entered.set)
                assert release.wait(2), "disconnect did not release the upstream receiver"
                return super().recv(timeout=timeout)

            def close(self):
                super().close()
                release.set()

        monkeypatch.setattr(test_siwc_websocket.siwc_transport, "connect", lambda *_args, **_kwargs:
                            WaitingSocket([test_siwc_provider._completed()]))
        body = _body(stream=True) if facade == "anthropic" else {
            "model": "synthetic-first", "messages": [{"role": "user", "content": "Question"}], "stream": True,
        }
        request = Request({"type": "http", "headers": []})
        if facade == "anthropic":
            async def receive():
                return {"type": "http.request", "body": json.dumps(body).encode()}

            request = Request({"type": "http", "headers": []}, receive=receive)
            response = await server.anthropic_messages(request)
        else:
            response = await server.chat_completions(server.ChatCompletionRequest(**body), request)

        async def consume():
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                async for _chunk in response.body_iterator:
                    pass
            finished.set()

        try:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(consume)
                with anyio.fail_after(1):
                    await entered.wait()
                assert pool.active_requests == 1
                scopes[0].cancel()
                try:
                    with anyio.fail_after(0.5):
                        await finished.wait()
                    assert pool.active_requests == 0
                finally:
                    release.set()
        finally:
            release.set()
            pool.close()

    anyio.run(exercise)


def test_siwc_image_generation_is_rejected_before_upstream(siwc_client):
    client, _, requests = siwc_client
    response = client.post("/v1/images/generations", json={"model": "gpt-6.1-sol", "prompt": "test"})
    assert response.status_code == 400
    assert requests == []


def test_siwc_health_reports_installed_vcs_revision(siwc_client, monkeypatch):
    client, _, _ = siwc_client
    commit = "a1" * 20
    document = {
        "url": "https://example.test/public-fork.git",
        "vcs_info": {"vcs": "git", "commit_id": commit, "requested_revision": "siwc-oauth"},
    }
    monkeypatch.setattr(server, "distribution", lambda _name: SimpleNamespace(
        read_text=lambda filename: json.dumps(document) if filename == "direct_url.json" else None,
    ))
    monkeypatch.setattr(server, "_RUNNING_PACKAGE_REVISION", server._package_revision())
    monkeypatch.setenv("CODEX_AS_API_PACKAGE_REVISION", "не-должно-использоваться")
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "siwc"
    assert body["upstream_base_url"] == SIWC_BASE_URL
    assert body["package_revision"] == commit
    assert body["status"] == "ok" and body["auth_available"] is True
    assert body["catalog_status"] == "fresh"
    assert body["model"] == "synthetic-first"
    assert "catalog_fetched_at" in body and "catalog_expires_at" in body
    assert "context_window" in body and "auto_compact_token_limit" in body
    assert "url" not in body and "auth_path" not in body


@pytest.mark.parametrize("limit", [None, 7, 32, 100])
def test_siwc_health_reports_the_running_pool_capacity(siwc_client, monkeypatch, limit):
    client, provider, _ = siwc_client
    pool = SIWCWebSocketPool(limit=limit) if limit is not None else None
    monkeypatch.setattr(provider, "_siwc_websockets", pool)
    try:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["upstream_max_connections"] == limit
        assert response.json()["upstream_active_requests"] == 0
    finally:
        if pool is not None:
            pool.close()


def test_siwc_health_does_not_claim_an_update_before_process_restart(siwc_client, monkeypatch):
    client, _, _ = siwc_client
    monkeypatch.setattr(server, "_RUNNING_PACKAGE_REVISION", "a" * 40)
    monkeypatch.setattr(server, "_package_revision", lambda: "b" * 40)
    assert client.get("/health").json()["package_revision"] == "a" * 40


@pytest.mark.parametrize("direct_url", [
    None, "не JSON", "[]", '{"dir_info":{"editable":true}}',
    '{"vcs_info":{"vcs":"git","commit_id":"branch-name"}}',
    '{"vcs_info":{"vcs":"svn","commit_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}',
])
def test_siwc_revision_is_unknown_without_installed_git_commit(monkeypatch, direct_url):
    monkeypatch.setattr(server, "distribution", lambda _name: SimpleNamespace(read_text=lambda _filename: direct_url))
    monkeypatch.setenv("CODEX_AS_API_PACKAGE_REVISION", "b" * 40)
    assert server._package_revision() is None


def test_siwc_health_keeps_transport_identity_on_auth_failure(siwc_client, monkeypatch):
    client, provider, _ = siwc_client

    def missing():
        raise ChatGPTOAuthMissingError("synthetic-access: недоступный секрет")

    monkeypatch.setattr(provider, "get_model_catalog", missing)
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["backend"] == "siwc" and body["upstream_base_url"] == SIWC_BASE_URL
    assert "package_revision" in body
    assert body["auth_available"] is False
    assert "synthetic-access" not in response.text


def test_siwc_models_expose_only_whitelisted_catalog_metadata(siwc_client):
    client, _, _ = siwc_client
    response = client.get("/v1/models")
    assert response.status_code == 200
    models = response.json()["data"]
    assert [model["id"] for model in models] == ["synthetic-first", "synthetic-second"]
    assert models[0]["context_window"] == 100_000
    assert models[0]["supported_reasoning_levels"] == [{"effort": "low", "description": "Низкий"}]
    assert "access_token" not in response.text
    assert "непубликуемое-поле" not in response.text
    assert "synthetic-subject" not in response.text
    assert "synthetic-access" not in response.text


def test_siwc_anthropic_system_environment_and_unique_ids(siwc_client):
    client, _, requests = siwc_client
    first = client.post("/v1/messages", json=_body())
    second = client.post("/v1/messages", json=_body())
    assert first.status_code == second.status_code == 200
    assert first.json()["id"] != second.json()["id"]
    assert first.json()["content"][0]["type"] == "text"
    assert first.json()["content"][0]["text"] == "Ответ"
    assert first.json()["usage"]["input_tokens"] == 8
    assert first.json()["usage"]["cache_read_input_tokens"] == 12
    payload = json.loads(requests[-1].content)
    assert payload["instructions"] == "Основные инструкции\n\nОкружение"
    assert not any(item.get("role") == "system" for item in payload["input"])
    assert "max_output_tokens" not in payload


def test_siwc_stream_ids_are_unique_and_cache_is_not_counted_twice(siwc_client):
    client, _, _ = siwc_client
    first = client.post("/v1/messages", json=_body(stream=True))
    second = client.post("/v1/messages", json=_body(stream=True))
    events = [json.loads(line[6:]) for line in first.text.splitlines() if line.startswith("data: ")]
    second_events = [json.loads(line[6:]) for line in second.text.splitlines() if line.startswith("data: ")]
    start = next(event for event in events if event["type"] == "message_start")
    second_start = next(event for event in second_events if event["type"] == "message_start")
    assert start["message"]["id"] != second_start["message"]["id"]
    finish = next(event for event in events if event["type"] == "message_delta")
    assert finish["usage"]["input_tokens"] == 8
    assert finish["usage"]["cache_read_input_tokens"] == 12
    assert events[-1]["type"] == "message_stop"


@pytest.mark.parametrize("events", [
    [],
    [{"type": "response.failed", "response": {"error": {"message": "Ошибка"}}}],
    [{"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}}],
])
def test_siwc_anthropic_stream_error_has_no_success_stop(siwc_provider, monkeypatch, events):
    provider, _ = siwc_provider(events=events)
    monkeypatch.setattr(server, "_provider", provider)
    monkeypatch.setattr(server, "CODEX_CONFIG", CodexConfig(codex_home="", config_path=""))
    client = TestClient(server.app, raise_server_exceptions=False)
    response = client.post("/v1/messages", json=_body(stream=True))
    assert response.status_code == 200
    assert "event: error" in response.text
    assert "event: message_stop" not in response.text


@pytest.mark.parametrize("path", ["/v1/compact", "/v1/messages/compact"])
def test_siwc_compact_is_explicitly_unsupported_without_upstream(siwc_client, path):
    client, _, requests = siwc_client
    response = client.post(path, json={"messages": [{"role": "user", "content": "История"}]})
    assert response.status_code == 400
    assert "SIWC" in response.text
    assert not requests


def test_siwc_factory_keeps_the_selected_backend(monkeypatch):
    monkeypatch.setattr(server, "BACKEND", "siwc")
    monkeypatch.setattr(server, "MODEL", None)
    monkeypatch.setattr(server, "_provider", None)
    provider = server._get_provider()
    assert provider.backend == "siwc" and provider.base_url == SIWC_BASE_URL


@pytest.mark.parametrize("args", [["login"], ["status"], ["--auth-path", "synthetic.json", "status"]])
@pytest.mark.parametrize("backend", ["siwc", "codex"])
def test_siwc_cli_dispatch_does_not_start_server(monkeypatch, args, backend):
    import uvicorn

    import codex_as_api.siwc_cli as siwc_cli

    calls = []
    monkeypatch.setattr(server, "BACKEND", backend)
    monkeypatch.setattr(siwc_cli, "main", lambda argv, *, backend: calls.append((argv, backend)) or 7)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Запуск сервера при login/status недопустим")

    monkeypatch.setattr(uvicorn, "run", forbidden)
    monkeypatch.setattr(server, "_get_provider", forbidden)
    assert server.main(args) == 7
    assert calls == [(args, backend)]


def test_siwc_browser_manager_uses_public_auth_factory(monkeypatch):
    import codex_as_api.siwc_auth as siwc_auth

    expected = object()
    calls = []
    monkeypatch.setattr(server, "BACKEND", "siwc")
    monkeypatch.setattr(server, "AUTH_PATH", None)
    monkeypatch.setattr(server, "_browser_login_manager", None)
    monkeypatch.setattr(siwc_auth, "make_browser_login_manager", lambda path: calls.append(path) or expected)
    assert server._get_browser_login_manager() is expected
    assert server._get_browser_login_manager() is expected
    assert calls == [None]


def test_siwc_device_login_is_not_routed_to_legacy(monkeypatch):
    monkeypatch.setattr(server, "BACKEND", "siwc")
    monkeypatch.setenv("CODEX_AS_API_ADMIN_KEY", "synthetic-admin")
    client = TestClient(server.app, raise_server_exceptions=False)
    response = client.post(
        "/auth/login", json={"method": "device"}, headers={"Authorization": "Bearer synthetic-admin"},
    )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "device_login_unsupported"


def test_siwc_management_status_checks_the_public_store_and_profile(monkeypatch, tmp_path):
    import codex_as_api.siwc_auth as siwc_auth

    path = tmp_path / "synthetic-public.json"
    path.write_text("{}")
    calls = []
    monkeypatch.setattr(server, "BACKEND", "siwc")
    monkeypatch.setattr(server, "AUTH_PATH", None)
    monkeypatch.setenv("CODEX_AS_API_ADMIN_KEY", "synthetic-admin")
    monkeypatch.setattr(siwc_auth, "resolve_auth_path", lambda _path: path)

    def load_token(selected_path):
        calls.append(selected_path)
        return SimpleNamespace(oauth_profile="codex")

    monkeypatch.setattr(server, "load_token_data", load_token)
    client = TestClient(server.app, raise_server_exceptions=False)
    response = client.get("/auth/status", headers={"Authorization": "Bearer synthetic-admin"})
    assert response.status_code == 200
    assert response.json()["status"] == "reauthentication_required"
    assert calls == [path]


def test_siwc_explicit_model_routing_does_not_use_facade_default(siwc_client):
    client, _, requests = siwc_client
    body = _body()
    body["model"] = "synthetic-second"
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 200
    assert json.loads(requests[-1].content)["model"] == "synthetic-second"
    body["model"] = "missing-synthetic-model"
    response = client.post("/v1/messages", json=body)
    assert response.status_code == 404
    assert len([request for request in requests if request.method == "POST"]) == 1
