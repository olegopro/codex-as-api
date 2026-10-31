from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from codex_as_api import server
from codex_as_api.auth import ChatGPTOAuthMissingError, ChatGPTOAuthProtocolError
from codex_as_api.codex_config import CodexConfig
from codex_as_api.provider import SIWC_BASE_URL
from codex_as_api.siwc_transport import SIWCWebSocketPool

from . import test_siwc_provider

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


@pytest.mark.parametrize("limit", [None, 7, 32])
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
def test_siwc_cli_dispatch_does_not_start_server(monkeypatch, args):
    import uvicorn

    import codex_as_api.siwc_cli as siwc_cli

    calls = []
    monkeypatch.setattr(server, "BACKEND", "siwc")
    monkeypatch.setattr(siwc_cli, "main", lambda argv: calls.append(argv) or 7)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Запуск сервера при login/status недопустим")

    monkeypatch.setattr(uvicorn, "run", forbidden)
    monkeypatch.setattr(server, "_get_provider", forbidden)
    assert server.main(args) == 7
    assert calls == [args]


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
