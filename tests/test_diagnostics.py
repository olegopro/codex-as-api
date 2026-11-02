from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import anyio
import pytest
from fastapi.testclient import TestClient

from codex_as_api import diagnostics, server, siwc_transport
from codex_as_api.auth import ChatGPTOAuthInvalidRequestError
from codex_as_api.siwc_transport import SIWCWebSocketPool

from .test_siwc_websocket import FakeSocket


@pytest.fixture()
def diagnostic_log(tmp_path, monkeypatch):
    path = tmp_path / "diagnostics.jsonl"
    monkeypatch.setenv("CODEX_AS_API_DIAGNOSTICS", "1")
    monkeypatch.setenv("CODEX_AS_API_DIAGNOSTICS_PATH", str(path))
    logger = diagnostics._make_logger()
    monkeypatch.setattr(diagnostics, "_LOG", logger)
    yield path
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_disabled_and_invalid_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_AS_API_DIAGNOSTICS", raising=False)
    monkeypatch.setenv("CODEX_AS_API_DIAGNOSTICS_PATH", str(tmp_path / "unused"))
    assert diagnostics._make_logger() is None
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("CODEX_AS_API_DIAGNOSTICS", "1")
    monkeypatch.setenv("CODEX_AS_API_DIAGNOSTICS_PATH", "relative")
    with pytest.raises(ValueError):
        diagnostics._make_logger()


def test_rejects_symlink_and_restricts_existing_backups(tmp_path, monkeypatch):
    path = tmp_path / "diagnostics.jsonl"
    backup = tmp_path / "diagnostics.jsonl.1"
    backup.write_text("old\n")
    backup.chmod(0o644)
    path.symlink_to(backup)
    monkeypatch.setenv("CODEX_AS_API_DIAGNOSTICS", "1")
    monkeypatch.setenv("CODEX_AS_API_DIAGNOSTICS_PATH", str(path))
    with pytest.raises(ValueError):
        diagnostics._make_logger()
    path.unlink()
    logger = diagnostics._make_logger()
    try:
        assert backup.stat().st_mode & 0o777 == 0o600
        assert path.stat().st_mode & 0o777 == 0o600
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def test_early_validation_and_secrets(diagnostic_log, monkeypatch, caplog):
    monkeypatch.setattr(server, "_get_provider", lambda: object())
    caplog.set_level(logging.INFO)
    client = TestClient(server.app, raise_server_exceptions=False)
    headers = {"authorization": "Bearer auth-canary", "x-test": "header-canary"}
    first = client.post("/v1/messages", content='{ "prompt-canary": ',
                        headers={**headers, "content-type": "application/json"})
    second = client.post("/v1/messages", headers=headers, json={
        "model": "gpt-test", "max_tokens": 100, "messages": [{"role": "user", "content": "prompt-canary"}],
        "thinking": {"type": "adaptive", "display": "full"},
        "tools": [{"name": "tool", "description": "toolargs-canary", "input_schema": {"type": "object"}}],
    })
    third = client.post("/v1/messages", headers=headers, json={"field-name-canary": "value-canary"})
    assert first.status_code == second.status_code == third.status_code == 400
    assert "field-name-canary" in third.text
    data = records(diagnostic_log)
    errors = [item for item in data if item["stage"] == "validation_error"]
    assert [item["reason"] for item in errors] == ["invalid_json", "thinking_display_invalid", "invalid_request"]
    assert len({item["http_id"] for item in errors}) == 3
    assert any(item.get("message_count") == 1 and item.get("content_items") == 1 for item in data)
    assert sum("body_bytes" in item for item in data) == 3
    assert not any(item["stage"] == "upstream_start" for item in data)
    combined = diagnostic_log.read_text() + caplog.text
    for canary in ("prompt-canary", "auth-canary", "header-canary", "toolargs-canary", "field-name-canary"):
        assert canary not in combined
    assert diagnostics.current() is None


def test_stream_error_stays_in_client_response_but_not_logs(diagnostic_log, monkeypatch, caplog):
    class Provider:
        backend = "siwc"

        def preflight_chat(self, *args, **kwargs):
            return object()

        def chat_stream(self, *args, **kwargs):
            raise ChatGPTOAuthInvalidRequestError("upstream-error-canary")
            yield  # pragma: no cover

    monkeypatch.setattr(server, "_get_provider", lambda: Provider())
    model = type("Model", (), {"slug": "gpt-test"})()
    monkeypatch.setattr(server, "_anthropic_backend_model", lambda *a: (None, model))
    monkeypatch.setattr(server, "_validate_configured_reasoning_effort", lambda *a, **kw: None)
    caplog.set_level(logging.WARNING)
    response = TestClient(server.app, raise_server_exceptions=False).post("/v1/messages", json={
        "model": "gpt-test", "max_tokens": 50, "messages": [{"role": "user", "content": "prompt-canary"}],
        "stream": True,
    })
    assert response.status_code == 200 and "upstream-error-canary" in response.text
    assert "event: message_stop" not in response.text
    assert records(diagnostic_log)[-1]["stage"] == "http_error"
    assert "canary" not in diagnostic_log.read_text() + caplog.text


def test_exception_log_keeps_category_without_exception_or_traceback_text(caplog):
    caplog.set_level(logging.WARNING)
    try:
        raise ChatGPTOAuthInvalidRequestError("secret-field-canary")
    except ChatGPTOAuthInvalidRequestError as exc:
        assert server._public_error_message(exc) == "secret-field-canary"
    assert "invalid_request_error" in caplog.text
    assert "secret-field-canary" not in caplog.text
    assert "Traceback" not in caplog.text


def test_private_rotation(diagnostic_log):
    handler = diagnostics._LOG.handlers[0]
    assert handler.maxBytes == 5 * 1024 * 1024 and handler.backupCount == 3
    handler.maxBytes = 100
    for _ in range(40):
        diagnostics.emit("http_start")
    files = sorted(diagnostic_log.parent.iterdir())
    assert len(files) == 4
    assert all(os.stat(path).st_mode & 0o777 == 0o600 for path in files)


def test_context_correlation_worker_reuse_and_concurrency(diagnostic_log, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="codex_as_api.siwc_transport")
    monkeypatch.setattr(siwc_transport, "connect", lambda *a, **kw: FakeSocket([
        {"type": "response.completed", "response": {"output": [], "usage": {
            "input_tokens": 30, "output_tokens": 5, "total_tokens": 35,
            "input_tokens_details": {"cached_tokens": 20},
        }}}
    ]))
    pool = SIWCWebSocketPool()

    async def app(scope, receive, send):
        async def request_worker():
            def run():
                session = scope["path"].strip("/")
                payload = {"model": "gpt-6.1-terra", "prompt_cache_key": session,
                           "input": [{"role": "user", "content": "prompt-canary"},
                                     {"type": "function_call", "arguments": "toolargs-canary",
                                      "signature": "signature-canary"}]}
                for _ in range(2):
                    list(pool.events(payload, account_id="account", access_token="auth-canary", timeout=1))
            await anyio.to_thread.run_sync(run)
        await request_worker()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    client = TestClient(diagnostics.LifecycleMiddleware(app))
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            assert list(executor.map(lambda path: client.get(path).status_code, ["/a", "/b"])) == [200, 200]
    finally:
        pool.close()
    data = records(diagnostic_log)
    starts = [r for r in data if r["stage"] == "upstream_start" and "session_hash" in r]
    assert len(starts) == 4 and all(r["model"] == "gpt-6.1-terra" for r in starts)
    assert len({r["http_id"] for r in starts}) == 2
    assert all(len({r["session_hash"] for r in starts if r["http_id"] == http_id}) == 1
               for http_id in {r["http_id"] for r in starts})
    outcomes = [r for r in data if r["stage"] == "upstream_completed"]
    assert len(outcomes) == 4 and all(r["cached_tokens"] == 20 and r["total_tokens"] == 35 for r in outcomes)
    assert sum(r["reused"] for r in outcomes) == 2
    reasons = [r["reason"] for r in data if r["stage"] == "upstream_start" and "reason" in r]
    assert sorted(reasons) == ["no_socket", "no_socket", "reused", "reused"]
    assert sum(r["stage"] == "http_stream_finished" for r in data) == 2
    assert all(r["attempt_id"] in {s["attempt_id"] for s in starts} for r in outcomes)
    assert "canary" not in diagnostic_log.read_text()
    transport_logs = " ".join(r.getMessage() for r in caplog.records if r.name == "codex_as_api.siwc_transport")
    assert "cached=" not in transport_logs and "session=" not in transport_logs


@pytest.mark.parametrize("event_type,reason", [
    ("response.failed", "upstream_failed"),
    ("response.incomplete", "upstream_incomplete"),
    ("error", "upstream_error"),
])
def test_real_transport_terminal_error(diagnostic_log, monkeypatch, event_type, reason):
    monkeypatch.setattr(siwc_transport, "connect", lambda *a, **kw: FakeSocket([
        {"type": event_type, "error": {"message": "upstream-error-canary"}}
    ]))
    pool = SIWCWebSocketPool()

    async def app(scope, receive, send):
        def run():
            return list(pool.events({"input": [], "model": "gpt-test"}, account_id="a",
                                    access_token="auth-canary", timeout=1))
        await anyio.to_thread.run_sync(run)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"error"})

    try:
        assert TestClient(diagnostics.LifecycleMiddleware(app)).get("/").status_code == 200
    finally:
        pool.close()
    data = records(diagnostic_log)
    assert any(r["stage"] == "upstream_error" and r["reason"] == reason for r in data)
    assert data[-1]["stage"] == "http_error"
    assert not any(r["stage"] in {"upstream_completed", "http_stream_finished"} for r in data)
    assert "canary" not in diagnostic_log.read_text()


def test_partial_stream_exception_does_not_finish_response(diagnostic_log, caplog):
    sent = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"event: message_start\n\n", "more_body": True})
        raise RuntimeError("upstream-secret-canary")

    async def run():
        async def receive():
            return {"type": "http.request"}
        async def send(message):
            sent.append(message)
        await diagnostics.LifecycleMiddleware(app)({"type": "http"}, receive, send)
        assert diagnostics.current() is None

    anyio.run(run)
    assert len(sent) == 2 and sent[-1]["more_body"] is True
    assert records(diagnostic_log)[-1]["stage"] == "http_error"
    assert "upstream-secret-canary" not in diagnostic_log.read_text() + caplog.text


def test_real_cancellation_clears_context(diagnostic_log):
    async def app(scope, receive, send):
        await anyio.sleep_forever()

    async def run():
        async def io(*args):
            pass
        with anyio.CancelScope() as cancel_scope:
            cancel_scope.cancel()
            await diagnostics.LifecycleMiddleware(app)({"type": "http"}, io, io)
        assert diagnostics.current() is None

    anyio.run(run)
    assert records(diagnostic_log)[-1]["stage"] == "http_cancelled"


@pytest.mark.parametrize("failure", ["failed", "incomplete", "disconnect", "exception"])
def test_terminal_failure_is_not_success(diagnostic_log, failure):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        if failure in {"failed", "incomplete"}:
            diagnostics.current()["failed"] = True
            await send({"type": "http.response.body", "body": b"error"})
        elif failure == "disconnect":
            await receive()
        else:
            raise RuntimeError("upstream-error-canary")

    async def run():
        async def receive():
            return {"type": "http.disconnect"}
        async def send(message):
            pass
        await diagnostics.LifecycleMiddleware(app)({"type": "http"}, receive, send)
    anyio.run(run)
    stages = [r["stage"] for r in records(diagnostic_log)]
    assert "http_stream_finished" not in stages
    assert stages[-1] == {"failed": "http_error", "incomplete": "http_error", "disconnect": "http_disconnected",
                          "exception": "http_error"}[failure]
    assert "upstream-error-canary" not in diagnostic_log.read_text()
