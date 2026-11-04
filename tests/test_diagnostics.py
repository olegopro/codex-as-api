from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import anyio
import pytest
from fastapi.testclient import TestClient

from codex_as_api import cache_diagnostics, diagnostics, server, siwc_transport
from codex_as_api.auth import ChatGPTOAuthInvalidRequestError
from codex_as_api.provider import ChatGPTOAuthProvider
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


def test_detached_subprocess_after_uvicorn_logging_configuration(tmp_path):
    path = tmp_path / "diagnostics.jsonl"
    code = """
import fcntl
import logging.config
import os
from codex_as_api import diagnostics
from uvicorn.config import LOGGING_CONFIG
logging.config.dictConfig(LOGGING_CONFIG)
diagnostics.emit('http_start', query='secret-canary')
fd = os.open(os.environ['CODEX_AS_API_DIAGNOSTICS_PATH'] + '.lock', os.O_WRONLY)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    pass
else:
    raise AssertionError('reopened handler did not retain exclusive lock')
finally:
    os.close(fd)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={"CODEX_AS_API_DIAGNOSTICS": "1", "CODEX_AS_API_DIAGNOSTICS_PATH": str(path)},
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        start_new_session=True, timeout=10, check=True,
    )
    assert "diagnostics write failed" not in result.stderr
    assert records(path)[0]["stage"] == "http_start"
    assert "secret-canary" not in path.read_text() + result.stdout + result.stderr
    assert path.stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "diagnostics.jsonl.lock").stat().st_mode & 0o777 == 0o600


def test_private_rotation(diagnostic_log):
    handler = diagnostics._LOG.handlers[0]
    assert handler.max_bytes == 5 * 1024 * 1024 and handler.max_total_bytes == 256 * 1024 * 1024
    handler.max_bytes = 150
    for _ in range(8):
        diagnostics.emit("http_start")
    files = sorted(diagnostic_log.parent.iterdir())
    assert len(files) > 1
    assert all(os.stat(path).st_mode & 0o777 == 0o600 for path in files)


def test_retention_and_capacity_warn_without_touching_other_files(diagnostic_log, caplog):
    handler = diagnostics._LOG.handlers[0]
    handler.max_bytes = 150
    handler.max_total_bytes = 450
    unrelated = diagnostic_log.parent / "unrelated"
    unrelated.write_text("private-canary")
    caplog.set_level(logging.WARNING)
    for _ in range(8):
        diagnostics.emit("http_start")
    assert sum(p.stat().st_size for p in diagnostic_log.parent.iterdir() if p != unrelated) <= 450
    assert "retention incomplete category=capacity" in caplog.text
    assert unrelated.read_text() == "private-canary"
    segments = list(diagnostic_log.parent.glob("*.segment.*"))
    assert segments
    os.utime(segments[0], (1, 1))
    diagnostics.emit("http_start")
    assert not segments[0].exists()


def test_startup_prunes_expired_segments_and_refuses_segment_symlink(tmp_path, monkeypatch):
    path = tmp_path / "diagnostics.jsonl"
    old = tmp_path / ("diagnostics.jsonl.segment." + "a" * 32)
    old.write_text("old")
    os.utime(old, (1, 1))
    target = tmp_path / "target"
    target.write_text("protected-canary")
    link = tmp_path / ("diagnostics.jsonl.segment." + "b" * 32)
    link.symlink_to(target)
    monkeypatch.setenv("CODEX_AS_API_DIAGNOSTICS", "1")
    monkeypatch.setenv("CODEX_AS_API_DIAGNOSTICS_PATH", str(path))
    with pytest.raises(ValueError):
        diagnostics._make_logger()
    assert target.read_text() == "protected-canary"
    link.unlink()
    logger = diagnostics._make_logger()
    try:
        assert not old.exists()
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def test_lock_rejects_second_writer_and_symlink(diagnostic_log, tmp_path):
    with pytest.raises(ValueError, match="already in use"):
        diagnostics._PrivateRotatingFileHandler(str(diagnostic_log))
    target = tmp_path / "lock-target"
    target.write_text("protected-canary")
    path = tmp_path / "other.jsonl"
    (tmp_path / "other.jsonl.lock").symlink_to(target)
    with pytest.raises(OSError):
        diagnostics._PrivateRotatingFileHandler(str(path))
    assert target.read_text() == "protected-canary"


def test_inherited_handler_does_not_write_from_another_process(diagnostic_log, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    handler = diagnostics._LOG.handlers[0]
    monkeypatch.setattr(handler, "_owner_pid", -1)
    diagnostics.emit("http_start")
    assert diagnostic_log.read_text() == ""
    assert "category=process_changed" in caplog.text


def test_diagnostic_allowlist_rejects_content(diagnostic_log):
    diagnostics.emit("search_completed", stream=True, result_count=2, search_requests=1,
                     query="query-canary", url="url-canary", reason="error-canary",
                     output_key_source="completed", output_key_error="invalid_output")
    data = records(diagnostic_log)[0]
    assert data["stream"] is True and data["result_count"] == 2 and data["search_requests"] == 1
    assert data["output_key_source"] == "completed" and data["output_key_error"] == "invalid_output"
    assert len(data["process_id"]) == 32 and "canary" not in diagnostic_log.read_text()


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


def test_selection_and_output_diagnostics_are_correlated(diagnostic_log, monkeypatch):
    monkeypatch.setattr(siwc_transport, "connect", lambda *a, **kw: FakeSocket([
        {"type": "response.output_item.done", "item": {"role": "assistant", "content": "answer"}},
        {"type": "response.completed", "response": {"output": []}},
    ]))
    pool = SIWCWebSocketPool()
    options = {"account_id": "account", "access_token": "auth-canary", "timeout": 1}
    root = [{"role": "user", "content": "prompt-canary"}]
    try:
        list(pool.events({"input": root}, **options))
        list(pool.events({"input": [*root, {"role": "assistant", "content": "wrong-canary"}]}, **options))
    finally:
        pool.close()
    data = records(diagnostic_log)
    selections = [item for item in data if item["stage"] == "channel_selected"]
    assert [item["selection_reason"] for item in selections] == ["new", "replay_mismatch"]
    assert selections[1]["free_candidates"] == 1 and selections[1]["busy_candidates"] == 0
    for selection in selections:
        related = [item for item in data if item.get("attempt_id") == selection["attempt_id"]]
        output = next(item for item in related if item["stage"] == "output_key")
        release = next(item for item in related if item["stage"] == "channel_released")
        assert output["channel_id"] == release["channel_id"] == selection["channel_id"]
        assert output["output_key_source"] == "done" and output["output_key_error"] == "none"
        assert output["done_count"] == 1 and output["completed_count"] == 0
        assert release["reason"] == "completed"
    assert "canary" not in diagnostic_log.read_text()


@pytest.mark.parametrize("output,expected_error", [
    ([{"role": "assistant", "content": "answer"}], "none"),
    ([{"role": "assistant", "content": None}], "invalid_output"),
])
def test_completed_output_key_source_and_error_category(diagnostic_log, monkeypatch, output, expected_error):
    monkeypatch.setattr(siwc_transport, "connect", lambda *a, **kw: FakeSocket([
        {"type": "response.completed", "response": {"output": output}},
    ]))
    pool = SIWCWebSocketPool()
    try:
        list(pool.events({"input": [{"role": "user", "content": "secret-canary"}]},
                         account_id="account", access_token="auth-canary", timeout=1))
    finally:
        pool.close()
    item = next(item for item in records(diagnostic_log) if item["stage"] == "output_key")
    assert item["output_key_source"] == "completed" and item["output_key_error"] == expected_error
    assert item["done_count"] == 0 and item["completed_count"] == 1
    assert "canary" not in diagnostic_log.read_text()


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


@pytest.mark.parametrize("backend,transport", [("codex", "http"), ("siwc", "http"), ("siwc", "websocket")])
def test_cache_transport_observation_preserves_events_and_payload(diagnostic_log, monkeypatch, backend, transport):
    provider = object.__new__(ChatGPTOAuthProvider)
    provider.backend = backend
    provider.siwc_transport = transport
    payload = {"model": "gpt-6.1-terra", "instructions": "instructions-canary",
               "input": [{"role": "user", "content": "prompt-canary"}],
               "tools": [{"name": "tool-canary"}], "prompt_cache_key": "key-canary"}
    headers = {"Authorization": "auth-canary"}
    events = [{"type": "response.output_text.delta", "delta": "output-canary"},
              {"type": "response.completed", "response": {"id": "resp_test123", "model": "gpt-6.1-terra",
               "service_tier": "default", "usage": {"input_tokens": 2000, "output_tokens": 10,
               "input_tokens_details": {"cached_tokens": 1920}}}}]
    observed = []

    def request(path, body, **kwargs):
        assert path == "/responses" and body is payload and kwargs["extra_headers"] is headers
        for event in events:
            observed.append(event)
            yield event

    monkeypatch.setattr(provider, "_request_sse", request)
    iterator = provider._post_sse("/responses", payload, extra_headers=headers)
    assert next(iterator) is events[0] and len(observed) == 1
    assert list(iterator) == events[1:]
    data = records(diagnostic_log)
    assert [r["stage"] for r in data] == ["cache_request", "cache_response"]
    start, finish = data
    assert start["backend"] == backend and start["transport"] == transport
    assert finish["attempt_id"] == start["attempt_id"]
    assert finish["payload_hash"] == start["payload_hash"]
    assert finish["cached_tokens"] == 1920 and finish["usage_available"] is True
    assert finish["upstream_response_id"] == "resp_test123"
    assert "canary" not in diagnostic_log.read_text()


def test_cache_prefixes_preserve_history_and_detect_early_edits(diagnostic_log):
    payload = {"model": "gpt-6-luna", "input": [{"role": "user", "content": "one-canary"}],
               "tools": [{"name": "Read"}], "reasoning": {"effort": "low"}}
    list(cache_diagnostics.observe_events(iter([]), payload, backend="codex", transport="http"))
    payload["input"].append({"role": "developer", "content": "late-note-canary"})
    list(cache_diagnostics.observe_events(iter([]), payload, backend="codex", transport="http"))
    payload["input"][0]["content"] = "changed-canary"
    payload["reasoning"]["effort"] = "high"
    list(cache_diagnostics.observe_events(iter([]), payload, backend="codex", transport="http"))
    starts = [r for r in records(diagnostic_log) if r["stage"] == "cache_request"]
    assert starts[0]["input_prefix_hashes"] == starts[1]["input_prefix_hashes"][:1]
    assert starts[2]["input_prefix_hashes"][0] != starts[1]["input_prefix_hashes"][0]
    assert starts[0]["settings_hash"] == starts[1]["settings_hash"] != starts[2]["settings_hash"]
    assert "canary" not in diagnostic_log.read_text()


def test_cache_observation_disabled_skips_fingerprints(monkeypatch):
    monkeypatch.setattr(diagnostics, "_LOG", None)

    def fail(_):
        raise AssertionError("disabled diagnostics must not fingerprint payloads")

    monkeypatch.setattr(cache_diagnostics, "_hash", fail)
    event = {"type": "response.completed"}
    assert list(cache_diagnostics.observe_events(iter([event]), {}, backend="codex", transport="http")) == [event]


@pytest.mark.parametrize("mode,expected", [("close", "interrupted"), ("error", "upstream_error"),
                                          ("end", "upstream_incomplete"), ("no_usage", "completed")])
def test_cache_observation_does_not_invent_zero_usage(diagnostic_log, mode, expected):
    def events():
        yield {"type": "response.created"}
        if mode == "error":
            raise RuntimeError("error-canary")
        if mode == "no_usage":
            yield {"type": "response.completed", "response": {}}

    iterator = cache_diagnostics.observe_events(events(), {}, backend="codex", transport="http")
    next(iterator)
    if mode == "close":
        iterator.close()
    elif mode == "error":
        with pytest.raises(RuntimeError, match="error-canary"):
            list(iterator)
    else:
        list(iterator)
    finish = records(diagnostic_log)[-1]
    assert finish["reason"] == expected and finish["usage_available"] is False
    assert "cached_tokens" not in finish
    assert "canary" not in diagnostic_log.read_text()


def test_cache_http_fingerprints_final_headers_without_credentials(diagnostic_log):
    payload = {"input": "prompt-canary"}
    headers = {"Authorization": "auth-canary", "Cookie": "cookie-canary", "session-id": "session-canary",
               "thread-id": "thread-canary", "ChatGPT-Account-ID": "account-canary"}
    cache_diagnostics.http_request(payload, headers, b"wire-body-canary", 0)
    headers["Authorization"] = "new-auth-canary"
    cache_diagnostics.http_request(payload, headers, b"wire-body-canary", 1)
    headers["thread-id"] = "new-thread-canary"
    cache_diagnostics.http_request(payload, headers, b"wire-body-canary", 0)
    cache_diagnostics.http_response(payload, 200, {"x-request-id": "req_test123"}, 0)
    a, b, c, response = records(diagnostic_log)
    assert a["routing_hash"] == b["routing_hash"] != c["routing_hash"]
    assert a["session_header_hash"] == b["session_header_hash"] == c["session_header_hash"]
    assert a["thread_header_hash"] != c["thread_header_hash"]
    assert response["upstream_request_id"] == "req_test123" and response["status"] == 200
    assert "canary" not in diagnostic_log.read_text()


@pytest.mark.parametrize("backend", ["codex", "siwc"])
def test_cache_http_real_provider_boundary(diagnostic_log, monkeypatch, backend):
    from .test_provider import _BufferedUpstreamResponse, _test_token

    provider = ChatGPTOAuthProvider(backend=backend, siwc_transport="http")
    payload = {"model": "gpt-6.1-terra", "store": False, "stream": True,
               "prompt_cache_key": "session-canary",
               "input": [{"type": "message", "role": "user", "content": "prompt-canary"}]}
    response = _BufferedUpstreamResponse(b'data: {"type":"response.completed","response":{}}\n\n')
    response.headers = {"x-request-id": "req_boundary123"}
    response.status = 200
    captured = []

    def open_request(request, **kwargs):
        assert request.data == json.dumps(payload).encode("utf-8")
        captured.append(request)
        return response

    token = _test_token()
    token.oauth_profile = backend
    monkeypatch.setattr(provider, "_token_for_request", lambda: token)
    monkeypatch.setattr(provider, "_open_request", open_request)
    try:
        assert list(provider._post_sse("/responses", payload)) == [
            {"type": "response.completed", "response": {}}
        ]
    finally:
        provider.close()
    data = records(diagnostic_log)
    assert len(captured) == 1
    assert [r["stage"] for r in data] == [
        "cache_request", "cache_http_request", "cache_http_response", "cache_response"
    ]
    assert len({r["payload_hash"] for r in data}) == 1
    assert data[2]["status"] == 200 and data[2]["upstream_request_id"] == "req_boundary123"
    assert "canary" not in diagnostic_log.read_text()


def test_cache_observation_forwards_close_and_handles_json_surrogates(diagnostic_log):
    closed = []

    def events():
        try:
            yield {"type": "response.created"}
        finally:
            closed.append(True)

    upstream = events()
    iterator = cache_diagnostics.observe_events(
        upstream, {"input": [{"role": "user", "content": "\ud800"}]}, backend="codex", transport="http",
    )
    next(iterator)
    iterator.close()
    assert closed == [True]
    assert records(diagnostic_log)[-1]["reason"] == "interrupted"


def test_cache_websocket_fingerprints_actual_wire_and_channel(diagnostic_log, monkeypatch):
    import hashlib

    sock = FakeSocket([{"type": "response.completed", "response": {"output": []}}])
    sent = []

    def send(body):
        sent.append(body)
        FakeSocket.send(sock, body)

    monkeypatch.setattr(sock, "send", send)
    monkeypatch.setattr(siwc_transport, "connect", lambda *a, **kw: sock)
    payload = {"model": "gpt-6-luna", "stream": True, "store": False,
               "input": [{"role": "user", "content": "prompt-canary"}]}
    pool = SIWCWebSocketPool()
    try:
        list(pool.events(payload, account_id="account-canary", access_token="auth-canary", timeout=1))
    finally:
        pool.close()
    expected = {k: v for k, v in payload.items() if k != "stream"}
    expected["type"] = "response.create"
    assert sent == [json.dumps(expected)]
    wire = next(r for r in records(diagnostic_log) if r["stage"] == "cache_websocket_request")
    assert wire["wire_hash"] == hashlib.sha256(sent[0].encode()).hexdigest()[:32]
    assert any(r.get("attempt_id") == wire["attempt_id"] and r.get("channel_id") == wire["channel_id"]
               and r["stage"] == "upstream_start" for r in records(diagnostic_log))
    assert "canary" not in diagnostic_log.read_text()
