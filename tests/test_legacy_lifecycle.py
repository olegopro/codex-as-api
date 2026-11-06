from __future__ import annotations

import ipaddress
import json
import ssl
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anyio
import httpx
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from starlette.responses import StreamingResponse

from codex_as_api import cache_diagnostics, diagnostics, server
from codex_as_api import provider as provider_module
from codex_as_api.auth import ChatGPTOAuthUpstreamError

from .test_diagnostics import diagnostic_log as diagnostic_log
from .test_diagnostics import records
from .test_provider import _test_token


@contextmanager
def upstream(monkeypatch, serve, *, timeout=1, routing="observe", before_headers=None, tls=None):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if before_headers is not None and not before_headers(self):
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("x-codex-turn-state", "route-canary")
            self.end_headers()
            try:
                serve(self, body)
            except (OSError, AssertionError):
                return

        def log_message(self, *args):
            pass

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    if tls is not None:
        backend.socket = tls[0].wrap_socket(backend.socket, server_side=True)
    backend.daemon_threads = True
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()
    scheme = "https" if tls is not None else "http"
    client = provider_module.ChatGPTOAuthProvider(base_url=f"{scheme}://127.0.0.1:{backend.server_port}",
                                   timeout=timeout, legacy_routing=routing)
    if tls is not None:
        client._http_client = httpx.Client(verify=tls[1], trust_env=False)
    monkeypatch.setattr(client, "_token_for_request", _test_token)
    monkeypatch.setattr(client, "_resolve_client_version", lambda: "0.166.0")
    try:
        yield client
    finally:
        client.close()
        if client._http_client is not None:
            client._http_client.close()
        backend.shutdown()
        backend.server_close()
        thread.join(2)


def send(handler, event):
    handler.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
    handler.wfile.flush()


def test_reader_separates_upstream_read_wait_from_consumer_delay(monkeypatch):
    clock = [0.0]
    lines = iter((b"data: {}\n", b": keepalive\n"))

    class Response:
        def readline(self):
            clock[0] += .125
            return next(lines)

    monkeypatch.setattr(provider_module.time, "monotonic", lambda: clock[0])
    reader = provider_module._LegacySSEReader(Response(), 10, started=0)
    assert reader.readline() == b"data: {}\n"
    clock[0] += .45
    assert reader.readline() == b": keepalive\n"
    assert reader.timings() == {"max_read_wait_ms": 125, "max_consumer_gap_ms": 450, "last_keepalive_ms": 700}
    reader.close()


@pytest.fixture()
def tls_pair(tmp_path):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "loopback-test")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=1))
            .add_extension(x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), False)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = tmp_path / "loopback.pem", tmp_path / "loopback.key"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                          serialization.NoEncryption()))
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert_path, key_path)
    return context, ssl.create_default_context(cafile=str(cert_path))


@pytest.mark.parametrize("use_tls", [False, True])
def test_asgi_disconnect_closes_blocking_legacy_socket_and_preserves_early_id(
    monkeypatch, diagnostic_log, tls_pair, use_tls,
):
    closed = threading.Event()
    calls = []

    def serve(handler, body):
        calls.append(body)
        send(handler, {"type": "response.created", "response": {"id": "resp_early"}})
        handler.connection.settimeout(3)
        if handler.connection.recv(1) == b"":
            closed.set()

    with upstream(monkeypatch, serve, timeout=2, tls=tls_pair if use_tls else None) as client:
        control = provider_module._RequestCancellation()
        headers = provider_module._TransportHeaders()
        headers.cancel = control

        async def exercise():
            entered = anyio.Event()

            def chunks():
                for event in client._post_sse("/responses", {"model": "gpt-6-luna"}, extra_headers=headers):
                    yield json.dumps(event)

            response = StreamingResponse(server._anthropic_heartbeat(chunks(), cancel=control))

            async def receive():
                await entered.wait()
                await anyio.sleep(0.02)
                return {"type": "http.disconnect"}

            async def send_message(message):
                if message["type"] == "http.response.body":
                    entered.set()

            scope = {"type": "http", "method": "POST", "path": "/v1/messages",
                     "headers": [], "asgi": {"spec_version": "2.3"}}
            with anyio.fail_after(1):
                await diagnostics.LifecycleMiddleware(response)(scope, receive, send_message)

        anyio.run(exercise)
        assert closed.wait(0.5)
        assert len(calls) == 1  # Cancelling an in-flight generation must not retry it.
    data = records(diagnostic_log)
    stages = [r["stage"] for r in data]
    assert stages.index("client_disconnect_observed") < stages.index("cache_http_closed")
    assert any(r["stage"] == "transport_cancelled" and r["socket_shutdown"] for r in data)
    assert next(r for r in data if r["stage"] == "cache_http_closed")["eof_kind"] == "cancelled"
    result = next(r for r in data if r["stage"] == "cache_response")
    assert result["upstream_response_id"] == "resp_early" and not result["usage_available"]
    assert "input_tokens" not in result and "route-canary" not in diagnostic_log.read_text()


def test_keepalive_bytes_do_not_defeat_sse_event_timeout(monkeypatch, diagnostic_log):
    def serve(handler, body):
        for _ in range(100):
            handler.wfile.write(b": ping\n\n")
            handler.wfile.flush()
            time.sleep(0.01)

    with upstream(monkeypatch, serve, timeout=0.08) as client:
        started = time.monotonic()
        with pytest.raises(ChatGPTOAuthUpstreamError):
            list(client._post_sse("/responses", {"model": "gpt-6-luna"}))
        assert time.monotonic() - started < 0.4
    result = next(r for r in records(diagnostic_log) if r["stage"] == "cache_http_closed")
    assert result["eof_kind"] == "event_timeout" and result["keepalive_lines"] > 0


def test_disconnect_while_waiting_for_http_headers_also_stops_upstream(monkeypatch, diagnostic_log):
    entered, closed, finished = threading.Event(), threading.Event(), threading.Event()
    errors = []

    def before_headers(handler):
        entered.set()
        handler.connection.settimeout(2)
        if handler.connection.recv(1) == b"":
            closed.set()
        return False

    with upstream(monkeypatch, lambda *_: None, timeout=2, before_headers=before_headers) as client:
        headers = provider_module._TransportHeaders()
        headers.cancel = provider_module._RequestCancellation()

        def consume():
            try:
                list(client._post_sse("/responses", {"model": "gpt-6-luna"}, extra_headers=headers))
            except ChatGPTOAuthUpstreamError as exc:
                errors.append(exc)
            finally:
                finished.set()

        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        assert entered.wait(1)
        headers.cancel.cancel()
        assert finished.wait(1) and closed.wait(0.5), [
            (r["stage"], r.get("socket_shutdown"), r.get("eof_kind")) for r in records(diagnostic_log)]
        worker.join(1)
        assert errors and errors[0].status == 502


def test_sse_events_reset_idle_deadline_and_clean_eof_is_distinct(monkeypatch, diagnostic_log):
    def serve(handler, body):
        for _ in range(4):
            send(handler, {"type": "response.in_progress"})
            time.sleep(0.03)
        send(handler, {"type": "response.created", "response": {"id": "resp_without_terminal"}})

    with upstream(monkeypatch, serve, timeout=0.07) as client:
        assert len(list(client._post_sse("/responses", {"model": "gpt-6-luna"}))) == 5
    data = records(diagnostic_log)
    assert next(r for r in data if r["stage"] == "cache_http_closed")["eof_kind"] == "clean_eof"
    result = next(r for r in data if r["stage"] == "cache_response")
    assert result["upstream_response_id"] == "resp_without_terminal"
    assert result["reason"] == "upstream_incomplete" and result["event_count"] == 5


def test_opt_in_routing_headers_and_turn_reset_reach_wire_without_mutating_body(monkeypatch):
    captured = []

    def serve(handler, body):
        captured.append((dict(handler.headers), body))
        send(handler, {"type": "response.completed", "response": {"output": [{"role": "assistant", "content": "OK"}]}})

    first = {"model": "gpt-6-luna", "prompt_cache_key": "session", "input": [{"role": "user", "content": "task"}]}
    second = {**first, "input": [*first["input"], {"role": "assistant", "content": "OK"},
                                 {"role": "user", "content": "new task"}]}
    with upstream(monkeypatch, serve, routing="native") as client:
        list(client._post_sse("/responses", first))
        list(client._post_sse("/responses", second))
    a, b = [{k.lower(): v for k, v in h.items()} for h, _ in captured]
    assert a["session-id"] == b["session-id"] == "session"
    assert a["thread-id"] == b["thread-id"] == a["x-client-request-id"]
    assert a["x-codex-routing-hint"] == b["x-codex-routing-hint"] == "model=gpt-6-luna"
    assert "x-codex-turn-state" not in a and "x-codex-turn-state" not in b
    assert [body for _, body in captured] == [first, second]


def test_routing_headers_are_observed_without_logging_values(diagnostic_log):
    headers = {"X-Codex-Turn-State": "route-canary", "X-Oai-Request-Id": "req_support123",
               "Cf-Ray": "1234567890abcdef-LHR", "authorization": "auth-canary"}
    cache_diagnostics.http_response({}, 200, headers, 0)
    cache_diagnostics.http_request({}, headers, b"{}", 0)
    response, request = records(diagnostic_log)
    assert response["turn_state_present"] and request["turn_state_present"]
    assert response["upstream_request_id"] == "req_support123"
    assert response["cf_ray"] == "1234567890abcdef-LHR"
    assert response["turn_state_hash"] == request["turn_state_hash"]
    assert "canary" not in diagnostic_log.read_text()


@pytest.mark.parametrize("routing", ["observe", "shadow", "turn-state", "native"])
def test_native_turn_state_echo_is_limited_to_tool_continuation(monkeypatch, diagnostic_log, routing):
    captured = []
    call = {"type": "function_call", "call_id": "c1", "name": "Read", "arguments": "{}"}
    answer = {"role": "assistant", "content": "done"}

    def serve(handler, body):
        captured.append({k.lower(): v for k, v in handler.headers.items()})
        output = [call] if len(captured) == 1 else [answer]
        send(handler, {"type": "response.completed", "response": {"output": output}})

    first = {"model": "gpt-6-luna", "prompt_cache_key": "s", "input": [{"role": "user", "content": "task"}]}
    second = {**first, "input": [*first["input"], call,
                                 {"type": "function_call_output", "call_id": "c1", "output": "value"}]}
    third = {**first, "input": [*second["input"], answer, {"role": "user", "content": "next"}]}
    with upstream(monkeypatch, serve, routing=routing) as client:
        for body in (first, second, third):
            list(client._post_sse("/responses", body))
    assert "x-codex-turn-state" not in captured[0]
    if routing in {"native", "turn-state"}:
        assert captured[1]["x-codex-turn-state"] == "route-canary"
    else:
        assert "x-codex-turn-state" not in captured[1]
    assert "x-codex-turn-state" not in captured[2]
    assert len({h["thread-id"] for h in captured}) == 1
    if routing != "native":
        assert all(h["thread-id"] == h["session-id"] == h["x-client-request-id"] == "s" for h in captured)
        assert all("x-codex-routing-hint" not in h for h in captured)
    requests = [r for r in records(diagnostic_log) if r["stage"] == "cache_http_request"]
    if routing != "observe":
        assert [r["lease_decision"] for r in requests] == ["new_no_anchor", "reused_replay", "reused_replay"]
        assert [r["new_turn"] for r in requests] == [True, False, True]
        assert [r["would_echo"] for r in requests] == [False, True, False]
    assert "route-canary" not in diagnostic_log.read_text()


@pytest.mark.parametrize("failure,expected", [("timeout", "socket_timeout"), ("chunked", "http_protocol_error"),
                                              ("tls", "ssl_eof")])
def test_httpx_error_causes_survive_transport_wrapping(monkeypatch, diagnostic_log, failure, expected):
    errors = {"timeout": httpx.ReadTimeout("timeout-canary"),
              "chunked": httpx.RemoteProtocolError("chunk-canary"), "tls": httpx.ReadError("tls-canary")}
    errors["tls"].__cause__ = ssl.SSLEOFError("ssl-canary")

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield b'data: {"type":"response.created","response":{"id":"resp_before_error"}}\n\n'
            raise errors[failure]

    with httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, stream=Stream()))) as http:
        client = provider_module.ChatGPTOAuthProvider(http_client=http)
        monkeypatch.setattr(client, "_token_for_request", _test_token)
        monkeypatch.setattr(client, "_resolve_client_version", lambda: "0.166.0")
        headers = provider_module._TransportHeaders()
        headers.cancel = provider_module._RequestCancellation()
        with pytest.raises(ChatGPTOAuthUpstreamError):
            list(client._post_sse("/responses", {"model": "gpt-6-luna"}, extra_headers=headers))
    result = next(r for r in records(diagnostic_log) if r["stage"] == "cache_http_closed")
    assert result["eof_kind"] == expected and "canary" not in diagnostic_log.read_text()


def test_final_unauthorized_response_retains_upstream_error_code(monkeypatch):
    attempts = []

    def fail(request):
        attempts.append(request)
        return httpx.Response(401, json={"error": {"code": "subscription_sharing_usage_limit_exceeded",
                                                   "message": "denied"}})

    with httpx.Client(transport=httpx.MockTransport(fail)) as http:
        client = provider_module.ChatGPTOAuthProvider(http_client=http)
        monkeypatch.setattr(client, "_token_for_request", _test_token)
        monkeypatch.setattr(client, "_resolve_client_version", lambda: "0.166.0")
        monkeypatch.setattr(client, "_refresh_after_unauthorized", lambda token: token)
        headers = provider_module._TransportHeaders()
        headers.cancel = provider_module._RequestCancellation()
        with pytest.raises(ChatGPTOAuthUpstreamError) as caught:
            list(client._post_sse("/responses", {"model": "gpt-6-luna"}, extra_headers=headers))
    assert len(attempts) == 2 and caught.value.code == "subscription_sharing_usage_limit_exceeded"
