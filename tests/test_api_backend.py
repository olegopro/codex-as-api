from __future__ import annotations

import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from codex_as_api import siwc_cli
from codex_as_api.auth import ChatGPTOAuthMissingError, ChatGPTOAuthUpstreamError
from codex_as_api.messages import Message, MessageRole
from codex_as_api.model_capabilities import parse_model_catalog
from codex_as_api.provider import (
    API_BASE_URL,
    DEFAULT_API_MAX_OUTPUT_TOKENS,
    ChatGPTOAuthProvider,
    _api_key_token,
    resolve_backend,
)

KEY = "sk-test-not-a-real-key"
PUBLIC_LIST = {
    "object": "list",
    "data": [
        {"id": "gpt-6-luna", "object": "model", "created": 1, "owned_by": "openai"},
        {"id": "text-embedding-3-small", "object": "model", "created": 1, "owned_by": "openai"},
        {"id": "gpt-6.1-sol", "object": "model", "created": 1, "owned_by": "openai"},
    ],
}


@pytest.fixture(autouse=True)
def _api_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    for name in ("CODEX_AS_API_PROMPT_CACHE_TTL", "CODEX_AS_API_MAX_OUTPUT_TOKENS"):
        monkeypatch.delenv(name, raising=False)


def _messages() -> list[Message]:
    return [Message(role=MessageRole.SYSTEM, content="system"), Message(role=MessageRole.USER, content="hello")]


def _snapshot():
    now = time.time()
    return parse_model_catalog(PUBLIC_LIST, key=("api-key-test", API_BASE_URL, "api"), etag=None,
                               fetched_at=now, expires_at=now + 3600, backend="api")


def test_backend_accepts_api_and_defaults_to_public_api():
    assert resolve_backend("api") == "api"
    with pytest.raises(ValueError, match="codex, siwc и api"):
        resolve_backend("openai")
    provider = ChatGPTOAuthProvider(backend="api")
    assert provider.base_url == API_BASE_URL
    assert provider.auth_json_path is None
    assert provider.prompt_cache_ttl == "12h"
    assert provider.max_output_tokens == DEFAULT_API_MAX_OUTPUT_TOKENS == 128000


def test_cache_ttl_and_output_ceiling_are_configurable(monkeypatch):
    monkeypatch.setenv("CODEX_AS_API_PROMPT_CACHE_TTL", "30m")
    monkeypatch.setenv("CODEX_AS_API_MAX_OUTPUT_TOKENS", "0")
    provider = ChatGPTOAuthProvider(backend="api")
    assert (provider.prompt_cache_ttl, provider.max_output_tokens) == ("30m", 0)
    for name, value in (("CODEX_AS_API_PROMPT_CACHE_TTL", "24h"), ("CODEX_AS_API_MAX_OUTPUT_TOKENS", "15"),
                        ("CODEX_AS_API_MAX_OUTPUT_TOKENS", "lots")):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError):
            ChatGPTOAuthProvider(backend="api")
        monkeypatch.delenv(name)


def test_key_comes_only_from_environment_and_is_not_echoed(monkeypatch):
    token = _api_key_token()
    assert token.access_token == KEY and token.oauth_profile == "api"
    assert KEY not in token.account_id and token.account_id.startswith("api-key-")
    assert ChatGPTOAuthProvider(backend="api")._headers() == {  # noqa: SLF001
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ChatGPTOAuthMissingError, match="OPENAI_API_KEY") as excinfo:
        _api_key_token()
    assert KEY not in str(excinfo.value)


def test_public_model_list_becomes_a_catalog_without_hidden_models():
    snapshot = _snapshot()
    luna = snapshot.model("gpt-6-luna")
    assert {"low", "high", "xhigh", "max"} <= set(luna.reasoning_effort_ids)
    assert luna.visibility == "list" and "image" in luna.input_modalities and not luna.use_responses_lite
    assert snapshot.model("text-embedding-3-small").visibility == "hide"
    assert snapshot.default_model().slug == "gpt-6-luna"
    assert [model.slug for model in snapshot.models] == ["gpt-6-luna", "text-embedding-3-small", "gpt-6.1-sol"]


def test_payload_carries_cache_policy_and_output_ceiling_instead_of_client_max_tokens(monkeypatch):
    provider = ChatGPTOAuthProvider(backend="api", model="gpt-6-luna")
    monkeypatch.setattr(provider, "get_model_catalog", _snapshot)
    payload = provider._responses_payload(_messages(), max_tokens=512, reasoning_effort="low")  # noqa: SLF001
    assert payload["prompt_cache_options"] == {"mode": "implicit", "ttl": "12h"}
    assert payload["max_output_tokens"] == 128000
    assert payload["store"] is False and payload["stream"] is True
    assert "client_metadata" not in payload and "prompt_cache_retention" not in payload
    explicit = provider._responses_payload(  # noqa: SLF001
        _messages(), prompt_cache_options={"mode": "explicit", "ttl": "30m", "prewarm": True})
    assert explicit["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m", "prewarm": True}
    with pytest.raises(Exception, match="ttl must be 30m or 12h"):
        provider._responses_payload(_messages(), prompt_cache_options={"ttl": "24h"})  # noqa: SLF001
    off = ChatGPTOAuthProvider(backend="api", model="gpt-6-luna", prompt_cache_ttl="off", max_output_tokens=0)
    monkeypatch.setattr(off, "get_model_catalog", _snapshot)
    bare = off._responses_payload(_messages())  # noqa: SLF001
    assert "prompt_cache_options" not in bare and "max_output_tokens" not in bare


def test_oauth_backends_still_reject_cache_options_and_max_tokens(model_catalog_snapshot, monkeypatch):
    provider = ChatGPTOAuthProvider(model="gpt-5.6-sol")
    monkeypatch.setattr(provider, "get_model_catalog", lambda: model_catalog_snapshot)
    with pytest.raises(Exception, match="prompt_cache_options is not supported"):
        provider._responses_payload(_messages(), prompt_cache_options={"ttl": "12h"})  # noqa: SLF001
    with pytest.raises(Exception, match="max_tokens is not supported"):
        provider._responses_payload(_messages(), max_tokens=512)  # noqa: SLF001


class _PublicApi:
    def __init__(self, status: int = 200) -> None:
        self.requests: queue.Queue[dict[str, Any]] = queue.Queue()
        self.status = status
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
                outer.requests.put({"method": "GET", "path": self.path, "headers": dict(self.headers.items())})
                encoded = json.dumps(PUBLIC_LIST).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.put({"method": "POST", "path": self.path, "headers": dict(self.headers.items()),
                                    "body": body})
                if outer.status != 200:
                    encoded = json.dumps({"error": {"message": f"Incorrect API key provided: {KEY}",
                                                    "type": "invalid_request_error"}}).encode()
                    self.send_response(outer.status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                    return
                events = [
                    {"type": "response.output_item.done", "item": {
                        "type": "message", "role": "assistant",
                        "content": [{"type": "output_text", "text": "api-ok"}]}},
                    {"type": "response.completed", "response": {
                        "id": "resp-api", "output": [], "prompt_cache_options": {"mode": "implicit", "ttl": "12h"},
                        "usage": {"input_tokens": 2048, "output_tokens": 2, "total_tokens": 2050,
                                  "input_tokens_details": {"cached_tokens": 1920}}}},
                ]
                encoded = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, _format: str, *args: object) -> None:
                del args

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> _PublicApi:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.thread.join(timeout=2)
        self.httpd.server_close()

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"


def test_wire_is_public_responses_with_bearer_key_only():
    with _PublicApi() as api:
        provider = ChatGPTOAuthProvider(backend="api", model="gpt-6-luna", base_url=api.base_url, timeout=5)
        result = provider.chat(_messages(), prompt_cache_key="session-key", subagent="worker")
        catalog = api.requests.get(timeout=2)
        post = api.requests.get(timeout=2)
    assert catalog["path"] == "/models"
    assert post["path"] == "/responses"
    headers = {k.lower(): v for k, v in post["headers"].items()}
    assert headers["authorization"] == f"Bearer {KEY}"
    for private in ("chatgpt-account-id", "originator", "version", "session-id", "thread-id",
                    "x-client-request-id", "x-openai-subagent", "x-codex-turn-state", "x-openai-fedramp"):
        assert private not in headers
    body = post["body"]
    assert body["prompt_cache_options"] == {"mode": "implicit", "ttl": "12h"}
    assert body["max_output_tokens"] == 128000
    assert body["prompt_cache_key"] == "session-key"
    assert "client_metadata" not in body
    assert result.content == "api-ok"
    assert result.usage is not None and result.usage.cached_tokens == 1920


def test_rejected_key_is_final_and_never_echoed():
    with _PublicApi(status=401) as api:
        provider = ChatGPTOAuthProvider(backend="api", model="gpt-6-luna", base_url=api.base_url, timeout=5)
        with pytest.raises(ChatGPTOAuthUpstreamError) as excinfo:
            provider.chat(_messages())
        seen = [api.requests.get(timeout=2) for _ in range(2)]
        assert api.requests.empty()
    assert [r["method"] for r in seen] == ["GET", "POST"]
    assert excinfo.value.status == 401
    assert KEY not in str(excinfo.value)


def test_cli_status_reports_only_key_presence(monkeypatch, capsys):
    assert siwc_cli.main(["status"], backend="api") == 0
    assert json.loads(capsys.readouterr().out) == {"auth_profile": "api", "state": "key_present"}
    monkeypatch.delenv("OPENAI_API_KEY")
    assert siwc_cli.main(["status"], backend="api") == 0
    assert json.loads(capsys.readouterr().out) == {"auth_profile": "api", "state": "key_missing"}
    assert siwc_cli.main(["login"], backend="api") == 2
    assert KEY not in capsys.readouterr().err


def test_health_reports_public_api_transport_and_cache_policy():
    from codex_as_api.server import _health_transport_fields

    fields = _health_transport_fields(ChatGPTOAuthProvider(backend="api"))
    assert {k: fields[k] for k in ("backend", "upstream_base_url", "upstream_transport",
                                   "prompt_cache_ttl", "max_output_tokens")} == {
        "backend": "api", "upstream_base_url": API_BASE_URL, "upstream_transport": "http",
        "prompt_cache_ttl": "12h", "max_output_tokens": 128000}
