from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest

import codex_as_api.provider as provider_module
from codex_as_api.anthropic_adapter import anthropic_request_to_internal, internal_response_to_anthropic
from codex_as_api.auth import (
    ChatGPTOAuthCatalogUnavailableError,
    ChatGPTOAuthInvalidRequestError,
    ChatGPTOAuthMissingError,
    ChatGPTOAuthModelNotFoundError,
    ChatGPTOAuthProtocolError,
    ChatGPTOAuthRefreshError,
    ChatGPTOAuthUpstreamError,
)
from codex_as_api.messages import Message, MessageRole, ToolSchema
from codex_as_api.model_capabilities import parse_model_catalog
from codex_as_api.provider import SIWC_BASE_URL, SIWC_TOOL_NAMESPACE, ChatGPTOAuthProvider
from codex_as_api.reasoning_signature import encode_reasoning_signature


def _catalog() -> dict:
    return {
        "models": [
            {
                "slug": "synthetic-first",
                "display_name": "Первая",
                "visibility": "list",
                "priority": 20,
                "context_window": 100_000,
                "input_modalities": ["text", "image"],
                "supported_reasoning_levels": [{"effort": "low", "description": "Низкий"}],
                "default_reasoning_level": "low",
                "support_verbosity": True,
                "default_verbosity": "low",
            },
            {"slug": "synthetic-second", "display_name": "Вторая", "visibility": "list", "priority": 0},
            {"slug": "synthetic-hidden", "display_name": "Скрытая", "visibility": "hide"},
        ],
        "access_token": "непубликуемое-поле",
    }


def _token(profile: str = "siwc") -> SimpleNamespace:
    return SimpleNamespace(
        oauth_profile=profile,
        account_id="synthetic-subject",
        access_token="synthetic-access",
        refresh_token="synthetic-refresh",
        id_token="synthetic-id",
        fedramp=True,
    )


def _text_item(text: str = "Ответ") -> dict:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _completed(output: list[dict] | None = None) -> dict:
    return {
        "type": "response.completed",
        "response": {
            "id": "resp_synthetic",
            "status": "completed",
            "output": [_text_item()] if output is None else output,
            "usage": {
                "input_tokens": 20,
                "output_tokens": 3,
                "total_tokens": 23,
                "input_tokens_details": {"cached_tokens": 12},
            },
        },
    }


def _sse(events: list[dict]) -> bytes:
    return "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events).encode()


@pytest.fixture()
def siwc_provider(monkeypatch):
    clients = []
    monkeypatch.setattr(provider_module, "token_for_request", lambda _path, **_kwargs: _token())

    def forbidden(*_args, **_kwargs):
        pytest.fail("Реальный HTTP или приватный транспорт недопустим")

    monkeypatch.setattr(provider_module, "_urlopen_no_redirect", forbidden)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)

    def factory(*, events=None, document=None, handler=None, model=None):
        requests = []

        def respond(request):
            requests.append(request)
            if handler is not None:
                return handler(request)
            if request.method == "GET":
                return httpx.Response(200, json=_catalog() if document is None else document)
            return httpx.Response(200, content=_sse([_completed()] if events is None else events))

        client = httpx.Client(
            transport=httpx.MockTransport(respond),
            headers={"ChatGPT-Account-Id": "synthetic-unused-account"},
            auth=("synthetic-unused-user", "synthetic-unused-password"),
        )
        clients.append(client)
        provider = ChatGPTOAuthProvider(backend="siwc", http_client=client, model=model)
        return provider, requests

    yield factory
    for client in clients:
        client.close()


def _messages() -> list[Message]:
    return [Message(role=MessageRole.SYSTEM, content="Правила"), Message(role=MessageRole.USER, content="Вопрос")]


def test_siwc_payload_catalog_headers_and_usage(siwc_provider, monkeypatch):
    monkeypatch.setenv("CODEX_AS_API_CODEX_CLI_VERSION", "не-используется")
    monkeypatch.setenv("CODEX_AS_API_RESPONSES_LITE", "on")
    monkeypatch.setenv("CODEX_AS_API_CODEX_METADATA", "on")
    provider, requests = siwc_provider()
    response = provider.chat(_messages(), client_metadata={"session_id": "не-должно-уйти"})
    assert response.content == "Ответ"
    assert response.usage.cached_tokens == 12
    assert [str(request.url) for request in requests] == [f"{SIWC_BASE_URL}/models", f"{SIWC_BASE_URL}/responses"]
    for request in requests:
        headers = dict(request.headers)
        assert headers.pop("authorization") == "Bearer synthetic-access"
        assert headers.pop("content-type") == "application/json"
        assert set(headers) <= {"host", "content-length"}
    payload = json.loads(requests[-1].content)
    assert payload["model"] == "synthetic-first"
    assert payload["instructions"] == "Правила"
    assert payload["store"] is False and payload["stream"] is True
    assert isinstance(payload["input"], list)
    assert payload["reasoning"] == {"effort": "low", "summary": "auto"}
    assert payload["include"] == ["reasoning.encrypted_content"]
    assert not ({
        "background", "conversation", "max_output_tokens", "max_tool_calls", "metadata", "moderation",
        "multi_agent", "prompt", "prompt_cache_retention", "safety_identifier", "temperature",
        "top_logprobs", "top_p", "truncation", "user", "previous_response_id", "client_metadata",
    } & payload.keys())
    result = internal_response_to_anthropic(response, "claude-synthetic", "msg_synthetic")
    assert result["usage"]["input_tokens"] == 8
    assert result["usage"]["cache_read_input_tokens"] == 12


@pytest.mark.parametrize("base_url", ["https://example.test/v1", "http://api.openai.com/v1", SIWC_BASE_URL + "/"])
def test_siwc_rejects_other_upstream(base_url):
    with pytest.raises(ValueError, match="SIWC"):
        ChatGPTOAuthProvider(backend="siwc", base_url=base_url)


def test_backend_selection_is_explicit_and_legacy_is_default(monkeypatch):
    monkeypatch.delenv("CODEX_AS_API_BACKEND", raising=False)
    assert ChatGPTOAuthProvider().backend == "codex"
    monkeypatch.setenv("CODEX_AS_API_BACKEND", "siwc")
    assert ChatGPTOAuthProvider().base_url == SIWC_BASE_URL
    assert ChatGPTOAuthProvider(backend="codex").base_url == provider_module.CHATGPT_OAUTH_DEFAULT_BASE_URL
    monkeypatch.setenv("CODEX_AS_API_BACKEND", "ошибка")
    with pytest.raises(ValueError):
        ChatGPTOAuthProvider()


def test_siwc_rejects_legacy_credentials_before_transport(siwc_provider, monkeypatch):
    provider, requests = siwc_provider()
    expected = []

    def legacy_token(_path, *, expected_profile):
        expected.append(expected_profile)
        return _token("codex")

    monkeypatch.setattr(provider_module, "token_for_request", legacy_token)
    with pytest.raises(ChatGPTOAuthMissingError):
        provider.get_model_catalog()
    assert expected == ["siwc"]
    assert not requests


def test_siwc_rechecks_profile_before_inference_and_refresh(siwc_provider, monkeypatch):
    provider, requests = siwc_provider()
    provider.get_model_catalog()
    monkeypatch.setattr(provider_module, "token_for_request", lambda *_args, **_kwargs: _token("codex"))
    with pytest.raises(ChatGPTOAuthMissingError):
        provider.chat(_messages())
    assert len(requests) == 1


def test_siwc_catalog_is_account_specific_without_static_fallbacks(siwc_provider):
    provider, _ = siwc_provider()
    snapshot = provider.get_model_catalog()
    assert [model.slug for model in snapshot.models] == [model["slug"] for model in _catalog()["models"]]
    assert snapshot.default_model().slug == "synthetic-first"
    assert snapshot.account_id == "synthetic-subject"
    assert snapshot.model("synthetic-first").context_window == 100_000
    assert snapshot.model("synthetic-second").context_window is None
    assert not snapshot.model("synthetic-first").use_responses_lite
    with pytest.raises(ChatGPTOAuthModelNotFoundError):
        provider.resolve_model("gpt-6-chat-thinking")
    assert provider.resolve_model("claude-synthetic", anthropic_facade=True)[1].slug == "synthetic-first"
    assert provider.resolve_model("synthetic-second")[1].slug == "synthetic-second"


def test_siwc_explicit_configuration_never_substitutes_a_model(siwc_provider):
    provider, _ = siwc_provider(model="synthetic-second")
    assert provider.resolve_model("claude-synthetic", anthropic_facade=True)[1].slug == "synthetic-second"
    provider.model = "absent-model"
    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        provider.resolve_model()


@pytest.mark.parametrize("document", [{"data": []}, {"models": []}, {"models": [{"slug": "incomplete"}]}])
def test_siwc_catalog_never_recovers_with_hidden_models(siwc_provider, document):
    provider, _ = siwc_provider(document=document)
    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        provider.resolve_model()


def test_siwc_does_not_promote_known_hidden_slug():
    snapshot = parse_model_catalog(
        {"models": [{"slug": "gpt-reserve", "display_name": "Скрытая", "visibility": "hide"}]},
        key=("subject", SIWC_BASE_URL, "siwc"), etag=None, fetched_at=1, expires_at=2, backend="siwc",
    )
    assert snapshot.models[0].visibility == "hide"
    with pytest.raises(ChatGPTOAuthCatalogUnavailableError):
        snapshot.default_model()


@pytest.mark.parametrize("ending", [
    [],
    [{"type": "response.failed", "response": {"error": {"message": "synthetic-access"}}}],
    [{"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}}],
    [{"type": "error", "message": "synthetic-access"}],
])
def test_siwc_failed_or_interrupted_stream_is_never_success(siwc_provider, ending):
    provider, _ = siwc_provider(events=[{"type": "response.output_text.delta", "delta": "Часть"}, *ending])
    events = []
    with pytest.raises((ChatGPTOAuthProtocolError, ChatGPTOAuthUpstreamError)) as caught:
        events.extend(provider.chat_stream(_messages()))
    assert not any(event["type"] == "finish" for event in events)
    assert "synthetic-access" not in str(caught.value)


def test_siwc_error_after_completed_is_not_success(siwc_provider):
    provider, _ = siwc_provider(events=[_completed(), {"type": "error", "message": "Ошибка хвоста"}])
    with pytest.raises(ChatGPTOAuthUpstreamError):
        provider.chat(_messages())


@pytest.mark.parametrize("arguments,status", [("{", "completed"), ("{}", "incomplete")])
def test_siwc_never_yields_truncated_tool_calls(siwc_provider, arguments, status):
    item = {"type": "function_call", "name": "read", "namespace": SIWC_TOOL_NAMESPACE,
            "arguments": arguments, "call_id": "call_synthetic", "status": status}
    provider, _ = siwc_provider(events=[{"type": "response.output_item.done", "item": item}, _completed([item])])
    events = []
    with pytest.raises(ChatGPTOAuthProtocolError):
        events.extend(provider.chat_stream(
            _messages(), tools=[ToolSchema(name="read", description="Чтение", parameters={})],
        ))
    assert not any(event["type"] in {"tool_call", "finish"} for event in events)


def test_siwc_done_tool_without_completed_is_not_yielded(siwc_provider):
    item = {"type": "function_call", "name": "read", "namespace": SIWC_TOOL_NAMESPACE,
            "arguments": "{}", "call_id": "call_synthetic"}
    provider, _ = siwc_provider(events=[{"type": "response.output_item.done", "item": item}])
    events = []
    with pytest.raises(ChatGPTOAuthProtocolError):
        events.extend(provider.chat_stream(
            _messages(), tools=[ToolSchema(name="read", description="Чтение", parameters={})],
        ))
    assert not any(event["type"] == "tool_call" for event in events)


def test_siwc_empty_completed_output_keeps_streamed_items(siwc_provider):
    item = {"type": "function_call", "name": "read", "namespace": SIWC_TOOL_NAMESPACE,
            "arguments": "{}", "call_id": "call_synthetic", "status": "completed"}
    provider, _ = siwc_provider(events=[{"type": "response.output_item.done", "item": item}, _completed([])])
    events = list(provider.chat_stream(
        _messages(), tools=[ToolSchema(name="read", description="Чтение", parameters={})],
    ))
    assert [event["id"] for event in events if event["type"] == "tool_call"] == ["call_synthetic"]
    assert events[-1]["type"] == "finish"


def test_siwc_namespace_function_and_encrypted_reasoning_roundtrip(siwc_provider):
    original_name = "synthetic.read-file"
    wire_name = provider_module._siwc_wire_name(original_name)
    reasoning = {"type": "reasoning", "id": "rs_fixture", "summary": [], "encrypted_content": "synthetic-cipher"}
    call = {"type": "function_call", "name": wire_name, "namespace": SIWC_TOOL_NAMESPACE,
            "call_id": "call_fixture", "arguments": '{"path":"fixture.txt"}', "status": "completed"}
    provider, requests = siwc_provider(events=[_completed([reasoning, call])])
    tools = [ToolSchema(name=original_name, description="Чтение", parameters={"type": "object", "properties": {}})]
    response = provider.chat(_messages(), tools=tools)
    assert response.tool_calls[0].name == original_name
    assert response.tool_calls[0].id == "call_fixture"
    first_payload = json.loads(requests[-1].content)
    assert first_payload["tools"][0]["type"] == "namespace"
    assert isinstance(first_payload["tools"][0]["description"], str)
    assert first_payload["tools"][0]["description"].strip()
    assert first_payload["tools"][0]["tools"][0]["name"] == wire_name
    converted = internal_response_to_anthropic(
        response, "claude-synthetic", "msg_fixture", backend_model="synthetic-first",
    )
    messages, replay_tools, *_ = anthropic_request_to_internal(
        model="claude-synthetic",
        messages=[
            {"role": "user", "content": "Вопрос"},
            {"role": "assistant", "content": converted["content"]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_fixture", "content": "Готово"}]},
        ],
        tools=[{"name": original_name, "input_schema": {"type": "object", "properties": {}}}],
    )
    payload = provider.preflight_chat(messages, tools=replay_tools).payload
    assert reasoning in payload["input"]
    replay_call = next(item for item in payload["input"] if item["type"] == "function_call")
    assert replay_call["name"] == wire_name and replay_call["namespace"] == SIWC_TOOL_NAMESPACE
    assert replay_call["call_id"] == "call_fixture"
    assert payload["input"][-1] == {"type": "function_call_output", "call_id": "call_fixture", "output": "Готово"}
    assert "previous_response_id" not in payload


def test_siwc_synthetic_cipher_from_another_backend_is_not_rewritten(siwc_provider):
    provider, _ = siwc_provider()
    reasoning = {"type": "reasoning", "id": "rs_migrated", "summary": [], "encrypted_content": "opaque-fixture"}
    signature = encode_reasoning_signature("synthetic-first", reasoning)
    messages, *_ = anthropic_request_to_internal(
        model="claude-synthetic", messages=[
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": signature}]},
            {"role": "user", "content": "Продолжение"},
        ],
    )
    payload = provider.preflight_chat(messages).payload
    assert payload["input"][0] == reasoning


def test_siwc_local_previous_id_is_expanded_without_upstream_id(siwc_provider):
    provider, _ = siwc_provider()
    response = provider.chat(_messages())
    payload = provider.preflight_chat(
        [Message(role=MessageRole.USER, content="Продолжение")], previous_response_id=response.response_id,
    ).payload
    assert "previous_response_id" not in payload
    assert len(payload["input"]) == 3
    assert payload["input"][1] == _text_item()


def test_siwc_forced_function_uses_one_namespaced_tool(siwc_provider):
    provider, _ = siwc_provider()
    payload = provider.preflight_chat(
        _messages(), tools=[
            ToolSchema(name="read", description="Чтение", parameters={}),
            ToolSchema(name="write", description="Запись", parameters={}),
        ],
        tool_choice={"type": "function", "name": "read"}, parallel_tool_calls=True,
    ).payload
    assert payload["parallel_tool_calls"] is True
    assert payload["tool_choice"] == "required"
    assert [tool["name"] for tool in payload["tools"][0]["tools"]] == ["read"]


@pytest.mark.parametrize("operation", ["compact_messages", "generate_image"])
def test_siwc_unsupported_operations_do_not_touch_catalog_or_transport(siwc_provider, operation):
    provider, requests = siwc_provider()
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        getattr(provider, operation)(_messages() if operation == "compact_messages" else "Рисунок")
    assert not requests


def test_siwc_refresh_uses_profile_and_refuses_account_change(siwc_provider, monkeypatch):
    def handler(_request):
        return httpx.Response(401, text="synthetic-access")

    provider, requests = siwc_provider(handler=handler)
    refreshed = deepcopy(_token())
    refreshed.account_id = "another-subject"
    calls = []

    def refresh(token, *, expected_profile):
        calls.append((token.oauth_profile, expected_profile))
        return refreshed

    monkeypatch.setattr(provider_module, "refresh_after_unauthorized", refresh)
    with pytest.raises(ChatGPTOAuthRefreshError):
        provider.get_model_catalog()
    assert calls == [("siwc", "siwc")]
    assert len(requests) == 1


def test_siwc_redirects_are_not_followed(siwc_provider):
    provider, requests = siwc_provider(handler=lambda _request: httpx.Response(
        307, headers={"Location": "https://example.test/token-sink"},
    ))
    with pytest.raises(ChatGPTOAuthUpstreamError):
        provider.get_model_catalog()
    assert len(requests) == 1


def test_siwc_default_store_is_separate_from_legacy(monkeypatch, tmp_path):
    public_path = tmp_path / "synthetic-public.json"
    monkeypatch.setenv("CODEX_AS_API_AUTH_PATH", str(tmp_path / "synthetic-legacy.json"))
    monkeypatch.setenv("CODEX_AS_API_SIWC_AUTH_PATH", str(public_path))
    assert ChatGPTOAuthProvider(backend="siwc").auth_json_path == str(public_path)


@pytest.mark.parametrize("error_line", [b"error: synthetic-access\n\n", b"data: {\"error\":\"synthetic-access\"}\n\n"])
def test_siwc_explicit_transport_error_is_not_ignored(siwc_provider, error_line):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json=_catalog())
        return httpx.Response(200, content=_sse([_completed()]) + error_line)

    provider, _ = siwc_provider(handler=handler)
    with pytest.raises((ChatGPTOAuthProtocolError, ChatGPTOAuthUpstreamError)) as caught:
        provider.chat(_messages())
    assert "synthetic-access" not in str(caught.value)


def test_siwc_duplicate_call_ids_are_rejected(siwc_provider):
    item = {"type": "function_call", "name": "read", "namespace": SIWC_TOOL_NAMESPACE,
            "arguments": "{}", "call_id": "call_duplicate"}
    provider, _ = siwc_provider(events=[_completed([item, item])])
    events = []
    with pytest.raises(ChatGPTOAuthProtocolError):
        events.extend(provider.chat_stream(
            _messages(), tools=[ToolSchema(name="read", description="Чтение", parameters={})],
        ))
    assert not any(event["type"] == "tool_call" for event in events)


def test_siwc_failed_status_inside_completed_is_rejected(siwc_provider):
    event = _completed()
    event["response"]["status"] = "incomplete"
    provider, _ = siwc_provider(events=[event])
    with pytest.raises(ChatGPTOAuthProtocolError):
        provider.chat(_messages())


def test_siwc_mutated_base_url_cannot_receive_token(siwc_provider):
    provider, requests = siwc_provider()
    provider.base_url = "https://example.test"
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        provider.get_model_catalog()
    assert not requests
