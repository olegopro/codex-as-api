from __future__ import annotations

from copy import deepcopy

import pytest

from codex_as_api.anthropic_adapter import anthropic_request_to_internal, internal_response_to_anthropic
from codex_as_api.legacy_routing import LegacyRouting
from codex_as_api.provider import ChatGPTOAuthProvider

USER = {"role": "user", "content": "task"}
CALL = {"type": "function_call", "call_id": "call_1", "name": "Read", "arguments": "{}"}
RESULT = {"type": "function_call_output", "call_id": "call_1", "output": "result"}


def payload(*items, model="gpt-6-luna", session="session"):
    return {"model": model, "prompt_cache_key": session, "instructions": "rules", "input": list(items or [USER])}


def reserve(pool, body=None, *, account="account", owner="owner"):
    lease = pool.reserve(body or payload(), account, owner)
    assert lease is not None
    return lease


def complete(pool, lease, *output):
    lease.response_headers({"x-codex-turn-state": "sticky-token"})
    for item in output:
        lease.observe({"type": "response.output_item.done", "item": item})
    lease.observe({"type": "response.completed", "response": {"output": []}})
    pool.release(lease)


def test_token_provenance_survives_continuation_but_not_a_new_turn(monkeypatch):
    pool = LegacyRouting()
    import codex_as_api.legacy_routing as routing
    monkeypatch.setattr(routing, "diagnostic_context", lambda: {"http_id": "a" * 32})
    first = reserve(pool)
    complete(pool, first, CALL)
    second = reserve(pool, payload(USER, CALL, RESULT))
    assert second.token_origin == "a" * 32 and second.token_issued_at == first.token_issued_at
    second.response_headers({"x-codex-turn-state": "ignored-second-token"})
    assert second.turn_state == "sticky-token" and second.token_origin == first.token_origin
    complete(pool, second, {"role": "assistant", "content": "done"})
    third = reserve(pool, payload(USER, CALL, RESULT, {"role": "assistant", "content": "done"},
                                  {"role": "user", "content": "next"}))
    assert third.turn_state is None and third.token_origin is None and third.token_issued_at is None
    pool.release(third)


def test_successful_replay_does_not_compute_mismatch_details_under_lock(monkeypatch):
    from codex_as_api.legacy_routing import _Replay

    pool = LegacyRouting()
    first = reserve(pool)
    complete(pool, first, CALL)

    def unexpected(_items):
        raise AssertionError("mismatch-only hashing on a successful replay")

    monkeypatch.setattr(_Replay, "from_output", unexpected)
    continuation = reserve(pool, payload(USER, CALL, RESULT))
    assert continuation.decision == "reused_replay" and continuation.turn_state == "sticky-token"
    pool.release(continuation)


@pytest.mark.parametrize("kind", ["count", "order", "reasoning", "text", "call"])
def test_replay_mismatch_reports_only_the_changed_component(kind):
    pool = LegacyRouting()
    first = reserve(pool)
    output = [{"type": "reasoning", "id": "r1", "encrypted_content": "cipher", "summary": []},
              {"role": "assistant", "content": "answer"}, CALL, {**CALL, "call_id": "call_2"}]
    complete(pool, first, *output)
    changed = deepcopy(output)
    if kind == "count":
        changed.pop()
    elif kind == "order":
        changed[-2:] = reversed(changed[-2:])
    elif kind == "reasoning":
        changed[0]["encrypted_content"] = "different"
    elif kind == "text":
        changed[1]["content"] = "different"
    else:
        changed[2]["arguments"] = '{"changed":true}'
    second = reserve(pool, payload(USER, *changed, RESULT))
    assert second.decision == "new_replay_mismatch" and second.replay_mismatch_kind == kind
    assert second.turn_state is None
    pool.release(second)


def test_continuation_preserves_turn_but_new_user_resets_it():
    pool = LegacyRouting()
    first = reserve(pool)
    complete(pool, first, CALL)
    continuation = payload(USER, CALL, RESULT, {"role": "developer", "content": "late note"})
    second = reserve(pool, continuation)
    assert second.headers() == {"thread-id": first.entry.thread_id, "x-client-request-id": first.entry.thread_id,
                                "x-codex-turn-state": "sticky-token"}
    answer = {"role": "assistant", "content": "done"}
    complete(pool, second, answer)
    third = reserve(pool, payload(*continuation["input"], answer, {"role": "user", "content": "next task"}))
    assert third.entry is first.entry
    assert "x-codex-turn-state" not in third.headers()
    pool.release(third)


def test_parallel_identical_requests_never_share_a_history_or_turn_state():
    pool = LegacyRouting()
    a, b = reserve(pool), reserve(pool)
    assert a.entry.thread_id != b.entry.thread_id
    a.response_headers({"x-codex-turn-state": "private-turn"})
    assert "x-codex-turn-state" not in b.headers()
    pool.release(a)
    pool.release(b)
    # Ambiguous equal anchors are not resolved by selecting an arbitrary sibling.
    c = reserve(pool)
    assert c.entry.thread_id not in {a.entry.thread_id, b.entry.thread_id}
    pool.release(c)


@pytest.mark.parametrize("changed", ["model", "instructions", "service_tier"])
def test_effective_config_change_does_not_echo_old_turn_state(changed):
    pool = LegacyRouting()
    a = reserve(pool)
    complete(pool, a, CALL)
    body = payload(USER, CALL, RESULT)
    body[changed] = {"tools": [], "reasoning": {"effort": "high"}}.get(changed, "changed")
    b = reserve(pool, body)
    assert "x-codex-turn-state" not in b.headers()
    pool.release(b)


@pytest.mark.parametrize("changed", ["tools", "reasoning"])
def test_tool_loading_and_reasoning_updates_do_not_reset_turn(changed):
    pool = LegacyRouting()
    first = reserve(pool)
    complete(pool, first, CALL)
    body = payload(USER, CALL, RESULT)
    body[changed] = {"tools": [{"type": "function", "name": "new_tool"}],
                     "reasoning": {"effort": "high"}}[changed]
    next_step = reserve(pool, body)
    assert next_step.turn_state == "sticky-token"
    pool.release(next_step)


@pytest.mark.parametrize("change", ["account", "session", "first_user", "replay"])
def test_accounts_credentials_branches_and_output_replays_are_isolated(change):
    pool = LegacyRouting()
    a = reserve(pool)
    complete(pool, a, CALL)
    body = payload(USER, CALL, RESULT)
    kwargs = {}
    if change in {"account", "token"}:
        kwargs[change] = "other"
    elif change == "session":
        body["prompt_cache_key"] = "other"
    elif change == "first_user":
        body["input"][0] = {"role": "user", "content": "agent or BTW task"}
    else:
        body["input"][1] = {**CALL, "arguments": '{"different":true}'}
    b = reserve(pool, body, **kwargs)
    assert b.entry.thread_id != a.entry.thread_id
    assert "x-codex-turn-state" not in b.headers()
    pool.release(b)


def test_unknown_replay_and_compacted_history_do_not_reuse_turn():
    pool = LegacyRouting()
    a = reserve(pool)
    a.response_headers({"x-codex-turn-state": "sticky-token"})
    pool.release(a)
    b = reserve(pool, payload(USER, CALL, RESULT))
    assert b.entry.thread_id != a.entry.thread_id
    pool.release(b)
    c = reserve(pool, payload({"role": "user", "content": "compacted"}))
    assert c.entry.thread_id not in {a.entry.thread_id, b.entry.thread_id}
    pool.release(c)


def test_exact_retry_uses_first_token_and_request_payload_is_unchanged():
    pool = LegacyRouting()
    body = payload()
    original = deepcopy(body)
    a = reserve(pool, body)
    a.response_headers({"x-codex-turn-state": "first"})
    a.response_headers({"x-codex-turn-state": "second"})
    pool.release(a)
    b = reserve(pool, body)
    assert b.headers()["x-codex-turn-state"] == "first"
    assert body == original
    pool.release(b)


def test_idle_capacity_never_evicts_active_history():
    pool = LegacyRouting(capacity=1, idle_timeout=0)
    a, b = reserve(pool), reserve(pool)
    assert a.entry.busy and b.entry.busy
    pool.release(a)
    c = reserve(pool)
    assert b.entry.busy and c.entry.thread_id != b.entry.thread_id
    pool.release(b)
    pool.release(c)


def test_routing_is_opt_in_and_never_applied_to_siwc(monkeypatch):
    monkeypatch.delenv("CODEX_AS_API_LEGACY_ROUTING", raising=False)
    assert ChatGPTOAuthProvider()._legacy_routing is None
    assert ChatGPTOAuthProvider(legacy_routing="native")._legacy_routing is not None
    assert ChatGPTOAuthProvider(backend="siwc", legacy_routing="native")._legacy_routing is None
    with pytest.raises(ValueError):
        ChatGPTOAuthProvider(legacy_routing="invalid")


def test_owner_change_preserves_thread_identity_and_resets_turn_state():
    pool = LegacyRouting()
    a = reserve(pool)
    complete(pool, a, CALL)
    b = reserve(pool, payload(USER, CALL, RESULT), owner="new-login")
    assert b.entry.thread_id == a.entry.thread_id and b.decision == "reused_replay"
    assert "x-codex-turn-state" not in b.headers()
    pool.release(b)


def test_normal_thread_identity_survives_registry_restart_and_anchors_are_compact():
    pool = LegacyRouting()
    a = reserve(pool)
    complete(pool, a, CALL)
    b = reserve(LegacyRouting())
    assert b.entry.thread_id == a.entry.thread_id
    assert a.entry.anchors[0].size == 1 and isinstance(a.entry.anchors[0].prefix, str)


def test_branch_between_tool_steps_cannot_replace_main_turn_state():
    pool = LegacyRouting()
    main = reserve(pool)
    complete(pool, main, CALL)
    fork = reserve(pool, payload(USER, CALL, {"role": "user", "content": "BTW task"}))
    assert fork.new_turn and fork.turn_state is None
    fork.response_headers({"x-codex-turn-state": "fork-token"})
    fork.observe({"type": "response.completed", "response": {"output": []}})
    pool.release(fork)
    continuation = reserve(pool, payload(USER, CALL, RESULT))
    assert continuation.turn_state == "sticky-token"
    assert continuation.entry is main.entry
    pool.release(continuation)


def test_rejection_invalidates_chain_but_not_sibling_turn():
    pool = LegacyRouting()
    main = reserve(pool)
    complete(pool, main, CALL)
    retry = reserve(pool)
    assert retry.turn_state == "sticky-token"
    retry.reject_turn_state()
    retry.response_headers({"x-codex-turn-state": "ignored-after-rejection"})
    pool.release(retry)
    continuation = reserve(pool, payload(USER, CALL, RESULT))
    assert continuation.turn_state is None
    pool.release(continuation)


@pytest.mark.parametrize("value", ["", "bad\r\nheader", "a" * 8193, "non-ascii-\u00e9"])
def test_turn_state_header_validation(value):
    pool = LegacyRouting()
    lease = reserve(pool)
    lease.response_headers({"x-codex-turn-state": value})
    assert lease.turn_state is None
    pool.release(lease)


@pytest.mark.parametrize("mode", ["shadow", "turn-state", "native"])
@pytest.mark.parametrize("failure", [None, "reserve", "prepare", "http", "sse"])
def test_provider_shadow_and_echo_only_preserve_payload_and_owner_refresh(monkeypatch, mode, failure):
    import io
    import json
    import urllib.error

    from codex_as_api import cache_diagnostics
    from codex_as_api.auth import ChatGPTOAuthUpstreamError

    from .test_provider import _BufferedUpstreamResponse, _test_token

    provider = ChatGPTOAuthProvider(legacy_routing=mode)
    token = _test_token()
    token.user_id, token.login_generation = "user", "login"
    monkeypatch.setattr(provider, "_token_for_request", lambda: token)
    sent = []
    if failure == "reserve":
        def broken_reserve(*args):
            raise RuntimeError("synthetic routing failure")
        monkeypatch.setattr(provider._legacy_routing, "reserve", broken_reserve)
    original_http_request = cache_diagnostics.http_request
    if failure == "prepare":
        def broken_prepare(*args, **kwargs):
            raise RuntimeError("synthetic wire preparation failure")
        monkeypatch.setattr(cache_diagnostics, "http_request", broken_prepare)

    def open_request(request, **kwargs):
        assert request.data == original.encode("utf-8")
        sent.append({k.lower(): v for k, v in request.header_items()})
        if failure == "http" and len(sent) == 2:
            raise urllib.error.HTTPError(request.full_url, 400, "rejected", {}, io.BytesIO(b'{}'))
        if failure == "sse" and len(sent) == 2:
            response = _BufferedUpstreamResponse(b'data: {"type":"response.failed","response":{"error":{}}}\n\n')
            response.headers, response.status = {}, 200
            return response
        response = _BufferedUpstreamResponse(b'data: {"type":"response.completed","response":{}}\n\n')
        response.headers = {"x-codex-turn-state": "first-token" if len(sent) == 1 else "ignored-later"}
        response.status = 200
        return response

    monkeypatch.setattr(provider, "_open_request", open_request)
    body = payload()
    original = json.dumps(body)
    try:
        if failure == "reserve" and mode == "native":
            with pytest.raises(RuntimeError, match="routing failure"):
                list(provider._post_sse("/responses", body))
            assert not sent
            return
        if failure == "prepare":
            with pytest.raises(RuntimeError, match="preparation failure"):
                list(provider._post_sse("/responses", body))
            assert not any(entry.busy for entry in provider._legacy_routing._entries.values())
            monkeypatch.setattr(cache_diagnostics, "http_request", original_http_request)
        for _ in range(3):
            if failure == "http" and len(sent) == 1:
                with pytest.raises(ChatGPTOAuthUpstreamError):
                    list(provider._post_sse("/responses", body))
                assert len(sent) == 2  # No automatic retry of the failed generation.
            else:
                list(provider._post_sse("/responses", body))
            token.access_token = "ordinary-refreshed-access"
    finally:
        provider.close()
    assert json.dumps(body) == original
    assert "x-codex-turn-state" not in sent[0]
    if mode == "shadow" or failure == "reserve":
        assert all("x-codex-turn-state" not in r for r in sent)
    elif failure in {"http", "sse"}:
        assert sent[1]["x-codex-turn-state"] == "first-token"
        assert "x-codex-turn-state" not in sent[2]
    else:
        assert sent[1]["x-codex-turn-state"] == sent[2]["x-codex-turn-state"] == "first-token"
    if mode != "native":
        assert all("x-codex-routing-hint" not in r for r in sent)
        assert all(r["thread-id"] == r["session-id"] == r["x-client-request-id"] for r in sent)


def test_mixed_tool_result_and_text_is_a_conservative_boundary():
    pool = LegacyRouting()
    first = reserve(pool)
    complete(pool, first, CALL)
    next_step = reserve(pool, payload(USER, CALL, RESULT, {"role": "user", "content": "hook or steering"}))
    assert next_step.new_turn and next_step.mixed_user_tool and next_step.turn_state is None
    assert next_step.new_turn_v2 is False  # Shadow comparison only; conservative rule still controls echo.
    pool.release(next_step)


def test_real_legacy_adapter_replays_encrypted_reasoning_text_and_call(monkeypatch, model_catalog_snapshot):
    model = "gpt-5.6-sol"
    provider = ChatGPTOAuthProvider(legacy_routing="native")
    monkeypatch.setattr(provider, "get_model_catalog", lambda: model_catalog_snapshot)
    initial, *_ = anthropic_request_to_internal(model="claude-synthetic", system="rules",
                                               messages=[{"role": "user", "content": "task"}])
    kwargs = {"model": model, "prompt_cache_key": "s", "codex_metadata": False, "responses_lite": False}
    prepared = provider.preflight_chat(initial, **kwargs)
    reasoning = {"type": "reasoning", "id": "rs_fixture", "encrypted_content": "cipher-fixture",
                 "summary": [{"type": "summary_text", "text": "summary"}]}
    text = {"type": "message", "id": "msg_fixture", "role": "assistant",
            "content": [{"type": "output_text", "text": "text"}]}
    output = [reasoning, text, CALL]
    events = [{"type": "response.output_item.done", "item": item} for item in output]
    events.append({"type": "response.completed", "response": {"id": "resp_fixture", "output": [],
                   "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}})
    monkeypatch.setattr(provider, "_post_sse", lambda *a, **kw: iter(deepcopy(events)))
    response = provider.chat(initial, **kwargs)
    rendered = internal_response_to_anthropic(response, "claude-synthetic", "msg_rendered", backend_model=model)
    messages, *_ = anthropic_request_to_internal(model="claude-synthetic", system="rules", messages=[
        {"role": "user", "content": "task"}, {"role": "assistant", "content": rendered["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "result"}]},
    ], structured_tool_results=False)
    next_request = provider.preflight_chat(messages, **kwargs)
    pool = LegacyRouting()
    a = reserve(pool, prepared.payload)
    complete(pool, a, *output)
    b = reserve(pool, next_request.payload)
    assert b.decision == "reused_replay" and b.entry.thread_id == a.entry.thread_id
    assert b.headers()["x-codex-turn-state"] == "sticky-token"
    pool.release(b)
