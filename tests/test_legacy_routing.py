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


def reserve(pool, body=None, *, account="account", token="token"):
    lease = pool.reserve(body or payload(), account, token)
    assert lease is not None
    return lease


def complete(pool, lease, *output):
    lease.response_headers({"x-codex-turn-state": "sticky-token"})
    for item in output:
        lease.observe({"type": "response.output_item.done", "item": item})
    lease.observe({"type": "response.completed", "response": {"output": []}})
    pool.release(lease)


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


@pytest.mark.parametrize("changed", ["model", "instructions", "tools", "reasoning", "service_tier"])
def test_effective_config_change_does_not_echo_old_turn_state(changed):
    pool = LegacyRouting()
    a = reserve(pool)
    complete(pool, a, CALL)
    body = payload(USER, CALL, RESULT)
    body[changed] = {"tools": [], "reasoning": {"effort": "high"}}.get(changed, "changed")
    b = reserve(pool, body)
    assert "x-codex-turn-state" not in b.headers()
    pool.release(b)


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


def test_token_refresh_preserves_thread_identity_and_only_resets_turn_state():
    pool = LegacyRouting()
    a = reserve(pool)
    complete(pool, a, CALL)
    b = reserve(pool, payload(USER, CALL, RESULT), token="refreshed")
    assert b.entry.thread_id == a.entry.thread_id and b.decision == "reused_replay"
    assert "x-codex-turn-state" not in b.headers()
    pool.release(b)


def test_normal_thread_identity_survives_registry_restart_and_anchors_are_compact():
    pool = LegacyRouting()
    a = reserve(pool)
    complete(pool, a, CALL)
    b = reserve(LegacyRouting())
    assert b.entry.thread_id == a.entry.thread_id
    assert len(a.entry.anchors[0]) == 3 and isinstance(a.entry.anchors[0][0], int)


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
