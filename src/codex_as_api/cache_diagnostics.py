"""Content-free fingerprints and usage at the inference transport boundary."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Iterator, Mapping
from typing import Any, cast

from . import diagnostics


def _encoded(value: Any) -> bytes:
    # Preserve ordering, including JSON schema property order.
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()[:32]


def _prefixes(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value]
    chain = hashlib.sha256()
    result = []
    for item in items:
        encoded = _encoded(item)
        chain.update(len(encoded).to_bytes(8, "big"))
        chain.update(encoded)
        result.append(chain.copy().hexdigest()[:32])
    return result


def _cache_control_positions(payload: dict[str, Any]) -> list[list[int]]:
    # Scope codes: request=0, system=1, messages=2, tools=3. Only array indexes follow.
    positions = [[0]] if "cache_control" in payload else []

    def content(value: Any, path: list[int]) -> None:
        if isinstance(value, list):
            for index, block in enumerate(value):
                if isinstance(block, dict):
                    if "cache_control" in block:
                        positions.append([*path, index])
                    content(block.get("content"), [*path, index])

    system = payload.get("system")
    if isinstance(system, dict) and "cache_control" in system:
        positions.append([1])
    content(system, [1])
    messages = payload.get("messages")
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            if isinstance(message, dict):
                content(message.get("content"), [2, index])
    tools = payload.get("tools")
    if isinstance(tools, list):
        for index, tool in enumerate(tools):
            if isinstance(tool, dict) and "cache_control" in tool:
                positions.append([3, index])
    return positions


def client_request(
    payload: dict[str, Any], headers: Mapping[str, str] | None = None, *, normalized: bool = False,
) -> None:
    if diagnostics.enabled():
        headers = headers or {}
        diagnostics.emit(
            "client_cache_normalized" if normalized else "client_cache_request", fingerprint_version=1,
            client_cache_control_positions=[] if normalized else _cache_control_positions(payload),
            client_prefix_hashes=_prefixes(payload.get("messages", [])),
            client_instructions_hash=_hash(payload.get("system")), client_tools_hash=_hash(payload.get("tools")),
            client_model_hash=_hash(payload.get("model")),
            client_settings_hash=_hash({
                "body": {k: v for k, v in payload.items() if k not in {"messages", "system", "tools", "model"}},
                "headers": {k: headers.get(k) for k in (
                    "x-claude-code-session-id", "x-openai-subagent", "x-openai-memgen-request",
                )},
            }),
        )


def _response_cache_metadata(
    response: dict[str, Any], usage: dict[str, Any], details: dict[str, Any],
) -> dict[str, Any]:
    writes = details.get("cache_write_tokens_details")
    writes = writes if isinstance(writes, dict) else {}
    fields = {"cache_write_tokens": details.get("cache_write_tokens"),
              "cache_write_input_tokens": usage.get("cache_write_input_tokens"),
              "cache_write_ttl_30m_tokens": writes.get("ttl_30m_tokens"),
              "cache_write_ttl_12h_tokens": writes.get("ttl_12h_tokens")}
    fields = {k: v for k, v in fields.items() if isinstance(v, int) and not isinstance(v, bool) and v >= 0}
    options = response.get("prompt_cache_options")
    options = options if isinstance(options, dict) else {}
    diagnostic = response.get("prompt_cache_diagnostics")
    diagnostic = diagnostic if isinstance(diagnostic, dict) else {}
    return {**fields, "cache_write_usage_available": bool(fields),
            "cache_ttl": options.get("ttl"), "cache_mode": options.get("mode"),
            "cache_retention": response.get("prompt_cache_retention"),
            "cache_diagnostic_type": diagnostic.get("type"), "cache_diagnostic_reason": diagnostic.get("reason")}


def observe_events(
    events: Iterator[dict[str, Any]], payload: dict[str, Any], *, backend: str, transport: str,
) -> Iterator[dict[str, Any]]:
    if not diagnostics.enabled():
        yield from events
        return
    started = time.monotonic()
    attempt_id = uuid.uuid4().hex
    payload_hash = _hash(payload)
    prefixes = _prefixes(payload.get("input", []))
    diagnostics.emit(
        "cache_request", attempt_id=attempt_id, backend=backend, transport=transport, fingerprint_version=1,
        model=diagnostics.safe_model(payload.get("model")), payload_hash=payload_hash,
        instructions_hash=_hash(payload.get("instructions")), tools_hash=_hash(payload.get("tools")),
        cache_key_hash=_hash(payload.get("prompt_cache_key")), input_prefix_hashes=prefixes,
        content_items=len(prefixes), settings_hash=_hash({
            k: v for k, v in payload.items() if k not in {"input", "instructions", "tools", "prompt_cache_key"}
        }),
    )
    terminal = False
    reason = "upstream_incomplete"
    response_id = None
    event_count = 0
    first_event_ms = first_delta_ms = last_event_ms = None
    last_event_type = "other"
    try:
        for event in events:
            kind = event.get("type")
            event_count += 1
            last_event_ms = int(1000 * (time.monotonic() - started))
            if first_event_ms is None:
                first_event_ms = last_event_ms
            if first_delta_ms is None and isinstance(kind, str) and kind.endswith(".delta"):
                first_delta_ms = last_event_ms
            last_event_type = kind if kind in diagnostics._ENUMS["last_event_type"] else "other"
            response = event.get("response")
            response = response if isinstance(response, dict) else {}
            if response_id is None and isinstance(response.get("id"), str):
                response_id = response["id"]
                diagnostics.emit("upstream_created", attempt_id=attempt_id, upstream_response_id=response_id,
                                 elapsed_ms=last_event_ms)
            if kind in {"response.completed", "response.failed", "response.incomplete", "error"}:
                usage = response.get("usage")
                usage = usage if isinstance(usage, dict) else {}
                details = usage.get("input_tokens_details")
                details = details if isinstance(details, dict) else {}
                counts = {"input_tokens": usage.get("input_tokens"),
                          "cached_tokens": details.get("cached_tokens"),
                          "output_tokens": usage.get("output_tokens")}
                available = all(isinstance(v, int) and not isinstance(v, bool) and v >= 0
                                for v in counts.values())
                if available:
                    available = cast(int, counts["cached_tokens"]) <= cast(int, counts["input_tokens"])
                diagnostics.emit(
                    "cache_response", attempt_id=attempt_id, payload_hash=payload_hash,
                    backend=backend, transport=transport,
                    reason="completed" if kind == "response.completed" else "upstream_failed",
                    elapsed_ms=int(1000 * (time.monotonic() - started)), usage_available=available,
                    upstream_response_id=response.get("id") or response_id,
                    event_count=event_count, first_event_ms=first_event_ms, first_delta_ms=first_delta_ms,
                    last_event_ms=last_event_ms, last_event_type=last_event_type,
                    actual_model_hash=_hash(response.get("model")),
                    service_tier_hash=_hash(response.get("service_tier")), **counts,
                    **_response_cache_metadata(response, usage, details),
                )
                terminal = True
            yield event
    except GeneratorExit:
        reason = "interrupted"
        raise
    except BaseException:
        reason = "upstream_error"
        raise
    finally:
        try:
            close = getattr(events, "close", None)
            if close is not None:
                close()
        finally:
            if not terminal:
                diagnostics.emit("cache_response", attempt_id=attempt_id, payload_hash=payload_hash,
                                 backend=backend, transport=transport, reason=reason, usage_available=False,
                                 upstream_response_id=response_id, event_count=event_count,
                                 first_event_ms=first_event_ms, first_delta_ms=first_delta_ms,
                                 last_event_ms=last_event_ms, last_event_type=last_event_type,
                                 elapsed_ms=int(1000 * (time.monotonic() - started)))


def http_request(
    payload: dict[str, Any], headers: Mapping[str, str], body: bytes, attempt: int,
    *, lease_decision: str | None = None, new_turn: bool | None = None,
    routing_mode: str | None = None, would_echo: bool | None = None,
    history_id: str | None = None, active_histories: int | None = None,
    mixed_user_tool: bool | None = None,
    new_turn_v2: bool | None = None, rule_disagreement: bool | None = None,
) -> str | None:
    if not diagnostics.enabled():
        return None
    payload_hash = _hash(payload)
    normalized = {k.lower(): v for k, v in headers.items()}
    routing = {k: normalized[k] for k in sorted(normalized) if k in {
        "session-id", "thread-id", "x-client-request-id", "chatgpt-account-id",
        "originator", "version", "user-agent", "openai-beta",
    } or k.startswith("x-codex-")}
    diagnostics.emit(
        "cache_http_request", payload_hash=payload_hash, routing_hash=_hash(routing), network_attempt=attempt,
        routing_config_hash=_hash({k: v for k, v in routing.items() if k != "x-codex-turn-state"}),
        lease_decision=lease_decision, new_turn=new_turn,
        routing_mode=routing_mode, would_echo=would_echo, history_hash=_hash(history_id),
        active_histories=active_histories,
        mixed_user_tool=mixed_user_tool,
        new_turn_v2=new_turn_v2, rule_disagreement=rule_disagreement,
        wire_hash=hashlib.sha256(body).hexdigest()[:32],
        session_header_hash=_hash(normalized.get("session-id")),
        thread_header_hash=_hash(normalized.get("thread-id")),
        client_request_header_hash=_hash(normalized.get("x-client-request-id")),
        account_hash=_hash(normalized.get("chatgpt-account-id")),
        turn_state_present=bool(normalized.get("x-codex-turn-state")),
        turn_state_hash=_hash(normalized.get("x-codex-turn-state")),
        routing_hint_present=bool(normalized.get("x-codex-routing-hint")),
        routing_hint_hash=_hash(normalized.get("x-codex-routing-hint")),
    )
    return payload_hash


def http_response(payload: dict[str, Any], status: Any, headers: Any, attempt: int) -> None:
    if diagnostics.enabled():
        normalized = {k.lower(): v for k, v in headers.items()} if headers else {}
        diagnostics.emit("cache_http_response", payload_hash=_hash(payload), status=status,
                         network_attempt=attempt,
                         upstream_request_id=next((normalized[k] for k in
                             ("x-request-id", "x-oai-request-id", "x-openai-request-id") if normalized.get(k)), None),
                         cf_ray=normalized.get("cf-ray"),
                         turn_state_present=bool(normalized.get("x-codex-turn-state")),
                         turn_state_hash=_hash(normalized.get("x-codex-turn-state")),
                         header_model_hash=_hash(normalized.get("openai-model")))


def websocket_request(
    payload: dict[str, Any], body: str, *, account_id: str, attempt_id: str, channel_id: str,
) -> None:
    if diagnostics.enabled():
        diagnostics.emit("cache_websocket_request", payload_hash=_hash(payload),
                         wire_hash=hashlib.sha256(body.encode("utf-8")).hexdigest()[:32],
                         account_hash=_hash(account_id), attempt_id=attempt_id, channel_id=channel_id)
