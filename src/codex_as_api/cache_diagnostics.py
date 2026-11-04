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


def client_request(payload: dict[str, Any], headers: Mapping[str, str] | None = None) -> None:
    if diagnostics.enabled():
        headers = headers or {}
        diagnostics.emit(
            "client_cache_request", client_prefix_hashes=_prefixes(payload.get("messages", [])),
            client_instructions_hash=_hash(payload.get("system")), client_tools_hash=_hash(payload.get("tools")),
            client_model_hash=_hash(payload.get("model")),
            client_settings_hash=_hash({
                "body": {k: v for k, v in payload.items() if k not in {"messages", "system", "tools", "model"}},
                "headers": {k: headers.get(k) for k in (
                    "x-claude-code-session-id", "x-openai-subagent", "x-openai-memgen-request",
                )},
            }),
        )


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
        "cache_request", attempt_id=attempt_id, backend=backend, transport=transport,
        model=diagnostics.safe_model(payload.get("model")), payload_hash=payload_hash,
        instructions_hash=_hash(payload.get("instructions")), tools_hash=_hash(payload.get("tools")),
        cache_key_hash=_hash(payload.get("prompt_cache_key")), input_prefix_hashes=prefixes,
        content_items=len(prefixes), settings_hash=_hash({
            k: v for k, v in payload.items() if k not in {"input", "instructions", "tools", "prompt_cache_key"}
        }),
    )
    terminal = False
    reason = "upstream_incomplete"
    try:
        for event in events:
            kind = event.get("type")
            if kind in {"response.completed", "response.failed", "response.incomplete", "error"}:
                response = event.get("response")
                response = response if isinstance(response, dict) else {}
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
                    upstream_response_id=response.get("id"),
                    actual_model_hash=_hash(response.get("model")),
                    service_tier_hash=_hash(response.get("service_tier")), **counts,
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
                                 elapsed_ms=int(1000 * (time.monotonic() - started)))


def http_request(payload: dict[str, Any], headers: Mapping[str, str], body: bytes, attempt: int) -> None:
    if not diagnostics.enabled():
        return
    normalized = {k.lower(): v for k, v in headers.items()}
    routing = {k: normalized[k] for k in sorted(normalized) if k in {
        "session-id", "thread-id", "x-client-request-id", "chatgpt-account-id",
        "originator", "version", "user-agent", "openai-beta",
    } or k.startswith("x-codex-")}
    diagnostics.emit(
        "cache_http_request", payload_hash=_hash(payload), routing_hash=_hash(routing), network_attempt=attempt,
        wire_hash=hashlib.sha256(body).hexdigest()[:32],
        session_header_hash=_hash(normalized.get("session-id")),
        thread_header_hash=_hash(normalized.get("thread-id")),
        client_request_header_hash=_hash(normalized.get("x-client-request-id")),
        account_hash=_hash(normalized.get("chatgpt-account-id")),
    )


def http_response(payload: dict[str, Any], status: Any, headers: Any, attempt: int) -> None:
    if diagnostics.enabled():
        diagnostics.emit("cache_http_response", payload_hash=_hash(payload), status=status,
                         network_attempt=attempt, upstream_request_id=headers.get("x-request-id") if headers else None)


def websocket_request(
    payload: dict[str, Any], body: str, *, account_id: str, attempt_id: str, channel_id: str,
) -> None:
    if diagnostics.enabled():
        diagnostics.emit("cache_websocket_request", payload_hash=_hash(payload),
                         wire_hash=hashlib.sha256(body.encode("utf-8")).hexdigest()[:32],
                         account_hash=_hash(account_id), attempt_id=attempt_id, channel_id=channel_id)
