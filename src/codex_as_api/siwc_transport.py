"""Сохраняемые SIWC WebSocket; полный input каждого хода, лимит только при явном выборе."""
from __future__ import annotations

import hashlib
import http.client
import io
import json
import logging
import os
import threading
import time
import urllib.error
import uuid
from collections import OrderedDict, deque
from collections.abc import Iterator
from contextlib import ExitStack, nullcontext, suppress
from dataclasses import dataclass, field
from typing import Any

from websockets.exceptions import ConnectionClosed, InvalidStatus, PayloadTooBig, WebSocketException
from websockets.protocol import State
from websockets.sync.client import connect

from . import cache_diagnostics
from .auth import ChatGPTOAuthProtocolError, ChatGPTOAuthUpstreamError
from .diagnostics import current as diagnostic_context
from .diagnostics import emit as diagnostic_emit
from .diagnostics import enabled as diagnostics_enabled
from .diagnostics import safe_model
from .strict_json import strict_json_loads

# Native Codex 0.166.0 tungstenite default. Queue/ping policy stays unchanged.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


def _close_details(error: BaseException, socket: Any) -> tuple[int | None, str, str | None]:
    code = None
    source = None
    if isinstance(error, ConnectionClosed):
        for name in ("rcvd", "sent"):
            close = getattr(error, name, None)
            if close is not None:
                code, source = close.code, "received" if name == "rcvd" else "sent"
                break
    if code is None and socket is not None:
        code = getattr(socket, "close_code", None)
        source = "socket" if code is not None else None
    category = ("message_too_large" if code == 1009 or isinstance(error.__cause__, PayloadTooBig)
                else "timeout" if isinstance(error, TimeoutError)
                else "connection_closed" if isinstance(error, ConnectionClosed)
                else "protocol_error" if isinstance(error, WebSocketException) else "transport_error")
    return code, category, source

SIWC_WEBSOCKET_URL = "wss://api.openai.com/v1/responses"
_TERMINAL_EVENTS = frozenset({"response.completed", "response.failed", "response.incomplete", "error"})
_LOG = logging.getLogger(__name__)


def _branch_key(payload: dict[str, Any]) -> str:
    # Claude Code gives siblings the parent's session ID. The initial user turn
    # and instructions distinguish their histories without retaining prompt text.
    initial_input = []
    for item in payload.get("input", []):
        initial_input.append(item)
        if isinstance(item, dict) and item.get("role") == "user":
            break
    identity = (payload.get("instructions"), initial_input)
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _fingerprint(value: Any) -> bytes:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).digest()


def _is_output_item(item: dict[str, Any]) -> bool:
    return item.get("role") == "assistant" or item.get("type") in {
        "reasoning", "function_call", "web_search_call",
    }


def _output_components(items: list[dict[str, Any]]) -> tuple[Any, ...]:
    # Только сравнение replay: wire-ответы и история не меняются. Фасад объединяет
    # текст и выдаёт tools в конце; служебные id/status сообщения в replay исчезают.
    text: list[str] = []
    reasoning: list[Any] = []
    calls: list[Any] = []
    other: list[Any] = []
    for item in items:
        typ = item.get("type")
        if item.get("role") == "assistant":
            content = item.get("content", [])
            if isinstance(content, str):
                text.append(content)
            else:
                text.extend(part["text"] for part in content
                            if part.get("type") in {"text", "input_text", "output_text"})
        elif typ == "reasoning":
            reasoning.append((item.get("id"), item.get("encrypted_content"),
                              [p.get("text") for p in item.get("summary", [])],
                              [p.get("text") for p in item.get("content", [])]))
        elif typ == "function_call":
            arguments = item.get("arguments")
            if isinstance(arguments, str):
                # Валидация остаётся в provider, здесь только отпечаток.
                with suppress(ValueError):
                    arguments = strict_json_loads(arguments)
            calls.append((item.get("call_id"), str(item.get("name", "")).rsplit(".", 1)[-1], arguments))
        else:
            other.append(item)
    return ("".join(text), reasoning, calls, other)


def _output_key(items: list[dict[str, Any]]) -> bytes:
    return _fingerprint(_output_components(items))


@dataclass(frozen=True)
class _Anchor:
    history: tuple[bytes, ...]
    output_key: bytes | None
    input_tokens: int = 0


@dataclass
class _Connection:
    lock: Any = field(default_factory=threading.Lock)
    anchors_lock: Any = field(default_factory=threading.Lock)
    users: int = 0
    last_used: float = 0.0
    opened_at: float = 0.0
    token_hash: str = ""
    stack: ExitStack = field(default_factory=ExitStack)
    socket: Any = None
    history: tuple[bytes, ...] | None = None
    output_key: bytes | None = None
    anchors: deque[_Anchor] = field(default_factory=lambda: deque(maxlen=16))
    channel_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def close(self) -> None:
        # Socket I/O stays outside the shared pool lock, including retirement.
        had_socket = self.socket is not None
        try:
            self.stack.close()
        finally:
            with self.anchors_lock:
                self.stack = ExitStack()
                self.socket = None
                self.token_hash = ""
                self.history = None
                self.output_key = None
                self.anchors.clear()
                if had_socket:
                    self.channel_id = uuid.uuid4().hex


class SIWCWebSocketPool:
    """Каналы по аккаунту/сессии/модели/ветке, без локального потолка по умолчанию."""

    def __init__(self, *, limit: int | None = None, idle_timeout: float = 3300.0,
                 warm_wait_seconds: float = 20.0, max_waiters: int = 8) -> None:
        if not 0 <= warm_wait_seconds <= 30 or not 0 <= max_waiters <= 64:
            raise ValueError("SIWC warm wait must be 0..30 seconds with 0..64 waiters")
        self._limit = limit
        self._idle_timeout = idle_timeout
        self._warm_wait_seconds = warm_wait_seconds
        self._max_waiters = max_waiters
        self._waiters = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._connections: OrderedDict[tuple[str, str, str, str, int], _Connection] = OrderedDict()
        self._retired: deque[_Connection] = deque()
        self._closers: set[threading.Thread] = set()
        self._closed = False

    @property
    def max_connections(self) -> int | None:
        return self._limit

    @property
    def active_requests(self) -> int:
        with self._lock:
            return self._waiters + sum(entry.users for entry in self._connections.values())

    def _wake_waiters(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _reserve(self, key: tuple[str, str, str, str], inputs: list[dict[str, Any]],
                 request_id: str, *, token_hash: str, timeout: float, cancel: Any = None) -> _Connection:
        expired = []
        history = tuple(_fingerprint(item) for item in inputs)
        replay_keys: dict[int, bytes] = {}

        mismatch_reasons: set[str] = set()

        def affinity(candidate: _Connection, *, observe_mismatch: bool) -> tuple[int, int]:
            if not candidate.anchors:
                return 0, 0
            scores = []
            for anchor in candidate.anchors:
                previous, output_key = anchor.history, anchor.output_key
                size = len(previous)
                if history == previous:
                    scores.append((size + 1, anchor.input_tokens))
                elif history[:size] != previous:
                    if observe_mismatch:
                        mismatch_reasons.add("input_mismatch")
                elif output_key is None:
                    if observe_mismatch:
                        mismatch_reasons.add("missing_key")
                else:
                    if size not in replay_keys:
                        replay = []
                        for item in inputs[size:]:
                            if not _is_output_item(item):
                                break
                            replay.append(item)
                        replay_keys[size] = _output_key(replay)
                    if replay_keys[size] == output_key:
                        scores.append((size + 1, anchor.input_tokens))
                    else:
                        if observe_mismatch:
                            mismatch_reasons.add("replay_mismatch")
            return max(scores, default=(-1, 0))

        started = time.monotonic()
        deadline = started + min(self._warm_wait_seconds, timeout)
        waiting = False
        outcome = "not_needed"
        try:
            with cancel.bind(self._wake_waiters) if cancel is not None else nullcontext(), self._condition:
                while True:
                    if cancel is not None:
                        cancel.check()
                    if self._closed:
                        raise ChatGPTOAuthUpstreamError(503, "SIWC transport is closed")
                    now = time.monotonic()
                    for old_key, candidate in list(self._connections.items()):
                        if candidate.users == 0 and now - candidate.last_used >= self._idle_timeout:
                            expired.append(self._connections.pop(old_key))
                    self._retire_locked(expired)
                    expired.clear()
                    mismatch_reasons.clear()
                    group = [(k, v) for k, v in reversed(self._connections.items()) if k[:4] == key]
                    free_candidates = sum(v.users == 0 for _, v in group)
                    busy_candidates = len(group) - free_candidates
                    candidates = [(affinity(v, observe_mismatch=v.users == 0), k, v) for k, v in group]
                    best = max(((score[0], k) for score, k, v in candidates if v.users == 0),
                               key=lambda candidate: candidate[0], default=(-1, None))
                    busy = max((score for score, _, v in candidates if v.users
                                and v.socket is not None and v.token_hash == token_hash
                                and getattr(v.socket, "state", State.OPEN) == State.OPEN
                                and now - v.opened_at < 3300.0), default=(-1, 0))
                    should_wait = busy[0] > max(0, best[0]) and busy[1] >= 32000
                    if not should_wait:
                        if waiting:
                            outcome = "acquired" if best[0] > 0 else "affinity_lost"
                        break
                    if deadline <= now:
                        outcome = "timeout"
                        break
                    if not waiting:
                        if self._waiters >= self._max_waiters:
                            outcome = "capacity"
                            break
                        self._waiters += 1
                        waiting = True
                        diagnostic_emit("channel_wait_started", attempt_id=request_id,
                                        active_waiters=self._waiters)
                    # Один дедлайн на весь подбор; wait отпускает lock. FIFO не обещается.
                    self._condition.wait(deadline - now)
                available = best[1] if best[0] >= 0 else None
                return self._lease_locked(key, history, request_id, available, free_candidates,
                                          busy_candidates, mismatch_reasons, outcome, started, waiting)
        except Exception:
            outcome = "cancelled" if cancel is not None and cancel.cancelled else "error"
            raise
        finally:
            if waiting:
                with self._condition:
                    self._waiters -= 1
                diagnostic_emit("channel_wait_finished", attempt_id=request_id,
                                channel_wait_ms=round((time.monotonic() - started) * 1000), wait_outcome=outcome)

    def _lease_locked(self, key: tuple[str, str, str, str], history: tuple[bytes, ...], request_id: str,
                      available: tuple[str, str, str, str, int] | None, free_candidates: int,
                      busy_candidates: int, mismatch_reasons: set[str], outcome: str,
                      started: float, waiting: bool) -> _Connection:
        expired = []
        selection_reason = "existing" if available is not None else next(
            (reason for reason in ("replay_mismatch", "missing_key", "input_mismatch")
             if reason in mismatch_reasons), "new")
        if available is None:
            if self._limit is not None and len(self._connections) >= self._limit:
                victim = next((k for k, v in self._connections.items() if v.users == 0), None)
                if victim is None:
                    raise ChatGPTOAuthUpstreamError(429, "All SIWC session channels are busy")
                expired.append(self._connections.pop(victim))
            entry = _Connection(anchors_lock=self._lock)
            lane = 0
            while (*key, lane) in self._connections:
                lane += 1
            available = (*key, lane)
            self._connections[available] = entry
        else:
            entry = self._connections[available]
        # An active channel is never leased twice, even for identical prompts.
        entry.users = 1
        entry.history = history
        entry.output_key = None
        self._connections.move_to_end(available)
        self._retire_locked(expired)
        diagnostic_emit("channel_selected", attempt_id=request_id, channel_id=entry.channel_id,
                        selection_reason=selection_reason,
                        free_candidates=free_candidates, busy_candidates=busy_candidates,
                        channel_wait_ms=round((time.monotonic() - started) * 1000) if waiting else 0,
                        wait_outcome=outcome)
        return entry

    def events(
        self,
        payload: dict[str, Any],
        *,
        account_id: str,
        access_token: str,
        timeout: float,
        cancel: Any = None,
    ) -> Iterator[dict[str, Any]]:
        key = (account_id, payload.get("prompt_cache_key") or "anonymous",
               payload.get("model", ""), _branch_key(payload))
        request_id = uuid.uuid4().hex[:12]
        token_hash = hashlib.sha256(access_token.encode()).hexdigest()
        entry = self._reserve(key, payload.get("input", []), request_id,
                              token_hash=token_hash, timeout=timeout, cancel=cancel)
        history = entry.history
        channel_id = entry.channel_id
        output: list[dict[str, Any]] = []
        session_hash = hashlib.sha256(key[1].encode()).hexdigest()[:12]
        started = time.monotonic()
        first_event_ms = first_delta_ms = None
        completed = False
        reused = False
        terminal_reason = "interrupted"
        socket = None
        close_code = close_category = close_source = None
        max_message_bytes = 0
        diagnostic_emit("upstream_start", attempt_id=request_id, session_hash=session_hash, branch_hash=key[3][:12],
                        model=payload.get("model"))
        if diagnostics_enabled():
            _LOG.info("SIWC WS start request=%s", request_id)
        else:
            _LOG.info("SIWC WS start request=%s model=%s session=%s branch=%s active=%s",
                      request_id, safe_model(key[2]), session_hash, key[3][:12], self.active_requests)
        try:
            with entry.lock:
                try:
                    if cancel is not None:
                        cancel.check()
                    reused = (entry.socket is not None and getattr(entry.socket, "state", State.OPEN) == State.OPEN
                              and entry.token_hash == token_hash
                              and time.monotonic() - entry.opened_at < 3300.0)
                    if reused:
                        connection_reason = "reused"
                    elif entry.socket is None:
                        connection_reason = "no_socket"
                    elif getattr(entry.socket, "state", State.OPEN) != State.OPEN:
                        connection_reason = "socket_closed"
                    elif entry.token_hash != token_hash:
                        connection_reason = "credentials_changed"
                    else:
                        connection_reason = "connection_expired"
                    if not reused:
                        entry.close()
                        channel_id = entry.channel_id
                        entry.socket = entry.stack.enter_context(connect(
                            SIWC_WEBSOCKET_URL,
                            additional_headers={"Authorization": f"Bearer {access_token}"},
                            proxy=os.getenv("https_proxy") or os.getenv("HTTPS_PROXY"),
                            user_agent_header=None,
                            open_timeout=min(timeout, 15.0),
                            close_timeout=3.0,
                            max_size=MAX_MESSAGE_BYTES,
                        ))
                        entry.token_hash = token_hash
                        entry.opened_at = time.monotonic()
                        handshake = getattr(entry.socket, "response", None)
                        handshake_headers = getattr(handshake, "headers", None)
                        if handshake_headers is not None:
                            diagnostic_emit("websocket_handshake", channel_id=entry.channel_id,
                                            turn_state_present="x-codex-turn-state" in handshake_headers)
                    diagnostic_emit("upstream_start", attempt_id=request_id, channel_id=entry.channel_id,
                                    reused=reused, reason=connection_reason)
                    socket = entry.socket

                    def stop() -> None:
                        # Invalidate before close so a following request cannot reuse a closing channel.
                        if not completed:
                            entry.token_hash = ""
                            socket.close()

                    with cancel.bind(stop) if cancel is not None else nullcontext():
                        wire = {k: v for k, v in payload.items() if k != "stream"}
                        wire["type"] = "response.create"
                        body = json.dumps(wire)
                        cache_diagnostics.websocket_request(payload, body, account_id=account_id,
                                                            attempt_id=request_id, channel_id=entry.channel_id)
                        socket.send(body)
                        completed = False
                        try:
                            while True:
                                raw = socket.recv(timeout=timeout)
                                if diagnostics_enabled():
                                    message_bytes = len(raw) if isinstance(raw, bytes) else len(raw.encode("utf-8"))
                                    max_message_bytes = max(max_message_bytes, message_bytes)
                                try:
                                    event = strict_json_loads(raw)
                                except (UnicodeDecodeError, ValueError) as exc:
                                    raise ChatGPTOAuthProtocolError("SIWC WebSocket returned invalid JSON") from exc
                                if not isinstance(event, dict) or not isinstance(event.get("type"), str):
                                    raise ChatGPTOAuthProtocolError("SIWC WebSocket event must have a type")
                                elapsed_ms = round((time.monotonic() - started) * 1000)
                                if first_event_ms is None:
                                    first_event_ms = elapsed_ms
                                if first_delta_ms is None and event["type"].endswith(".delta"):
                                    first_delta_ms = elapsed_ms
                                if event["type"] in {"codex.response.metadata", "responsesapi.websocket_timing"}:
                                    continue
                                if event["type"] == "response.output_item.done" and isinstance(event.get("item"), dict):
                                    output.append(event["item"])
                                if event["type"] in _TERMINAL_EVENTS:
                                    completed = event["type"] == "response.completed"
                                    terminal_reason = {"response.failed": "upstream_failed",
                                                       "response.incomplete": "upstream_incomplete",
                                                       "error": "upstream_error"}.get(event["type"], "interrupted")
                                    if completed:
                                        response = event.get("response")
                                        final_output = response.get("output") if isinstance(response, dict) else None
                                        key_error = "none"
                                        try:
                                            output_key = _output_key(final_output or output)
                                        except (TypeError, ValueError, KeyError, AttributeError):
                                            # Невалидный ответ проверит provider, не подбор канала.
                                            output_key = None
                                            key_error = "invalid_output"
                                        diagnostic_emit("output_key", attempt_id=request_id,
                                                        channel_id=entry.channel_id,
                                                        output_key_source="completed" if final_output else "done",
                                                        output_key_error=key_error, done_count=len(output),
                                                        completed_count=(len(final_output)
                                                                         if isinstance(final_output, list) else 0))
                                        usage = response.get("usage") if isinstance(response, dict) else None
                                        if isinstance(usage, dict):
                                            details = usage.get("input_tokens_details")
                                            cached = details.get("cached_tokens") if isinstance(details, dict) else None
                                            counters = (usage.get("input_tokens"), cached, usage.get("output_tokens"))
                                            if not diagnostics_enabled() and all(
                                                v is None or (isinstance(v, int) and not isinstance(v, bool))
                                                for v in counters
                                            ):
                                                _LOG.info("SIWC WS model=%s input=%s cached=%s output=%s session=%s "
                                                          "branch=%s request=%s elapsed_ms=%s first_event_ms=%s "
                                                          "first_delta_ms=%s reused=%s",
                                                          safe_model(key[2]), *counters,
                                                          session_hash, key[3][:12], request_id, elapsed_ms,
                                                          first_event_ms, first_delta_ms, reused)
                                        else:
                                            counters = (None, None, None)
                                        tokens = counters[0]
                                        anchor_tokens = (tokens if isinstance(tokens, int)
                                                         and not isinstance(tokens, bool) and tokens >= 0 else 0)
                                        with self._lock:
                                            entry.history = history
                                            entry.output_key = output_key
                                            if history is not None:
                                                anchor = _Anchor(history, output_key, anchor_tokens)
                                                with suppress(ValueError):
                                                    entry.anchors.remove(anchor)
                                                entry.anchors.append(anchor)
                                        if diagnostics_enabled():
                                            _LOG.info("SIWC WS completed request=%s", request_id)
                                        diagnostic_emit("upstream_completed", attempt_id=request_id,
                                                        max_message_bytes=max_message_bytes,
                                                        input_tokens=counters[0], cached_tokens=counters[1],
                                                        output_tokens=counters[2],
                                                        total_tokens=(usage.get("total_tokens")
                                                                      if isinstance(usage, dict) else None),
                                                        elapsed_ms=elapsed_ms, first_event_ms=first_event_ms,
                                                        first_delta_ms=first_delta_ms, reused=reused)
                                    yield event
                                    return
                                yield event
                        finally:
                            if not completed:
                                entry.close()
                except InvalidStatus as exc:
                    close_category = "handshake_error"
                    entry.close()
                    headers = http.client.HTTPMessage()
                    for name, value in exc.response.headers.raw_items():
                        headers[name] = value
                    raise urllib.error.HTTPError(
                        SIWC_WEBSOCKET_URL, exc.response.status_code, "SIWC WebSocket handshake failed",
                        headers, io.BytesIO(exc.response.body),
                    ) from exc
                except (WebSocketException, TimeoutError, OSError) as exc:
                    close_code, close_category, close_source = _close_details(exc, socket)
                    entry.close()
                    raise OSError(str(exc)) from exc
        finally:
            if not completed:
                if (context := diagnostic_context()) is not None:
                    context["failed"] = True
                diagnostic_emit("upstream_error", attempt_id=request_id, reason=terminal_reason,
                                close_code=close_code, close_category=close_category, close_source=close_source,
                                max_message_bytes=max_message_bytes,
                                elapsed_ms=round((time.monotonic() - started) * 1000),
                                first_event_ms=first_event_ms, first_delta_ms=first_delta_ms, reused=reused)
                if diagnostics_enabled():
                    _LOG.info("SIWC WS interrupted request=%s", request_id)
                else:
                    _LOG.info("SIWC WS interrupted request=%s model=%s session=%s branch=%s elapsed_ms=%s "
                              "first_event_ms=%s first_delta_ms=%s",
                              request_id, safe_model(key[2]), session_hash, key[3][:12],
                              round((time.monotonic() - started) * 1000), first_event_ms, first_delta_ms)
            with self._lock:
                entry.users -= 1
                entry.last_used = time.monotonic()
                self._condition.notify_all()
                diagnostic_emit("channel_released", attempt_id=request_id, channel_id=channel_id,
                                close_code=close_code, close_category=close_category, close_source=close_source,
                                max_message_bytes=max_message_bytes,
                                reason="completed" if completed else terminal_reason)

    def _retire_locked(self, entries: list[_Connection]) -> None:
        self._retired.extend(entries)
        for _ in range(min(8 - len(self._closers), len(self._retired))):
            worker = threading.Thread(target=self._drain_retired, name="siwc-close", daemon=False)
            self._closers.add(worker)
            worker.start()

    def _drain_retired(self) -> None:
        while True:
            with self._lock:
                if not self._retired:
                    self._closers.remove(threading.current_thread())
                    return
                entry = self._retired.popleft()
            try:
                entry.close()
            except Exception:
                _LOG.warning("SIWC WebSocket cleanup failed category=cleanup_error")

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._condition.notify_all()
            self._retire_locked(list(self._connections.values()))
            self._connections.clear()
            workers = list(self._closers)
        # Запрос не ждёт idle-close; явная остановка дожидается всей очереди.
        for worker in workers:
            worker.join()
