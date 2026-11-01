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

from websockets.exceptions import InvalidStatus, WebSocketException
from websockets.protocol import State
from websockets.sync.client import connect

from .auth import ChatGPTOAuthProtocolError, ChatGPTOAuthUpstreamError
from .strict_json import strict_json_loads

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


def _output_key(items: list[dict[str, Any]]) -> bytes:
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
    return _fingerprint(("".join(text), reasoning, calls, other))


@dataclass
class _Connection:
    lock: Any = field(default_factory=threading.Lock)
    users: int = 0
    last_used: float = 0.0
    opened_at: float = 0.0
    token_hash: str = ""
    stack: ExitStack = field(default_factory=ExitStack)
    socket: Any = None
    history: tuple[bytes, ...] | None = None
    output_key: bytes | None = None

    def close(self) -> None:
        try:
            self.stack.close()
        finally:
            self.stack = ExitStack()
            self.socket = None
            self.token_hash = ""
            self.history = None
            self.output_key = None


class SIWCWebSocketPool:
    """Каналы по аккаунту/сессии/модели/ветке, без локального потолка по умолчанию."""

    def __init__(self, *, limit: int | None = None, idle_timeout: float = 1800.0) -> None:
        self._limit = limit
        self._idle_timeout = idle_timeout
        self._lock = threading.Lock()
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
            return sum(entry.users for entry in self._connections.values())

    def _reserve(self, key: tuple[str, str, str, str], inputs: list[dict[str, Any]]) -> _Connection:
        expired = []
        history = tuple(_fingerprint(item) for item in inputs)
        replay_keys: dict[int, bytes] = {}

        def affinity(candidate: _Connection) -> int:
            previous = candidate.history
            if previous is None:
                return 0
            if history == previous:
                return len(previous) + 1  # Точный повтор тоже сохраняет канал.
            size = len(previous)
            if history[:size] != previous or candidate.output_key is None:
                return -1
            if size not in replay_keys:
                replay = []
                for item in inputs[size:]:
                    if not _is_output_item(item):
                        break
                    replay.append(item)
                replay_keys[size] = _output_key(replay)
            return size + 1 if replay_keys[size] == candidate.output_key else -1

        now = time.monotonic()
        with self._lock:
            if self._closed:
                raise ChatGPTOAuthUpstreamError(503, "SIWC transport is closed")
            for old_key, candidate in list(self._connections.items()):
                if candidate.users == 0 and now - candidate.last_used >= self._idle_timeout:
                    expired.append(self._connections.pop(old_key))
            candidates = ((affinity(v), k) for k, v in reversed(self._connections.items())
                          if k[:4] == key and v.users == 0)
            best = max(candidates, key=lambda candidate: candidate[0], default=(-1, None))
            available = best[1] if best[0] >= 0 else None
            if available is None:
                if self._limit is not None and len(self._connections) >= self._limit:
                    victim = next((k for k, v in self._connections.items() if v.users == 0), None)
                    if victim is None:
                        raise ChatGPTOAuthUpstreamError(429, "All SIWC session channels are busy")
                    expired.append(self._connections.pop(victim))
                entry = _Connection()
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
        entry = self._reserve(key, payload.get("input", []))
        history = entry.history
        output: list[dict[str, Any]] = []
        token_hash = hashlib.sha256(access_token.encode()).hexdigest()
        request_id = uuid.uuid4().hex[:12]
        session_hash = hashlib.sha256(key[1].encode()).hexdigest()[:12]
        started = time.monotonic()
        first_event_ms = first_delta_ms = None
        completed = False
        reused = False
        _LOG.info("SIWC WS start request=%s model=%s session=%s branch=%s active=%s",
                  request_id, key[2], session_hash, key[3][:12], self.active_requests)
        try:
            with entry.lock:
                try:
                    if cancel is not None:
                        cancel.check()
                    reused = (entry.socket is not None and getattr(entry.socket, "state", State.OPEN) == State.OPEN
                              and entry.token_hash == token_hash
                              and time.monotonic() - entry.opened_at < 3300.0)
                    if not reused:
                        entry.close()
                        entry.socket = entry.stack.enter_context(connect(
                            SIWC_WEBSOCKET_URL,
                            additional_headers={"Authorization": f"Bearer {access_token}"},
                            proxy=os.getenv("https_proxy") or os.getenv("HTTPS_PROXY"),
                            user_agent_header=None,
                            open_timeout=min(timeout, 15.0),
                            close_timeout=3.0,
                        ))
                        entry.token_hash = token_hash
                        entry.opened_at = time.monotonic()
                    socket = entry.socket

                    def stop() -> None:
                        # Invalidate before close so a following request cannot reuse a closing channel.
                        if not completed:
                            entry.token_hash = ""
                            socket.close()

                    with cancel.bind(stop) if cancel is not None else nullcontext():
                        wire = {k: v for k, v in payload.items() if k != "stream"}
                        wire["type"] = "response.create"
                        socket.send(json.dumps(wire))
                        completed = False
                        try:
                            while True:
                                raw = socket.recv(timeout=timeout)
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
                                    if completed:
                                        response = event.get("response")
                                        final_output = response.get("output") if isinstance(response, dict) else None
                                        entry.history = history
                                        try:
                                            entry.output_key = _output_key(final_output or output)
                                        except (TypeError, ValueError, KeyError, AttributeError):
                                            # Невалидный ответ проверит provider, не подбор канала.
                                            entry.output_key = None
                                        usage = response.get("usage") if isinstance(response, dict) else None
                                        if isinstance(usage, dict):
                                            details = usage.get("input_tokens_details")
                                            cached = details.get("cached_tokens") if isinstance(details, dict) else None
                                            counters = (usage.get("input_tokens"), cached, usage.get("output_tokens"))
                                            if all(v is None or (isinstance(v, int) and not isinstance(v, bool))
                                                   for v in counters):
                                                _LOG.info("SIWC WS model=%s input=%s cached=%s output=%s session=%s "
                                                          "branch=%s request=%s elapsed_ms=%s first_event_ms=%s "
                                                          "first_delta_ms=%s reused=%s",
                                                          payload.get("model"), *counters,
                                                          session_hash, key[3][:12], request_id, elapsed_ms,
                                                          first_event_ms, first_delta_ms, reused)
                                    yield event
                                    return
                                yield event
                        finally:
                            if not completed:
                                entry.close()
                except InvalidStatus as exc:
                    entry.close()
                    headers = http.client.HTTPMessage()
                    for name, value in exc.response.headers.raw_items():
                        headers[name] = value
                    raise urllib.error.HTTPError(
                        SIWC_WEBSOCKET_URL, exc.response.status_code, "SIWC WebSocket handshake failed",
                        headers, io.BytesIO(exc.response.body),
                    ) from exc
                except (WebSocketException, TimeoutError, OSError) as exc:
                    entry.close()
                    raise OSError(str(exc)) from exc
        finally:
            if not completed:
                _LOG.info("SIWC WS interrupted request=%s model=%s session=%s branch=%s elapsed_ms=%s "
                          "first_event_ms=%s first_delta_ms=%s",
                          request_id, key[2], session_hash, key[3][:12],
                          round((time.monotonic() - started) * 1000), first_event_ms, first_delta_ms)
            with self._lock:
                entry.users -= 1
                entry.last_used = time.monotonic()

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
                _LOG.warning("SIWC WebSocket cleanup failed", exc_info=True)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._retire_locked(list(self._connections.values()))
            self._connections.clear()
            workers = list(self._closers)
        # Запрос не ждёт idle-close; явная остановка дожидается всей очереди.
        for worker in workers:
            worker.join()
