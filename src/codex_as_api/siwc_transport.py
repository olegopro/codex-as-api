"""Bounded session affinity for public SIWC WebSockets; every turn sends full input."""
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
from collections import OrderedDict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
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


@dataclass
class _Connection:
    lock: Any = field(default_factory=threading.Lock)
    users: int = 0
    last_used: float = 0.0
    opened_at: float = 0.0
    token_hash: str = ""
    stack: ExitStack = field(default_factory=ExitStack)
    socket: Any = None

    def close(self) -> None:
        try:
            self.stack.close()
        finally:
            self.stack = ExitStack()
            self.socket = None
            self.token_hash = ""


class SIWCWebSocketPool:
    """Warm channels per account/session/model/branch, with bounded concurrent leases."""

    def __init__(self, *, limit: int = 32, idle_timeout: float = 1800.0) -> None:
        self._limit = limit
        self._idle_timeout = idle_timeout
        self._lock = threading.Lock()
        self._connections: OrderedDict[tuple[str, str, str, str, int], _Connection] = OrderedDict()
        self._closed = False

    @property
    def max_connections(self) -> int:
        return self._limit

    @property
    def active_requests(self) -> int:
        with self._lock:
            return sum(entry.users for entry in self._connections.values())

    def _reserve(self, key: tuple[str, str, str, str]) -> _Connection:
        expired = []
        now = time.monotonic()
        with self._lock:
            if self._closed:
                raise ChatGPTOAuthUpstreamError(503, "SIWC transport is closed")
            for old_key, candidate in list(self._connections.items()):
                if candidate.users == 0 and now - candidate.last_used >= self._idle_timeout:
                    expired.append(self._connections.pop(old_key))
            available = next((k for k, v in reversed(self._connections.items())
                              if k[:4] == key and v.users == 0), None)
            if available is None:
                if len(self._connections) >= self._limit:
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
            self._connections.move_to_end(available)
        _close_connections(expired)
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
        entry = self._reserve(key)
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
                                if event["type"] in _TERMINAL_EVENTS:
                                    completed = event["type"] == "response.completed"
                                    if completed:
                                        response = event.get("response")
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

    def close(self) -> None:
        with self._lock:
            self._closed = True
            entries = list(self._connections.values())
            self._connections.clear()
        _close_connections(entries)


def _close_connections(entries: list[_Connection]) -> None:
    if len(entries) < 2:
        for entry in entries:
            entry.close()
        return
    # Bound cleanup time without keeping another thread pool alive between calls.
    with ThreadPoolExecutor(max_workers=min(8, len(entries))) as workers:
        list(workers.map(lambda entry: entry.close(), entries))
