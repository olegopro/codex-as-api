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
from collections import OrderedDict
from collections.abc import Iterator
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
    """One serialized channel per account/session/model, with lazy idle/LRU eviction."""

    def __init__(self, *, limit: int = 16, idle_timeout: float = 1800.0) -> None:
        self._limit = limit
        self._idle_timeout = idle_timeout
        self._lock = threading.Lock()
        self._connections: OrderedDict[tuple[str, str, str], _Connection] = OrderedDict()
        self._closed = False

    def _reserve(self, key: tuple[str, str, str]) -> _Connection:
        expired = []
        now = time.monotonic()
        with self._lock:
            if self._closed:
                raise ChatGPTOAuthUpstreamError(503, "SIWC transport is closed")
            for old_key, candidate in list(self._connections.items()):
                if candidate.users == 0 and now - candidate.last_used >= self._idle_timeout:
                    expired.append(self._connections.pop(old_key))
            entry = self._connections.get(key)
            if entry is None:
                if len(self._connections) >= self._limit:
                    victim = next((k for k, v in self._connections.items() if v.users == 0), None)
                    if victim is None:
                        raise ChatGPTOAuthUpstreamError(429, "All SIWC session channels are busy")
                    expired.append(self._connections.pop(victim))
                entry = _Connection()
                self._connections[key] = entry
            entry.users += 1
            self._connections.move_to_end(key)
        for candidate in expired:
            candidate.close()
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
        key = (account_id, payload.get("prompt_cache_key") or "anonymous", payload.get("model", ""))
        entry = self._reserve(key)
        token_hash = hashlib.sha256(access_token.encode()).hexdigest()
        try:
            with entry.lock:
                try:
                    if cancel is not None:
                        cancel.check()
                    if (entry.socket is None or getattr(entry.socket, "state", State.OPEN) != State.OPEN
                            or entry.token_hash != token_hash
                            or time.monotonic() - entry.opened_at >= 3300.0):
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
                                                _LOG.info("SIWC WS model=%s input=%s cached=%s output=%s session=%s",
                                                          payload.get("model"), *counters,
                                                          hashlib.sha256(key[1].encode()).hexdigest()[:12])
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
            with self._lock:
                entry.users -= 1
                entry.last_used = time.monotonic()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            entries = list(self._connections.values())
            self._connections.clear()
        for entry in entries:
            entry.close()
