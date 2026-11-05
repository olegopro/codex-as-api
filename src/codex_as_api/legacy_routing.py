"""Opt-in Legacy routing experiment; hash-only, exclusive history leases."""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

from . import cache_diagnostics
from .siwc_transport import _branch_key, _fingerprint, _is_output_item, _output_key


@dataclass
class _History:
    thread_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    busy: bool = False
    used_at: float = 0.0
    turn_state: str | None = None
    config: bytes | None = None
    token_hash: bytes | None = None
    anchors: deque[tuple[int, str, bytes | None]] = field(default_factory=lambda: deque(maxlen=4))


@dataclass
class LegacyLease:
    entry: _History
    history: tuple[int, str]
    decision: str
    new_turn: bool
    output: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False

    def headers(self) -> dict[str, str]:
        result = {"thread-id": self.entry.thread_id, "x-client-request-id": self.entry.thread_id}
        if self.entry.turn_state is not None:
            result["x-codex-turn-state"] = self.entry.turn_state
        return result

    def response_headers(self, headers: Any) -> None:
        value = headers.get("x-codex-turn-state") if headers else None
        if (self.entry.turn_state is None and isinstance(value, str) and 0 < len(value) <= 8192
                and all("!" <= ch <= "~" for ch in value)):
            self.entry.turn_state = value

    def observe(self, event: dict[str, Any]) -> None:
        if event.get("type") == "response.output_item.done" and isinstance(event.get("item"), dict):
            self.output.append(event["item"])
        elif event.get("type") == "response.completed":
            response = event.get("response")
            output = response.get("output") if isinstance(response, dict) else None
            if isinstance(output, list) and output:
                self.output = output
            self.completed = True


class LegacyRouting:
    """A free history is reused only after matching the full prior input and replay."""

    def __init__(self, *, capacity: int = 256, idle_timeout: float = 1800) -> None:
        self._entries: OrderedDict[tuple[str, str, str, str], _History] = OrderedDict()
        self._lock = threading.Lock()
        self._capacity = capacity
        self._idle_timeout = idle_timeout

    def reserve(self, payload: dict[str, Any], account_id: str, token: str) -> LegacyLease | None:
        session = payload.get("prompt_cache_key")
        inputs = payload.get("input")
        if not isinstance(session, str) or not session or not isinstance(inputs, list) or not inputs:
            return None
        key = (account_id, session, _branch_key(payload))
        history = cache_diagnostics._prefixes(inputs)
        token_hash = _fingerprint(token)
        config = _fingerprint({k: v for k, v in payload.items() if k != "input"})
        now = time.monotonic()
        with self._lock:
            for old_key, old in list(self._entries.items()):
                if not old.busy and now - old.used_at >= self._idle_timeout:
                    del self._entries[old_key]
            candidates = []
            group = [(k, v) for k, v in self._entries.items() if k[:3] == key]
            for entry_key, entry in group:
                if entry.busy:
                    continue
                for size, previous, output_key in entry.anchors:
                    if len(history) == size and history[-1] == previous:
                        candidates.append((size, entry_key, False, "reused_exact"))
                    elif len(history) > size and history[size - 1] == previous and output_key is not None:
                        replay = []
                        for item in inputs[size:]:
                            if not _is_output_item(item):
                                break
                            replay.append(item)
                        if _output_key(replay) == output_key:
                            # User text following replay starts a new turn. Tool outputs
                            # and developer notes continue it. Unknown branches get a lease.
                            new_turn = any(item.get("role") == "user" for item in inputs[size + len(replay):])
                            candidates.append((size, entry_key, new_turn, "reused_replay"))
            best_size = max((candidate[0] for candidate in candidates), default=-1)
            best = {(entry_key, new_turn, decision) for size, entry_key, new_turn, decision in candidates
                    if size == best_size}
            if len(best) == 1:
                entry_key, new_turn, decision = best.pop()
                entry = self._entries[entry_key]
                if new_turn or entry.config != config or entry.token_hash != token_hash:
                    entry.turn_state = None
            else:
                decision = ("new_ambiguous" if best else "new_no_anchor" if not group else
                            "new_busy" if all(v.busy for _, v in group) else "new_replay_mismatch")
                # Stable normal branch; competing histories require distinct lanes.
                thread_id = _fingerprint(key).hex()[:32] if not group else uuid.uuid4().hex
                entry = _History(thread_id=thread_id)
                entry_key = (*key, entry.thread_id)
                self._entries[entry_key] = entry
                new_turn = True
            entry.busy, entry.config, entry.used_at, entry.token_hash = True, config, now, token_hash
            self._entries.move_to_end(entry_key)
            # Capacity bounds idle metadata only; active histories are never evicted.
            for old_key in list(self._entries):
                if len(self._entries) <= self._capacity:
                    break
                if not self._entries[old_key].busy:
                    del self._entries[old_key]
            return LegacyLease(entry, (len(history), history[-1]), decision, new_turn)

    def release(self, lease: LegacyLease) -> None:
        try:
            output_key = _output_key(lease.output) if lease.completed else None
        except (TypeError, ValueError, KeyError, AttributeError):
            # Provider validation owns the original error; cleanup must not mask it.
            output_key = None
        with self._lock:
            lease.entry.anchors.append((*lease.history, output_key))
            lease.entry.busy = False
            lease.entry.used_at = time.monotonic()
        lease.output.clear()
