"""Opt-in Legacy routing experiment; hash-only, exclusive history leases."""
from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict, deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from . import cache_diagnostics
from .diagnostics import emit as diagnostic_emit
from .siwc_transport import _branch_key, _fingerprint, _is_output_item, _output_key


@dataclass
class _Anchor:
    size: int
    prefix: str
    output_key: bytes | None
    turn_state: str | None
    owner_hash: bytes
    config: bytes


@dataclass
class _History:
    thread_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    busy: bool = False
    used_at: float = 0.0
    anchors: deque[_Anchor] = field(default_factory=lambda: deque(maxlen=16))


@dataclass
class LegacyLease:
    entry: _History
    history: tuple[int, str]
    decision: str
    new_turn: bool
    owner_hash: bytes
    config: bytes
    turn_state: str | None = None
    rejected: bool = False
    active_histories: int = 1
    mixed_user_tool: bool = False
    new_turn_v2: bool = True
    output: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False

    def headers(self, *, threads: bool = True) -> dict[str, str]:
        result = ({"thread-id": self.entry.thread_id, "x-client-request-id": self.entry.thread_id}
                  if threads else {})
        if self.turn_state is not None:
            result["x-codex-turn-state"] = self.turn_state
        return result

    def response_headers(self, headers: Any) -> None:
        value = headers.get("x-codex-turn-state") if headers else None
        if (self.turn_state is None and not self.rejected and isinstance(value, str) and 0 < len(value) <= 8192
                and all("!" <= ch <= "~" for ch in value)):
            self.turn_state = value

    def reject_turn_state(self) -> None:
        previous = self.turn_state
        self.rejected = True
        self.turn_state = None
        for anchor in self.entry.anchors:
            if anchor.turn_state == previous:
                anchor.turn_state = None

    def observe(self, event: dict[str, Any], *, echoed: bool = False) -> None:
        response = event.get("response")
        error = response.get("error") if isinstance(response, dict) else None
        failed = (event.get("type") in {"error", "response.failed"} or
                  event.get("type") == "response.incomplete" and isinstance(error, dict) and bool(error.get("code")))
        if echoed and failed:
            self.reject_turn_state()
            diagnostic_emit("turn_state_rejected", rejection_source="sse_terminal")
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

    def reserve(self, payload: dict[str, Any], account_id: str, owner: str) -> LegacyLease | None:
        session = payload.get("prompt_cache_key")
        inputs = payload.get("input")
        if not isinstance(session, str) or not session or not isinstance(inputs, list) or not inputs:
            return None
        key = (account_id, session, _branch_key(payload))
        history = cache_diagnostics._prefixes(inputs)
        owner_hash = _fingerprint(owner)
        config = _fingerprint({k: payload.get(k) for k in ("model", "service_tier")})
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
                # Only the newest observation of an identical input is authoritative.
                seen = set()
                for index in reversed(range(len(entry.anchors))):
                    anchor = entry.anchors[index]
                    size, previous, output_key = anchor.size, anchor.prefix, anchor.output_key
                    if (size, previous) in seen:
                        continue
                    seen.add((size, previous))
                    if len(history) == size and history[-1] == previous:
                        candidates.append((size, entry_key, index, False, "reused_exact"))
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
                            candidates.append((size, entry_key, index, new_turn, "reused_replay"))
            best_size = max((candidate[0] for candidate in candidates), default=-1)
            best = {(entry_key, index, new_turn, decision) for size, entry_key, index, new_turn, decision in candidates
                    if size == best_size}
            parent = None
            turn_state = None
            new_turn_v2 = True
            if len(best) == 1:
                entry_key, index, new_turn, decision = best.pop()
                entry = self._entries[entry_key]
                parent = entry.anchors[index]
                if decision == "reused_exact":
                    new_turn_v2 = False
                else:
                    tail = inputs[parent.size:]
                    replay = []
                    for item in tail:
                        if not _is_output_item(item):
                            break
                        replay.append(item)
                    calls = {item.get("call_id") for item in replay if item.get("type") == "function_call"}
                    results = [item for item in tail[len(replay):] if item.get("type") == "function_call_output"]
                    fulfilled = {item.get("call_id") for item in results}
                    aborted = any(isinstance(item.get("output"), str) and item["output"].startswith("[tool_error]\n")
                                  for item in results)
                    new_turn_v2 = not bool(calls and calls <= fulfilled and not (aborted and new_turn))
                if not new_turn and parent.config == config and parent.owner_hash == owner_hash:
                    turn_state = parent.turn_state
            else:
                decision = ("new_ambiguous" if best else "new_no_anchor" if not group else
                            "new_busy" if all(v.busy for _, v in group) else "new_replay_mismatch")
                # Stable normal branch; competing histories require distinct lanes.
                thread_id = _fingerprint(key).hex()[:32] if not group else uuid.uuid4().hex
                entry = _History(thread_id=thread_id)
                entry_key = (*key, entry.thread_id)
                self._entries[entry_key] = entry
                new_turn = True
            entry.busy, entry.used_at = True, now
            self._entries.move_to_end(entry_key)
            # Capacity bounds idle metadata only; active histories are never evicted.
            for old_key in list(self._entries):
                if len(self._entries) <= self._capacity:
                    break
                if not self._entries[old_key].busy:
                    del self._entries[old_key]
            return LegacyLease(entry, (len(history), history[-1]), decision, new_turn,
                               owner_hash, config, turn_state, active_histories=sum(
                                   v.busy for k, v in self._entries.items() if k[:2] == key[:2]),
                               mixed_user_tool=bool(new_turn and parent is not None and any(
                                   item.get("type") == "function_call_output" for item in inputs[parent.size:])),
                               new_turn_v2=new_turn_v2)

    def release(self, lease: LegacyLease) -> None:
        output_key = None
        try:
            # Provider validation owns the original error; cleanup must not mask it.
            with suppress(TypeError, ValueError, KeyError, AttributeError):
                output_key = _output_key(lease.output) if lease.completed else None
        finally:
            with self._lock:
                lease.entry.anchors.append(_Anchor(*lease.history, output_key, lease.turn_state,
                                                   lease.owner_hash, lease.config))
                lease.entry.busy = False
                lease.entry.used_at = time.monotonic()
            lease.output.clear()
