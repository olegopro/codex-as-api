"""Opt-in, content-free lifecycle diagnostics for HTTP and SIWC requests."""
from __future__ import annotations

import contextvars
import fcntl
import json
import logging
import os
import re
import stat
import time
import uuid
from datetime import datetime, timezone
from typing import Any


class _PrivateRotatingFileHandler(logging.FileHandler):
    """Rotate private segments; prune only this handler's own names."""

    max_bytes = 5 * 1024 * 1024
    max_total_bytes = 256 * 1024 * 1024
    retention = 7 * 24 * 60 * 60

    def __init__(self, path: str) -> None:
        self._owner_pid = os.getpid()
        self._lock_fd = self._open_lock(path)
        try:
            super().__init__(path, mode="a", encoding="utf-8")
            assert self.stream is not None
            current = os.fstat(self.stream.fileno())
            if (current.st_size > self.max_total_bytes
                    or (current.st_size and current.st_mtime < time.time() - self.retention)):
                self._rollover()
            self._prune(0)
        except BaseException:
            if hasattr(self, "stream") and self.stream is not None:
                self.stream.close()
            os.close(self._lock_fd)
            self._lock_fd = -1
            raise

    @staticmethod
    def _open_lock(path: str) -> int:
        fd = os.open(path + ".lock", os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("diagnostics lock must be a regular file")
            os.fchmod(fd, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("diagnostics path is already in use") from exc
            return fd
        except BaseException:
            os.close(fd)
            raise

    def close(self) -> None:
        super().close()
        if self._lock_fd >= 0:
            os.close(self._lock_fd)
            self._lock_fd = -1

    def _rollover(self) -> None:
        assert self.stream is not None
        self.stream.close()
        self.stream = None
        os.rename(self.baseFilename, self.baseFilename + ".segment." + uuid.uuid4().hex)
        self.stream = self._open()

    def _open(self) -> Any:
        fd = os.open(self.baseFilename, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("diagnostics path must be a regular file")
            os.fchmod(fd, 0o600)
            return os.fdopen(fd, self.mode, encoding=self.encoding, errors=self.errors)
        except BaseException:
            os.close(fd)
            raise

    def _segments(self) -> list[tuple[float, str, int]]:
        directory, base = os.path.split(self.baseFilename)
        pattern = re.compile(re.escape(base) + r"(?:\.segment\.[0-9a-f]{32}|\.[123])\Z")
        segments = []
        for item in os.scandir(directory):
            if pattern.fullmatch(item.name):
                info = item.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("diagnostics files must be regular files")
                fd = os.open(item.path, os.O_RDONLY | os.O_NOFOLLOW)
                try:
                    if not stat.S_ISREG(os.fstat(fd).st_mode):
                        raise ValueError("diagnostics files must be regular files")
                    os.fchmod(fd, 0o600)
                finally:
                    os.close(fd)
                segments.append((info.st_mtime, item.path, info.st_size))
        return sorted(segments)

    def _prune(self, incoming: int) -> None:
        assert self.stream is not None
        cutoff = time.time() - self.retention
        segments = self._segments()
        total = os.fstat(self.stream.fileno()).st_size + sum(size for _, _, size in segments)
        lost_coverage = False
        for modified, path, size in segments:
            if modified < cutoff or total + incoming > self.max_total_bytes:
                if modified >= cutoff:
                    lost_coverage = True
                os.unlink(path)
                total -= size
        if lost_coverage or total + incoming > self.max_total_bytes:
            logging.getLogger("codex_as_api").warning("diagnostics retention incomplete category=capacity")

    def emit(self, record: logging.LogRecord) -> None:
        if os.getpid() != self._owner_pid:
            logging.getLogger("codex_as_api").warning("diagnostics write failed category=process_changed")
            return
        try:
            if self.stream is None:
                # Uvicorn's dictConfig closes existing handlers, but leaves them on the logger.
                if self._lock_fd < 0:
                    self._lock_fd = self._open_lock(self.baseFilename)
                try:
                    self.stream = self._open()
                except BaseException:
                    os.close(self._lock_fd)
                    self._lock_fd = -1
                    raise
                self._closed = False
            message = self.format(record) + self.terminator
            size = len(message.encode("utf-8"))
            current = os.fstat(self.stream.fileno())
            if current.st_size and (current.st_size + size > min(self.max_bytes, self.max_total_bytes)
                                    or current.st_mtime < time.time() - self.retention):
                self._rollover()
            self._prune(size)
            if size <= self.max_total_bytes and os.fstat(self.stream.fileno()).st_size + size <= self.max_total_bytes:
                self.stream.write(message)
                self.flush()
        except Exception:
            logging.getLogger("codex_as_api").warning("diagnostics write failed category=io_error")


def _make_logger() -> logging.Logger | None:
    if os.getenv("CODEX_AS_API_DIAGNOSTICS") != "1":
        return None
    path = os.getenv("CODEX_AS_API_DIAGNOSTICS_PATH", "")
    if not os.path.isabs(path):
        raise ValueError("CODEX_AS_API_DIAGNOSTICS_PATH must be an absolute local path")
    if os.path.lexists(path) and not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError("diagnostics files must be regular files")
    logger = logging.getLogger("codex_as_api.diagnostics")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(_PrivateRotatingFileHandler(path))
    return logger


_LOG = _make_logger()
_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar("diagnostics", default=None)

# Strings in log records must come from these enums or identifiers generated by the server.
_ENUMS = {
    "stage": {"http_start", "validation_error", "http_start_response", "http_stream_finished", "http_error",
              "http_cancelled", "http_disconnected", "upstream_start", "upstream_completed", "upstream_error",
              "request_metadata", "channel_selected", "output_key", "channel_released", "cancel_requested",
              "search_start", "search_completed", "search_error", "cache_request", "cache_response",
              "cache_http_request", "cache_http_response", "cache_websocket_request", "client_cache_request"},
    "backend": {"codex", "siwc"},
    "transport": {"http", "websocket"},
    "reason": {"invalid_json", "invalid_content_type", "body_too_large", "invalid_request", "thinking_display_invalid",
               "no_socket", "socket_closed", "credentials_changed", "connection_expired", "reused", "upstream_failed",
               "upstream_incomplete", "upstream_error", "interrupted", "stream_error", "completed",
               "search_unavailable", "search_timeout", "search_failed"},
    "selection_reason": {"new", "existing", "input_mismatch", "missing_key", "replay_mismatch"},
    "output_key_source": {"done", "completed"},
    "output_key_error": {"none", "invalid_output"},
    "thinking_display": {"omitted", "summarized", "updates"},
    "effort": {"none", "minimal", "low", "medium", "high", "xhigh", "max"},
    "route": {"/health", "/v1/models", "/v1/messages", "/v1/messages/count_tokens",
              "/v1/chat/completions", "/v1/responses"},
    "model": {"gpt-5.2", "gpt-5.3-codex", "gpt-5.3-codex-spark", "gpt-5.4", "gpt-5.4-mini", "gpt-5.5",
              "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra", "gpt-6-luna", "gpt-6-chat-thinking",
              "gpt-6.1-sol", "gpt-6.1-terra"},
}
_NUMBERS = {"status", "elapsed_ms", "first_event_ms", "first_delta_ms", "input_tokens", "cached_tokens",
            "output_tokens", "total_tokens", "message_count", "tool_count", "content_items", "body_bytes",
            "free_candidates", "busy_candidates", "done_count", "completed_count", "result_count", "search_requests",
            "network_attempt"}
_IDENTIFIERS = {"http_id", "attempt_id", "session_hash", "branch_hash", "channel_id", "payload_hash",
                "instructions_hash", "tools_hash", "settings_hash", "cache_key_hash", "routing_hash", "wire_hash",
                "session_header_hash", "thread_header_hash", "client_request_header_hash", "account_hash",
                "actual_model_hash", "service_tier_hash", "client_instructions_hash", "client_tools_hash",
                "client_model_hash", "client_settings_hash"}
_UPSTREAM_IDS = {"upstream_request_id": re.compile(r"(?:req_[A-Za-z0-9_-]{1,160}|[0-9a-f-]{32,36})\Z"),
                 "upstream_response_id": re.compile(r"resp_[A-Za-z0-9_-]{1,160}\Z")}
_PROCESS_ID = uuid.uuid4().hex


def _new_process_id() -> None:
    global _PROCESS_ID
    _PROCESS_ID = uuid.uuid4().hex


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_new_process_id)


def emit(stage: str, **fields: Any) -> None:
    if _LOG is None:
        return
    if stage not in _ENUMS["stage"]:
        raise ValueError("unknown diagnostics stage")
    context = _context.get()
    if context is not None:
        fields.setdefault("http_id", context["http_id"])
        fields.setdefault("route", context.get("route"))
        fields.setdefault("session_hash", context.get("session_hash"))
    record: dict[str, Any] = {"timestamp": datetime.now(timezone.utc).isoformat(), "stage": stage,
                              "process_id": _PROCESS_ID}
    for key, value in fields.items():
        if (
            (key in _ENUMS and isinstance(value, str) and value in _ENUMS[key])
            or (key in _NUMBERS and isinstance(value, int) and not isinstance(value, bool) and value >= 0)
            or (key in _IDENTIFIERS and isinstance(value, str) and len(value) in {12, 32}
                and all(char in "0123456789abcdef" for char in value))
            or (key in {"reused", "stream", "usage_available"} and isinstance(value, bool))
            or (key in _UPSTREAM_IDS and isinstance(value, str) and _UPSTREAM_IDS[key].fullmatch(value))
            or (key in {"input_prefix_hashes", "client_prefix_hashes"} and isinstance(value, list)
                and all(isinstance(v, str) and len(v) == 32 and all(c in "0123456789abcdef" for c in v)
                        for v in value))
        ):
            record[key] = value
    _LOG.info(json.dumps(record, separators=(",", ":")))


def safe_model(value: Any) -> str:
    return value if isinstance(value, str) and value in _ENUMS["model"] else "other"


def enabled() -> bool:
    return _LOG is not None


def current() -> dict[str, Any] | None:
    return _context.get()


class LifecycleMiddleware:
    """Observe ASGI messages without consuming or buffering request/response bodies."""

    def __init__(self, app: Any, error_response: Any = None) -> None:
        self.app = app
        self.error_response = error_response

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        state: dict[str, Any] = {"http_id": uuid.uuid4().hex, "started": time.monotonic(),
                                 "failed": False, "route": scope.get("path")}
        token = _context.set(state)
        emit("http_start")

        async def observed_receive() -> Any:
            message = await receive()
            if message["type"] == "http.disconnect" and not state.get("finished"):
                state["disconnected"] = True
            return message

        async def observed_send(message: Any) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                if 400 <= message["status"] < 500:
                    reason = state.get("reason", "invalid_request")
                    logging.getLogger("codex_as_api").warning(
                        "HTTP validation_error http_id=%s status=%s reason=%s",
                        state["http_id"], message["status"], reason,
                    )
                    emit("validation_error", status=message["status"], reason=reason)
                emit("http_start_response", status=message["status"])
            await send(message)
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                state["finished"] = True

        try:
            await self.app(scope, observed_receive, observed_send)
        except Exception as exc:
            state["failed"] = True
            state["raised"] = True
            frames = []
            tb = exc.__traceback__
            while tb is not None:
                filename = tb.tb_frame.f_code.co_filename
                if filename.startswith(os.path.dirname(__file__) + os.sep):
                    frames.append(f"{os.path.basename(filename)}:{tb.tb_lineno}")
                tb = tb.tb_next
            error_type = (type(exc).__name__
                          if type(exc).__module__ in {"builtins", "codex_as_api.auth"} else "Exception")
            logging.getLogger("uvicorn.error").warning("%s category=server_error stack=%s", error_type, frames)
            if "status" not in state:
                if self.error_response is not None:
                    await self.error_response(scope, observed_receive, observed_send, exc)
                else:
                    await observed_send({"type": "http.response.start", "status": 500, "headers": []})
                    await observed_send({"type": "http.response.body", "body": b"Internal Server Error"})
        except BaseException:
            state["failed"] = True
            raise
        finally:
            if state.get("disconnected"):
                stage = "http_disconnected"
            elif state.get("raised"):
                stage = "http_error"
            elif not state.get("finished"):
                stage = "http_cancelled"
            elif state["failed"] or state.get("status", 500) >= 400:
                stage = "http_error"
            else:
                stage = "http_stream_finished"
            emit(stage, status=state.get("status"), elapsed_ms=round((time.monotonic() - state["started"]) * 1000))
            _context.reset(token)
