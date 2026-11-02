"""Локальный поисковый backend для отдельного WebSearch-подзапроса Claude Code."""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import anyio

from .auth import ChatGPTOAuthInvalidRequestError
from .diagnostics import emit as diagnostic_emit

_COMMAND_ENV = "CODEX_AS_API_WEB_SEARCH_COMMAND"
_QUERY_PREFIX = "Perform a web search for the query: "
_SYSTEM_MARKER = "You run a web search for another Claude model"
_TIMEOUT = 35.0
_MAX_OUTPUT = 2 * 1024 * 1024
_DOMAIN = re.compile(r"(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)*[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\Z")


@dataclass(frozen=True)
class SearchRequest:
    query: str
    allowed: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()
    max_uses: int = 8


def _domains(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > 100:
        raise ChatGPTOAuthInvalidRequestError("WebSearch domains must be an array of hostnames")
    result = []
    for domain in value:
        if not isinstance(domain, str):
            raise ChatGPTOAuthInvalidRequestError("WebSearch domains must be hostnames")
        domain = domain.lower().rstrip(".")
        if domain.startswith("*."):
            domain = domain[2:]
        try:
            domain = domain.encode("idna").decode("ascii")
        except UnicodeError:
            domain = ""
        if len(domain) > 253 or not _DOMAIN.fullmatch(domain):
            raise ChatGPTOAuthInvalidRequestError("WebSearch supports hostname domain filters only")
        result.append(domain)
    return tuple(result)


def builtin_request(body: dict[str, Any]) -> SearchRequest | None:
    """Не перехватывает обычные сообщения или произвольные hosted-tools запросы."""
    if not os.getenv(_COMMAND_ENV):
        return None
    tools = body.get("tools")
    if not isinstance(tools, list) or len(tools) != 1 or not isinstance(tools[0], dict):
        return None
    tool = tools[0]
    if tool.get("type") != "web_search_20250305" or tool.get("name") != "web_search":
        return None
    system = body.get("system")
    if not isinstance(system, list) or not any(
        isinstance(b, dict) and b.get("type") == "text" and str(b.get("text", "")).startswith(_SYSTEM_MARKER)
        for b in system
    ):
        return None
    messages = body.get("messages")
    if (not isinstance(messages, list) or len(messages) != 1
            or not isinstance(messages[0], dict) or messages[0].get("role") != "user"):
        return None
    content = messages[0].get("content")
    if (not isinstance(content, list) or len(content) != 1
            or not isinstance(content[0], dict) or content[0].get("type") != "text"):
        return None
    text = content[0].get("text")
    if not isinstance(text, str) or not text.startswith(_QUERY_PREFIX):
        return None
    if set(tool) - {"type", "name", "max_uses", "allowed_domains", "blocked_domains"}:
        raise ChatGPTOAuthInvalidRequestError("Unsupported WebSearch configuration")
    choice = body.get("tool_choice", {"type": "auto"})
    if choice != {"type": "auto"}:
        raise ChatGPTOAuthInvalidRequestError("Builtin WebSearch requires automatic tool choice")
    if any(key in body for key in ("stop_sequences", "output_format", "responses_lite", "reasoning")):
        raise ChatGPTOAuthInvalidRequestError("Unsupported builtin WebSearch request options")
    query = text[len(_QUERY_PREFIX):].strip()
    if not query or len(query) > 8192:
        raise ChatGPTOAuthInvalidRequestError("WebSearch query must contain 1 to 8192 characters")
    uses = tool.get("max_uses", 8)
    if isinstance(uses, bool) or not isinstance(uses, int) or uses < 0:
        raise ChatGPTOAuthInvalidRequestError("WebSearch max_uses must be a non-negative integer")
    allowed = _domains(tool.get("allowed_domains", []))
    blocked = _domains(tool.get("blocked_domains", []))
    if allowed and blocked:
        raise ChatGPTOAuthInvalidRequestError("WebSearch cannot combine allowed and blocked domains")
    return SearchRequest(query, allowed, blocked, uses)


def _matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def _safe_rows(value: Any, request: SearchRequest) -> list[dict[str, str]]:
    if not isinstance(value, dict) or not isinstance(value.get("results"), list):
        raise ValueError("invalid search response")
    rows = []
    seen = set()
    for row in value["results"]:
        if (not isinstance(row, dict)
                or not all(isinstance(row.get(k, ""), str) for k in ("title", "url", "snippet", "date"))):
            raise ValueError("invalid search result")
        url = row.get("url", "")
        if any(c.isspace() or ord(c) < 32 or c in "<>\\\\" for c in url):
            continue
        try:
            parsed = urlsplit(url)
            host = (parsed.hostname or "").encode("idna").decode("ascii").lower().rstrip(".")
            if parsed.scheme not in {"https", "http"} or not host or parsed.username or parsed.password:
                continue
        except (ValueError, UnicodeError):
            continue
        if request.allowed and not any(_matches(host, d) for d in request.allowed):
            continue
        if any(_matches(host, d) for d in request.blocked) or url in seen:
            continue
        seen.add(url)
        rows.append({k: row.get(k, "") for k in ("title", "url", "snippet", "date")})
        if len(rows) == 10:
            break
    return rows


async def execute(request: SearchRequest) -> list[dict[str, str]]:
    command = json.loads(os.environ[_COMMAND_ENV])
    if (not isinstance(command, list) or not command or len(command) > 16
            or not all(isinstance(v, str) and os.path.isabs(v) for v in command)):
        raise ValueError("invalid local search command")
    query = request.query
    if request.allowed:
        query += " (" + " OR ".join("site:" + d for d in request.allowed) + ")"
    query += "".join(" -site:" + d for d in request.blocked)
    process = await asyncio.create_subprocess_exec(
        *command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    async def exchange() -> bytes:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(json.dumps({"query": query, "count": 20}).encode())
        await process.stdin.drain()
        process.stdin.close()
        chunks = []
        size = 0
        while chunk := await process.stdout.read(65536):
            size += len(chunk)
            if size > _MAX_OUTPUT:
                raise ValueError("search response too large")
            chunks.append(chunk)
        await process.wait()
        return b"".join(chunks)

    try:
        stdout = await asyncio.wait_for(exchange(), timeout=_TIMEOUT)
        if process.returncode:
            raise ValueError("search backend failed")
        return _safe_rows(json.loads(stdout), request)
    finally:
        if process.returncode is None:
            with anyio.CancelScope(shield=True):
                process.kill()
                await process.wait()


def _text(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "DeepSeek Search: no matching results were found."
    parts = ["DeepSeek Search results (titles, links and source excerpts; not Anthropic hosted search):"]
    for n, row in enumerate(rows, 1):
        title = " ".join(row["title"].split()).replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
        parts.append(f"{n}. [{title or row['url']}]({row['url']})\n{row['snippet']}"[:16000])
    return "\n\n".join(parts)


async def search_message(request: SearchRequest, model: str) -> dict[str, Any]:
    started = time.monotonic()
    diagnostic_emit("search_start")
    if request.max_uses == 0:
        text = "WebSearch error: max_uses_exceeded; no search was performed."
        diagnostic_emit("search_error", reason="search_failed", search_requests=0)
    else:
        try:
            rows = await execute(request)
        except TimeoutError:
            diagnostic_emit("search_error", reason="search_timeout", search_requests=1)
            raise
        except (OSError, ValueError, KeyError):
            diagnostic_emit("search_error", reason="search_failed", search_requests=1)
            raise
        text = _text(rows)
        diagnostic_emit("search_completed", result_count=len(rows), search_requests=1,
                        elapsed_ms=round((time.monotonic() - started) * 1000))
    content: list[dict[str, Any]] = []
    if request.max_uses > 0:
        # Факт реально выполненного поиска: клиент использует этот блок для searchCount.
        content.append({"type": "server_tool_use", "id": "srvtoolu_" + uuid.uuid4().hex[:24],
                        "name": "web_search", "input": {"query": request.query}})
    content.append({"type": "text", "text": text})
    return {"id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": "end_turn", "stop_sequence": None,
            # Это прямой поиск без LLM-инференса, а не выдуманная оценка токенов провайдера.
            "usage": {"input_tokens": 0, "output_tokens": 0,
                      "server_tool_use": {"web_search_requests": int(request.max_uses > 0)}}}


def message_events(message: dict[str, Any]) -> Iterator[str]:
    def event(kind: str, **fields: Any) -> str:
        return f"event: {kind}\ndata: {json.dumps({'type': kind, **fields}, ensure_ascii=False)}\n\n"

    yield event("message_start", message={**message, "content": [], "stop_reason": None})
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            initial: dict[str, Any] = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:
            initial = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"], ensure_ascii=False)}
        yield event("content_block_start", index=index, content_block=initial)
        yield event("content_block_delta", index=index, delta=delta)
        yield event("content_block_stop", index=index)
    yield event("message_delta", delta={"stop_reason": "end_turn", "stop_sequence": None}, usage=message["usage"])
    yield event("message_stop")
