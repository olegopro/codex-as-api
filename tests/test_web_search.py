from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock

import anyio
import pytest
from fastapi.testclient import TestClient

from codex_as_api import server, web_search
from codex_as_api.auth import ChatGPTOAuthInvalidRequestError


def body(*, stream=False):
    return {"model": "gpt-6.1-sol", "max_tokens": 128000, "stream": stream,
            "messages": [{"role": "user", "content": [{"type": "text", "text":
                "Perform a web search for the query: Python documentation"}]}],
            "system": [{"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK."},
                       {"type": "text", "text": "You run a web search for another Claude model, which is waiting "
                        "on the result and can search again. Search once, then reply with a brief summary "
                        "of the results."}],
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 8}],
            "tool_choice": {"type": "auto"}, "output_config": {"effort": "low"}}


@pytest.fixture(autouse=True)
def configured(monkeypatch):
    monkeypatch.setenv("CODEX_AS_API_WEB_SEARCH_COMMAND", json.dumps([sys.executable, "/fixture-search.py"]))
    monkeypatch.setattr(server, "_get_provider", lambda: object())


@pytest.mark.parametrize("stream", [False, True])
def test_builtin_search_keeps_websearch_and_returns_plain_source_links(monkeypatch, stream):
    search = AsyncMock(return_value=[{"title": "Python docs", "url": "https://docs.python.org/3/",
                                     "snippet": "Real source excerpt", "date": ""}])
    monkeypatch.setattr(web_search, "execute", search)
    response = TestClient(server.app).post("/v1/messages", json=body(stream=stream))
    assert response.status_code == 200, response.text
    assert "https://docs.python.org/3/" in response.text
    assert "Real source excerpt" in response.text
    assert "encrypted_content" not in response.text
    assert search.await_count == 1
    if stream:
        assert "event: message_stop" in response.text
    else:
        assert response.json()["stop_reason"] == "end_turn"
        assert response.json()["usage"]["input_tokens"] == 0


def test_plain_user_text_cannot_trigger_search_backend():
    request = body()
    request.pop("system")
    assert web_search.builtin_request(request) is None
    request = body()
    request["tools"] = [{"name": "WebSearch", "input_schema": {"type": "object"}}]
    assert web_search.builtin_request(request) is None


def test_domain_filters_are_enforced_and_not_substring_matches():
    request = web_search.SearchRequest("query", allowed=("python.org",))
    rows = {"results": [{"url": url, "title": "Title", "snippet": "", "date": ""} for url in [
        "https://docs.python.org/3/", "https://python.org.evil.test/", "https://evilpython.org/",
        "https://python.org@evil.test/", "javascript:alert(1)", "https://python.org/3/",
    ]]}
    assert [r["url"] for r in web_search._safe_rows(rows, request)] == [
        "https://docs.python.org/3/", "https://python.org/3/",
    ]
    blocked = web_search.SearchRequest("query", blocked=("python.org",))
    assert [r["url"] for r in web_search._safe_rows(rows, blocked)] == [
        "https://python.org.evil.test/", "https://evilpython.org/",
    ]


@pytest.mark.parametrize("tool", [
    {"max_uses": True}, {"max_uses": -1}, {"user_location": {"country": "US"}},
    {"allowed_domains": ["https://python.org/path"]},
    {"allowed_domains": ["python.org"], "blocked_domains": ["example.com"]},
])
def test_unsupported_search_options_are_not_silently_ignored(tool):
    request = body()
    request["tools"][0].update(tool)
    with pytest.raises(ChatGPTOAuthInvalidRequestError):
        web_search.builtin_request(request)


@pytest.mark.parametrize("stream", [False, True])
def test_search_failure_is_not_empty_success(monkeypatch, stream):
    monkeypatch.setattr(web_search, "execute", AsyncMock(side_effect=ValueError("SECRET query")))
    response = TestClient(server.app, raise_server_exceptions=False).post("/v1/messages", json=body(stream=stream))
    assert response.status_code == 502
    assert "SECRET" not in response.text


def test_backend_timeout_is_explicit_and_sanitized(monkeypatch):
    monkeypatch.setattr(web_search, "execute", AsyncMock(side_effect=TimeoutError("SECRET query")))
    response = TestClient(server.app).post("/v1/messages", json=body())
    assert response.status_code == 504
    assert "SECRET" not in response.text


def test_helper_timeout_reaps_child(monkeypatch, tmp_path):
    helper = tmp_path / "timeout.py"
    helper.write_text("import time\ntime.sleep(30)\n")
    monkeypatch.setenv("CODEX_AS_API_WEB_SEARCH_COMMAND", json.dumps([sys.executable, str(helper)]))
    monkeypatch.setattr(web_search, "_TIMEOUT", 0.01)
    processes = []
    original = web_search.asyncio.create_subprocess_exec

    async def exercise():
        async def create(*args, **kwargs):
            process = await original(*args, **kwargs)
            processes.append(process)
            return process

        monkeypatch.setattr(web_search.asyncio, "create_subprocess_exec", create)
        with pytest.raises(TimeoutError):
            await web_search.execute(web_search.SearchRequest("query"))
        assert processes[0].returncode is not None

    anyio.run(exercise)


def test_zero_max_uses_does_not_execute_search(monkeypatch):
    search = AsyncMock()
    monkeypatch.setattr(web_search, "execute", search)
    request = body()
    request["tools"][0]["max_uses"] = 0
    response = TestClient(server.app).post("/v1/messages", json=request)
    assert response.status_code == 200
    assert "max_uses_exceeded" in response.text
    search.assert_not_awaited()


def test_command_uses_json_stdin_without_shell_or_query_argv(monkeypatch, tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text("import json,sys\nd=json.load(sys.stdin)\nprint(json.dumps({'results':[{'title':d['query'],"
                      "'url':'https://docs.python.org/3/','snippet':'ok','date':''}]}))\n")
    monkeypatch.setenv("CODEX_AS_API_WEB_SEARCH_COMMAND", json.dumps([sys.executable, str(helper)]))
    request = web_search.SearchRequest("safe ; $(never-run)", allowed=("python.org",))
    rows = anyio.run(web_search.execute, request)
    assert rows[0]["title"] == "safe ; $(never-run) (site:python.org)"


def test_disconnect_reaps_search_child(monkeypatch, tmp_path):
    helper = tmp_path / "waiting.py"
    helper.write_text("import time\ntime.sleep(30)\n")
    monkeypatch.setenv("CODEX_AS_API_WEB_SEARCH_COMMAND", json.dumps([sys.executable, str(helper)]))
    processes = []
    original = web_search.asyncio.create_subprocess_exec

    async def exercise():
        from starlette.requests import Request

        entered = anyio.Event()

        async def create(*args, **kwargs):
            process = await original(*args, **kwargs)
            processes.append(process)
            entered.set()
            return process

        monkeypatch.setattr(web_search.asyncio, "create_subprocess_exec", create)

        async def receive():
            await entered.wait()
            return {"type": "http.disconnect"}

        request = Request({"type": "http", "headers": []}, receive=receive)
        with anyio.fail_after(2):
            result = await server._builtin_search_message(request, web_search.SearchRequest("query"), "gpt-6.1-sol")
        assert result is None
        assert processes[0].returncode is not None

    anyio.run(exercise)


def test_nested_search_result_ids_are_unique(monkeypatch):
    monkeypatch.setattr(web_search, "execute", AsyncMock(return_value=[]))
    request = web_search.SearchRequest("query")
    first = anyio.run(web_search.search_message, request, "gpt-6.1-sol")
    second = anyio.run(web_search.search_message, request, "gpt-6.1-sol")
    assert first["id"] != second["id"]
    assert "no matching results" in first["content"][-1]["text"]
    assert first["content"][0]["type"] == "server_tool_use"
    assert first["content"][0]["id"] != second["content"][0]["id"]
