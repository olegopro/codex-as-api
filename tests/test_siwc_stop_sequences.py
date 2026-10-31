from __future__ import annotations

import json

import pytest

from codex_as_api.anthropic_adapter import anthropic_stream_adapter, internal_response_to_anthropic
from codex_as_api.auth import ChatGPTOAuthProtocolError
from codex_as_api.messages import AssistantResponse, Usage

from . import test_siwc_server

siwc_client = test_siwc_server.siwc_client
siwc_provider = test_siwc_server.siwc_provider


def events_for(parts):
    for part in parts:
        yield {"type": "content", "text": part}
    yield {"type": "finish", "finish_reason": "stop",
           "usage": {"input_tokens": 20, "output_tokens": 12, "total_tokens": 32,
                     "input_tokens_details": {"cached_tokens": 8}}}


def decode(chunks):
    return [json.loads(line[6:]) for chunk in chunks for line in chunk.splitlines() if line.startswith("data: ")]


@pytest.mark.parametrize("split", range(1, 15))
def test_stop_marker_is_removed_even_when_split_between_chunks(split):
    text = "before</block>after"
    result = decode(anthropic_stream_adapter(events_for([text[:split], text[split:]]),
                    "synthetic", "msg_synthetic", stop_sequences=["</block>"]))
    visible = "".join(event.get("delta", {}).get("text", "") for event in result)
    assert visible == "before"
    delta = next(event for event in result if event["type"] == "message_delta")
    assert delta["delta"]["stop_reason"] == "stop_sequence"
    assert delta["delta"]["stop_sequence"] == "</block>"
    assert delta["usage"]["input_tokens"] == 12
    assert delta["usage"]["cache_read_input_tokens"] == 8
    assert delta["usage"]["output_tokens"] == 12
    assert result[-1]["type"] == "message_stop"


def test_unmatched_partial_marker_and_text_are_preserved():
    result = decode(anthropic_stream_adapter(events_for(["before", "</blo"]),
                    "synthetic", "msg_synthetic", stop_sequences=["</block>"]))
    assert "".join(event.get("delta", {}).get("text", "") for event in result) == "before</blo"
    delta = next(event for event in result if event["type"] == "message_delta")
    assert delta["delta"]["stop_reason"] == "end_turn" and delta["delta"]["stop_sequence"] is None


def test_stop_does_not_hide_an_upstream_failure_or_forge_success():
    def interrupted():
        yield {"type": "content", "text": "before</block>after"}
        raise ChatGPTOAuthProtocolError("synthetic upstream failure")

    chunks = []
    with pytest.raises(ChatGPTOAuthProtocolError, match="synthetic upstream failure"):
        chunks.extend(anthropic_stream_adapter(interrupted(), "synthetic", "msg_synthetic",
                                              stop_sequences=["</block>"]))
    assert "event: message_stop" not in "".join(chunks)


def test_nonstream_stop_uses_earliest_match_and_preserves_authoritative_usage():
    response = AssistantResponse(content="a STOP b END c", finish_reason="stop",
                                 usage=Usage(prompt_tokens=20, completion_tokens=12, total_tokens=32, cached_tokens=8))
    result = internal_response_to_anthropic(response, "synthetic", "msg_synthetic",
                                           stop_sequences=["END", "STOP"])
    assert result["content"][0]["text"] == "a "
    assert result["stop_reason"] == "stop_sequence" and result["stop_sequence"] == "STOP"
    assert result["usage"]["output_tokens"] == 12


@pytest.mark.parametrize("stream", [False, True])
def test_siwc_facade_accepts_stop_without_sending_it_to_upstream(siwc_client, monkeypatch, stream):
    from . import test_siwc_provider
    from .test_siwc_server import _body

    client, _, requests = siwc_client
    document = test_siwc_provider._completed([test_siwc_provider._text_item("before</block>after")])
    import httpx

    def respond(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=test_siwc_provider._catalog())
        return httpx.Response(200, content=test_siwc_provider._sse([document]))

    client_provider = siwc_client[1]
    monkeypatch.setattr(client_provider._http_client, "_transport", httpx.MockTransport(respond))
    response = client.post("/v1/messages", json={**_body(stream=stream), "stop_sequences": ["</block>"]})
    assert response.status_code == 200
    if stream:
        assert '"stop_reason": "stop_sequence"' in response.text
        assert '"text": "before"' in response.text
    else:
        assert response.json()["stop_reason"] == "stop_sequence"
        assert response.json()["content"][0]["text"] == "before"
    payload = json.loads(requests[-1].content)
    assert "stop" not in payload and "stop_sequences" not in payload
