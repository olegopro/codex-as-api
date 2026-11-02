"""Updates-only display hides reasoning without changing encrypted replay."""

from __future__ import annotations

import json

import pytest

from codex_as_api.anthropic_adapter import (
    anthropic_request_to_internal,
    anthropic_stream_adapter,
    internal_response_to_anthropic,
    resolve_anthropic_thinking_options,
)
from codex_as_api.auth import ChatGPTOAuthProtocolError
from codex_as_api.messages import AssistantResponse, Usage
from codex_as_api.reasoning_signature import decode_reasoning_signature, encode_reasoning_signature
from codex_as_api.siwc_transport import _output_key

MODEL = "gpt-6.1-terra"
ITEM = {"type": "reasoning", "id": "rs_updates", "encrypted_content": "encrypted", "summary": []}


@pytest.mark.parametrize("mode", ["enabled", "adaptive"])
def test_updates_request_keeps_effort_and_disables_summary(mode):
    thinking = {"type": mode, "display": "updates"}
    if mode == "enabled":
        thinking["budget_tokens"] = 1024
    converted = anthropic_request_to_internal(
        model=MODEL, messages=[{"role": "user", "content": "test"}],
        max_tokens=32000, thinking=thinking, output_config={"effort": "high"},
    )
    assert converted[4] == "high"
    assert resolve_anthropic_thinking_options(thinking, 32000) == ("updates", "none", False)


def _response(item=ITEM, **kwargs):
    return AssistantResponse(
        content="Answer", tool_calls=(), finish_reason="stop",
        usage=Usage(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        reasoning_items=({"model": MODEL, "item": item},), **kwargs,
    )


def test_updates_nonstream_keeps_signature_and_affinity():
    result = internal_response_to_anthropic(_response(), MODEL, "msg_updates", thinking_display="updates")
    block = result["content"][0]
    assert block["type"] == "thinking" and block["thinking"] == ""
    assert "codex_reasoning" not in result
    assert decode_reasoning_signature(block["signature"])["item"] == ITEM
    replay, *_ = anthropic_request_to_internal(
        model=MODEL, messages=[{"role": "assistant", "content": result["content"]}],
    )
    assert replay[0].reasoning_items[0]["item"] == ITEM
    assert _output_key([ITEM]) == _output_key([replay[0].reasoning_items[0]["item"]])
    assert result["content"][1]["text"] == "Answer"


def test_updates_stream_keeps_encrypted_replay_and_text():
    events = [
        {"type": "reasoning_item", "model": MODEL, "item": ITEM},
        {"type": "content", "text": "Answer"},
        {"type": "finish", "finish_reason": "stop",
         "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}},
    ]
    rendered = "".join(anthropic_stream_adapter(iter(events), MODEL, "msg_updates", thinking_display="updates"))
    parsed = [json.loads(line[6:]) for line in rendered.splitlines() if line.startswith("data: ")]
    signatures = [e["delta"]["signature"] for e in parsed if e.get("delta", {}).get("type") == "signature_delta"]
    assert len(signatures) == 1
    assert decode_reasoning_signature(signatures[0])["item"] == ITEM
    assert "thinking_delta" not in rendered and "codex_reasoning_delta" not in rendered
    assert parsed[-1]["type"] == "message_stop"
    assert "Answer" in rendered


@pytest.mark.parametrize("field", ["summary", "content"])
def test_updates_rejects_unexpected_reasoning_without_encoding_it(field):
    item = {**ITEM, field: [{"type": "summary_text" if field == "summary" else "reasoning_text",
                            "text": "PRIVATE_CANARY"}]}
    with pytest.raises(ChatGPTOAuthProtocolError, match="updates-only"):
        internal_response_to_anthropic(_response(item), MODEL, "msg_updates", thinking_display="updates")
    output = []
    with pytest.raises(ChatGPTOAuthProtocolError, match="updates-only"):
        for chunk in anthropic_stream_adapter(
            iter([{"type": "reasoning_item", "model": MODEL, "item": item}]),
            MODEL, "msg_updates", thinking_display="updates",
        ):
            output.append(chunk)
    assert "PRIVATE_CANARY" not in "".join(output)
    assert "signature_delta" not in "".join(output)


@pytest.mark.parametrize("kind", ["reasoning_delta", "reasoning_raw_delta"])
def test_updates_rejects_unexpected_delta_without_exposure(kind):
    output = []
    with pytest.raises(ChatGPTOAuthProtocolError, match="updates-only"):
        for chunk in anthropic_stream_adapter(
            iter([{"type": kind, "text": "PRIVATE_CANARY", "item_id": ITEM["id"]}]),
            MODEL, "msg_updates", thinking_display="updates",
        ):
            output.append(chunk)
    assert "PRIVATE_CANARY" not in "".join(output)


def test_updates_rejects_nonitem_reasoning_content():
    with pytest.raises(ChatGPTOAuthProtocolError, match="updates-only"):
        internal_response_to_anthropic(
            _response(reasoning_content="PRIVATE_CANARY"), MODEL, "msg_updates", thinking_display="updates",
        )


def test_old_signatures_with_summary_remain_readable():
    item = {**ITEM, "summary": [{"type": "summary_text", "text": "Existing summary"}]}
    signature = encode_reasoning_signature(MODEL, item)
    assert decode_reasoning_signature(signature)["item"] == item
