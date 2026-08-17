"""Tests for ``BedrockInvokeModel``."""

import asyncio
import json
import sys
import time
import traceback
import unittest.mock

import pydantic
import pytest
from botocore.exceptions import ClientError

import strands
from strands import _exception_notes
from strands.models.bedrock import DEFAULT_BEDROCK_MODEL_ID
from strands.models.bedrock_invoke import BedrockInvokeModel
from strands.models.model import Model
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException

CLAUDE_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"


@pytest.fixture
def session_cls():
    with unittest.mock.patch.object(strands.models.bedrock.boto3, "Session") as mock_cls:
        sess = unittest.mock.Mock()
        sess.region_name = None
        mock_cls.return_value = sess
        yield mock_cls


@pytest.fixture
def bedrock_client(session_cls):
    client = session_cls.return_value.client.return_value
    client.meta = unittest.mock.MagicMock()
    client.meta.region_name = "us-west-2"
    return client


@pytest.fixture
def model(bedrock_client):
    _ = bedrock_client
    return BedrockInvokeModel(model_id=CLAUDE_ID)


def _chunks(payloads):
    return [{"chunk": {"bytes": json.dumps(p).encode("utf-8")}} for p in payloads]


async def _collect(m, *args, **kwargs):
    return [e async for e in m.stream(*args, **kwargs)]


def _texts(events):
    return "".join(
        e["contentBlockDelta"]["delta"]["text"]
        for e in events
        if "contentBlockDelta" in e and "text" in e["contentBlockDelta"]["delta"]
    )


def _tool_inputs(events):
    return "".join(
        e["contentBlockDelta"]["delta"]["toolUse"]["input"]
        for e in events
        if "contentBlockDelta" in e and "toolUse" in e["contentBlockDelta"]["delta"]
    )


def _stop_reason(events):
    return next(e for e in events if "messageStop" in e)["messageStop"]["stopReason"]


def _metadata(events):
    return next(e for e in events if "metadata" in e)["metadata"]


def _tool_use_blocks(events):
    """Group tool-use content blocks by their delimiting start/stop events, as the consumer sees them.

    Returns a list of ``(start, joined_input)`` pairs, one per tool call, so a test can assert that
    parallel tool calls stay separate blocks instead of being merged or overwritten.
    """
    blocks = []
    current = None
    for event in events:
        if "contentBlockStart" in event:
            start = event["contentBlockStart"]["start"].get("toolUse")
            current = (start, []) if start else None
        elif "contentBlockStop" in event and current is not None:
            blocks.append((current[0], "".join(current[1])))
            current = None
        elif "contentBlockDelta" in event and current is not None:
            delta = event["contentBlockDelta"]["delta"]
            if "toolUse" in delta:
                current[1].append(delta["toolUse"]["input"])
    return blocks


pytestmark = pytest.mark.usefixtures("bedrock_client")


def test_init_default_model_id():
    m = BedrockInvokeModel()
    assert m.get_config()["model_id"] == DEFAULT_BEDROCK_MODEL_ID
    assert m.get_config()["streaming"] is True


def test_init_explicit_model_id():
    m = BedrockInvokeModel(model_id="my-model", streaming=False)
    assert m.get_config()["model_id"] == "my-model"
    assert m.get_config()["streaming"] is False


def test_init_rejects_session_and_region():
    with pytest.raises(ValueError):
        BedrockInvokeModel(boto_session=unittest.mock.Mock(), region_name="us-east-1")


def test_update_config():
    m = BedrockInvokeModel(model_id="m")
    m.update_config(temperature=0.7, max_tokens=128)
    cfg = m.get_config()
    assert cfg["temperature"] == 0.7
    assert cfg["max_tokens"] == 128


def test_get_config_resolves_context_window_limit_for_known_model():
    """A known model id resolves its context window limit so ConversationManager can compress proactively."""
    m = BedrockInvokeModel(model_id=CLAUDE_ID)
    assert m.get_config()["context_window_limit"] == 200_000
    assert m.context_window_limit == 200_000


def test_get_config_keeps_explicit_context_window_limit():
    m = BedrockInvokeModel(model_id=CLAUDE_ID, context_window_limit=42)
    assert m.get_config()["context_window_limit"] == 42
    assert m.context_window_limit == 42


def test_get_config_context_window_limit_none_for_unknown_model():
    m = BedrockInvokeModel(model_id="arn:aws:bedrock:us-east-1:123:imported-model/abc")
    assert m.get_config().get("context_window_limit") is None
    assert m.context_window_limit is None


# ---- token counting


@pytest.mark.asyncio
async def test_count_tokens_uses_heuristic(bedrock_client):
    """Native CountTokens is Converse-shaped, so this provider always estimates locally."""
    messages = [{"role": "user", "content": [{"text": "count these tokens please"}]}]

    m = BedrockInvokeModel(model_id=CLAUDE_ID)
    tru_count = await m.count_tokens(messages)
    exp_count = await Model.count_tokens(m, messages)

    assert tru_count == exp_count
    bedrock_client.count_tokens.assert_not_called()


@pytest.mark.asyncio
async def test_count_tokens_ignores_use_native_token_count(bedrock_client):
    messages = [{"role": "user", "content": [{"text": "count these tokens please"}]}]

    m = BedrockInvokeModel(model_id=CLAUDE_ID)
    m.config["use_native_token_count"] = True  # type: ignore[typeddict-unknown-key]

    tru_count = await m.count_tokens(messages)
    exp_count = await Model.count_tokens(m, messages)

    assert tru_count == exp_count
    bedrock_client.count_tokens.assert_not_called()


@pytest.mark.parametrize(
    "model_id, expected",
    [
        (CLAUDE_ID, "anthropic"),
        ("global.anthropic.claude-sonnet-4-6", "anthropic"),
        ("us.anthropic.claude-3-haiku", "anthropic"),
        ("arn:aws:bedrock:us-east-1:123:imported-model/abc", "openai"),
        ("meta.llama3-1-8b-instruct-v1:0", "openai"),
        ("mistral.mistral-large-2402-v1:0", "openai"),
    ],
)
def test_model_family_detection(model_id, expected):
    assert BedrockInvokeModel(model_id=model_id)._get_model_family() == expected


def test_model_family_override():
    m = BedrockInvokeModel(model_id="arn:aws:bedrock:us-east-1:123:imported-model/abc", model_family="anthropic")
    assert m._get_model_family() == "anthropic"


# ---- request formatting


def test_format_anthropic_request_minimal(model):
    req = model._format_anthropic_request(
        [{"role": "user", "content": [{"text": "hello"}]}], None, [{"text": "be nice"}], None
    )
    assert req["anthropic_version"] == "bedrock-2023-05-31"
    assert req["system"] == "be nice"
    assert req["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]


def test_format_anthropic_request_image_media_type(model):
    msg = {"role": "user", "content": [{"image": {"format": "png", "source": {"bytes": b"\x89PNG\r\n"}}}]}
    req = model._format_anthropic_request([msg], None, None, None)
    image = req["messages"][0]["content"][0]
    assert image["type"] == "image"
    assert image["source"]["media_type"] == "image/png"


def test_format_anthropic_request_tool_use_and_result(model):
    tu = {"toolUseId": "tu1", "name": "weather", "input": {"city": "Paris"}}
    tr = {"toolUseId": "tu1", "status": "error", "content": [{"text": "boom"}]}
    msgs = [
        {"role": "assistant", "content": [{"toolUse": tu}]},
        {"role": "user", "content": [{"toolResult": tr}]},
    ]
    req = model._format_anthropic_request(msgs, None, None, None)
    expected = {"type": "tool_use", "id": "tu1", "name": "weather", "input": tu["input"]}
    assert req["messages"][0]["content"][0] == expected
    user = req["messages"][1]["content"][0]
    assert user["type"] == "tool_result"
    assert user["tool_use_id"] == "tu1"
    assert user["is_error"] is True
    assert user["content"] == [{"type": "text", "text": "boom"}]


def test_format_anthropic_request_tool_choice(model):
    req = model._format_anthropic_request(
        [{"role": "user", "content": [{"text": "x"}]}],
        [{"name": "t", "description": "d", "inputSchema": {"type": "object"}}],
        None,
        {"any": {}},
    )
    assert req["tool_choice"] == {"type": "any"}
    assert req["tools"][0]["name"] == "t"
    assert req["tools"][0]["input_schema"] == {"type": "object"}


def test_format_openai_request_basic():
    m = BedrockInvokeModel(model_id="meta.llama3-1-8b-instruct-v1:0")
    req = m._format_openai_request(
        [{"role": "user", "content": [{"text": "Hello"}]}], None, [{"text": "sys"}], None
    )
    assert req["model"] == "meta.llama3-1-8b-instruct-v1:0"
    assert req["messages"][0] == {"role": "system", "content": "sys"}
    assert req["messages"][1] == {"role": "user", "content": "Hello"}


def test_format_openai_request_tool_calls_and_results():
    m = BedrockInvokeModel(model_id="my-imported-model", model_family="openai")
    tu = {"toolUseId": "tu1", "name": "fn", "input": {"x": 1}}
    tr = {"toolUseId": "tu1", "status": "success", "content": [{"text": "ok"}]}
    spec = [{"name": "fn", "description": "d", "inputSchema": {"type": "object"}}]
    msgs = [
        {"role": "assistant", "content": [{"toolUse": tu}]},
        {"role": "user", "content": [{"toolResult": tr}]},
    ]
    req = m._format_openai_request(msgs, spec, None, {"tool": {"name": "fn"}})
    fn = req["messages"][0]["tool_calls"][0]["function"]
    assert fn == {"name": "fn", "arguments": json.dumps({"x": 1})}
    assert req["messages"][1] == {"role": "tool", "tool_call_id": "tu1", "content": "ok"}
    assert req["tool_choice"] == {"type": "function", "function": {"name": "fn"}}


# ---- params passthrough


def test_format_anthropic_request_merges_params(model):
    """``params`` carries Anthropic-only wire fields the typed config does not model."""
    model.update_config(params={"thinking": {"type": "enabled", "budget_tokens": 1024}, "anthropic_beta": ["beta-1"]})
    req = model._format_anthropic_request([{"role": "user", "content": [{"text": "hi"}]}], None, None, None)
    assert req["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert req["anthropic_beta"] == ["beta-1"]


def test_format_openai_request_merges_params():
    m = BedrockInvokeModel(model_id="meta.llama3-1-8b-instruct-v1:0", params={"logprobs": True})
    req = m._format_openai_request([{"role": "user", "content": [{"text": "hi"}]}], None, None, None)
    assert req["logprobs"] is True


def test_format_request_params_override_computed_fields(model):
    """``params`` is splatted last, matching anthropic.py, so it wins over computed fields."""
    model.update_config(max_tokens=100, params={"max_tokens": 999})
    req = model._format_anthropic_request([{"role": "user", "content": [{"text": "hi"}]}], None, None, None)
    assert req["max_tokens"] == 999


# ---- unsupported content blocks


@pytest.mark.parametrize(
    "block",
    [
        {"document": {"format": "pdf", "name": "doc", "source": {"bytes": b"%PDF-"}}},
        {"video": {"format": "mp4", "source": {"bytes": b"\x00"}}},
        {"cachePoint": {"type": "default"}},
        {"reasoningContent": {"reasoningText": {"text": "hmm", "signature": "sig"}}},
    ],
)
def test_format_anthropic_request_rejects_unsupported_block(model, block):
    """An unformattable block raises rather than silently vanishing from the request."""
    with pytest.raises(TypeError, match="unsupported type"):
        model._format_anthropic_request([{"role": "user", "content": [{"text": "hi"}, block]}], None, None, None)


@pytest.mark.parametrize(
    "block",
    [
        {"image": {"format": "png", "source": {"bytes": b"\x89PNG\r\n"}}},
        {"document": {"format": "pdf", "name": "doc", "source": {"bytes": b"%PDF-"}}},
        {"cachePoint": {"type": "default"}},
    ],
)
def test_format_openai_request_rejects_unsupported_block(block):
    m = BedrockInvokeModel(model_id="meta.llama3-1-8b-instruct-v1:0")
    with pytest.raises(TypeError, match="unsupported type"):
        m._format_openai_request([{"role": "user", "content": [{"text": "hi"}, block]}], None, None, None)


def test_format_anthropic_request_all_unsupported_blocks_does_not_drop_message(model):
    """A message of only unsupported blocks raises instead of silently dropping the whole message."""
    msgs = [
        {"role": "user", "content": [{"text": "hi"}]},
        {"role": "assistant", "content": [{"reasoningContent": {"reasoningText": {"text": "x", "signature": "s"}}}]},
    ]
    with pytest.raises(TypeError, match="unsupported type"):
        model._format_anthropic_request(msgs, None, None, None)


def test_format_openai_request_all_unsupported_blocks_does_not_drop_message():
    m = BedrockInvokeModel(model_id="meta.llama3-1-8b-instruct-v1:0")
    msgs = [
        {"role": "user", "content": [{"text": "hi"}]},
        {"role": "user", "content": [{"image": {"format": "png", "source": {"bytes": b"\x89PNG\r\n"}}}]},
    ]
    with pytest.raises(TypeError, match="unsupported type"):
        m._format_openai_request(msgs, None, None, None)


# ---- unsupported inherited methods


def test_format_request_not_supported(model):
    """Converse-shaped request formatting is not what this provider sends."""
    with pytest.raises(NotImplementedError, match="format_request"):
        model.format_request([{"role": "user", "content": [{"text": "hi"}]}])


def test_convert_non_streaming_to_streaming_not_supported(model):
    """Converse-shaped response translation is not what this provider receives."""
    with pytest.raises(NotImplementedError, match="convert_non_streaming_to_streaming"):
        list(model.convert_non_streaming_to_streaming({"content": [{"type": "text", "text": "hi"}]}))


# ---- streaming


@pytest.mark.asyncio
async def test_stream_anthropic_text_only(bedrock_client):
    bedrock_client.invoke_model_with_response_stream.return_value = {
        "body": _chunks([
            {"type": "message_start", "message": {"usage": {"input_tokens": 5, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " there"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
            {"type": "message_stop"},
        ])
    }
    events = await _collect(BedrockInvokeModel(model_id=CLAUDE_ID), [{"role": "user", "content": [{"text": "hi"}]}])
    assert _texts(events) == "Hi there"
    assert _stop_reason(events) == "end_turn"
    assert _metadata(events)["usage"] == {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8}


@pytest.mark.asyncio
async def test_stream_anthropic_tool_use(bedrock_client):
    cb_start = {"type": "tool_use", "id": "tu1", "name": "weather", "input": {}}
    delta1 = {"type": "input_json_delta", "partial_json": '{"city":'}
    delta2 = {"type": "input_json_delta", "partial_json": '"Paris"}'}
    bedrock_client.invoke_model_with_response_stream.return_value = {
        "body": _chunks([
            {"type": "message_start", "message": {"usage": {"input_tokens": 7, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 0, "content_block": cb_start},
            {"type": "content_block_delta", "index": 0, "delta": delta1},
            {"type": "content_block_delta", "index": 0, "delta": delta2},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 11}},
            {"type": "message_stop"},
        ])
    }
    events = await _collect(BedrockInvokeModel(model_id=CLAUDE_ID), [{"role": "user", "content": [{"text": "?"}]}])
    starts = [e["contentBlockStart"]["start"] for e in events if "contentBlockStart" in e]
    assert {"toolUse": {"toolUseId": "tu1", "name": "weather"}} in starts
    assert _tool_inputs(events) == '{"city":"Paris"}'
    assert _stop_reason(events) == "tool_use"


@pytest.mark.asyncio
async def test_stream_openai_text_and_tool(bedrock_client):
    tc1 = {"index": 0, "id": "call_abc", "function": {"name": "fn", "arguments": '{"x":'}}
    tc2 = {"index": 0, "function": {"arguments": "1}"}}
    bedrock_client.invoke_model_with_response_stream.return_value = {
        "body": _chunks([
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [tc1]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [tc2]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}},
        ])
    }
    m = BedrockInvokeModel(model_id="meta.llama3-1-8b-instruct-v1:0")
    events = await _collect(m, [{"role": "user", "content": [{"text": "go"}]}])
    assert _texts(events) == "Hello world"
    assert _tool_inputs(events) == '{"x":1}'
    assert _stop_reason(events) == "tool_use"
    assert _metadata(events)["usage"]["totalTokens"] == 14


@pytest.mark.asyncio
async def test_stream_openai_parallel_tool_calls_stay_separate_blocks(bedrock_client):
    """Each parallel tool call gets its own start/stop pair, since the consumer keeps only the newest open block."""
    tc0_start = {"index": 0, "id": "call_0", "function": {"name": "weather", "arguments": '{"city":'}}
    tc0_args = {"index": 0, "function": {"arguments": '"Paris"}'}}
    tc1_start = {"index": 1, "id": "call_1", "function": {"name": "time", "arguments": '{"tz":'}}
    tc1_args = {"index": 1, "function": {"arguments": '"UTC"}'}}
    bedrock_client.invoke_model_with_response_stream.return_value = {
        "body": _chunks([
            {"choices": [{"delta": {"tool_calls": [tc0_start]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [tc0_args]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [tc1_start]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [tc1_args]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 8, "total_tokens": 17}},
        ])
    }
    m = BedrockInvokeModel(model_id="meta.llama3-1-8b-instruct-v1:0")
    events = await _collect(m, [{"role": "user", "content": [{"text": "go"}]}])

    tru_blocks = _tool_use_blocks(events)
    exp_blocks = [
        ({"toolUseId": "call_0", "name": "weather"}, '{"city":"Paris"}'),
        ({"toolUseId": "call_1", "name": "time"}, '{"tz":"UTC"}'),
    ]
    assert tru_blocks == exp_blocks
    assert _stop_reason(events) == "tool_use"


@pytest.mark.asyncio
async def test_stream_openai_content_blocks_are_delimited(bedrock_client):
    """Every block is closed before the next opens, in both the text->tool and tool->text directions."""
    tc0 = {"index": 0, "id": "call_0", "function": {"name": "a", "arguments": "{}"}}
    tc1 = {"index": 1, "id": "call_1", "function": {"name": "b", "arguments": "{}"}}
    bedrock_client.invoke_model_with_response_stream.return_value = {
        "body": _chunks([
            {"choices": [{"delta": {"content": "thinking"}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [tc0]}, "finish_reason": None}]},
            {"choices": [{"delta": {"tool_calls": [tc1]}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " done"}, "finish_reason": "tool_calls"}]},
        ])
    }
    m = BedrockInvokeModel(model_id="meta.llama3-1-8b-instruct-v1:0")
    events = await _collect(m, [{"role": "user", "content": [{"text": "go"}]}])

    open_blocks = 0
    for event in events:
        if "contentBlockStart" in event:
            open_blocks += 1
        elif "contentBlockStop" in event:
            open_blocks -= 1
        assert 0 <= open_blocks <= 1
    assert open_blocks == 0
    assert _texts(events) == "thinking done"
    assert [start["name"] for start, _ in _tool_use_blocks(events)] == ["a", "b"]


@pytest.mark.asyncio
async def test_stream_non_streaming_anthropic(bedrock_client):
    body = unittest.mock.Mock()
    body.read.return_value = json.dumps({
        "content": [{"type": "text", "text": "ack"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode("utf-8")
    bedrock_client.invoke_model.return_value = {"body": body}

    m = BedrockInvokeModel(model_id=CLAUDE_ID, streaming=False)
    events = await _collect(m, [{"role": "user", "content": [{"text": "hi"}]}])
    assert _texts(events) == "ack"


@pytest.mark.asyncio
async def test_stream_non_streaming_openai_text_and_tool(bedrock_client):
    body = unittest.mock.Mock()
    body.read.return_value = json.dumps({
        "choices": [{
            "message": {
                "content": "hi there",
                "tool_calls": [{"id": "call_1", "function": {"name": "fn", "arguments": '{"x":1}'}}],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
    }).encode("utf-8")
    bedrock_client.invoke_model.return_value = {"body": body}

    m = BedrockInvokeModel(model_id="meta.llama3-1-8b-instruct-v1:0", streaming=False)
    events = await _collect(m, [{"role": "user", "content": [{"text": "hi"}]}])
    assert _texts(events) == "hi there"
    assert _tool_inputs(events) == '{"x":1}'
    assert _stop_reason(events) == "tool_use"


# ---- latency metrics


@pytest.mark.asyncio
async def test_stream_reports_measured_latency(bedrock_client):
    """Latency is measured around the boto3 call so telemetry sees a real number."""

    def slow_invoke(**kwargs):
        time.sleep(0.02)
        return {
            "body": _chunks([
                {"type": "message_start", "message": {"usage": {"input_tokens": 5, "output_tokens": 0}}},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}},
                {"type": "message_stop"},
            ])
        }

    bedrock_client.invoke_model_with_response_stream.side_effect = slow_invoke

    events = await _collect(BedrockInvokeModel(model_id=CLAUDE_ID), [{"role": "user", "content": [{"text": "hi"}]}])
    assert _metadata(events)["metrics"]["latencyMs"] > 0


@pytest.mark.asyncio
async def test_stream_openai_reports_measured_latency(bedrock_client):
    def slow_invoke(**kwargs):
        time.sleep(0.02)
        return {
            "body": _chunks([
                {"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
            ])
        }

    bedrock_client.invoke_model_with_response_stream.side_effect = slow_invoke

    m = BedrockInvokeModel(model_id="meta.llama3-1-8b-instruct-v1:0")
    events = await _collect(m, [{"role": "user", "content": [{"text": "hi"}]}])
    assert _metadata(events)["metrics"]["latencyMs"] > 0


@pytest.mark.asyncio
async def test_stream_non_streaming_anthropic_reports_measured_latency(bedrock_client):
    body = unittest.mock.Mock()
    body.read.return_value = json.dumps({
        "content": [{"type": "text", "text": "ack"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode("utf-8")

    def slow_invoke(**kwargs):
        time.sleep(0.02)
        return {"body": body}

    bedrock_client.invoke_model.side_effect = slow_invoke

    m = BedrockInvokeModel(model_id=CLAUDE_ID, streaming=False)
    events = await _collect(m, [{"role": "user", "content": [{"text": "hi"}]}])
    assert _metadata(events)["metrics"]["latencyMs"] > 0


@pytest.mark.asyncio
async def test_stream_non_streaming_openai_reports_measured_latency(bedrock_client):
    body = unittest.mock.Mock()
    body.read.return_value = json.dumps({
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8},
    }).encode("utf-8")

    def slow_invoke(**kwargs):
        time.sleep(0.02)
        return {"body": body}

    bedrock_client.invoke_model.side_effect = slow_invoke

    m = BedrockInvokeModel(model_id="meta.llama3-1-8b-instruct-v1:0", streaming=False)
    events = await _collect(m, [{"role": "user", "content": [{"text": "hi"}]}])
    assert _metadata(events)["metrics"]["latencyMs"] > 0


# ---- errors


@pytest.mark.asyncio
async def test_stream_throttling_raises(bedrock_client):
    bedrock_client.invoke_model_with_response_stream.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        "InvokeModelWithResponseStream",
    )
    with pytest.raises(ModelThrottledException):
        await _collect(BedrockInvokeModel(model_id=CLAUDE_ID), [{"role": "user", "content": [{"text": "x"}]}])


@pytest.mark.asyncio
async def test_stream_context_window_overflow(bedrock_client):
    bedrock_client.invoke_model_with_response_stream.side_effect = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Input is too long for requested model"}},
        "InvokeModelWithResponseStream",
    )
    with pytest.raises(ContextWindowOverflowException):
        await _collect(BedrockInvokeModel(model_id=CLAUDE_ID), [{"role": "user", "content": [{"text": "x"}]}])


@pytest.mark.skipif(sys.version_info < (3, 11), reason="This test requires Python 3.11 or higher (need add_note)")
@pytest.mark.asyncio
async def test_stream_access_denied_adds_note(bedrock_client):
    bedrock_client.invoke_model_with_response_stream.side_effect = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "You don't have access to the model"}},
        "InvokeModelWithResponseStream",
    )
    with pytest.raises(ClientError) as err:
        await _collect(BedrockInvokeModel(model_id=CLAUDE_ID), [{"role": "user", "content": [{"text": "x"}]}])
    notes = getattr(err.value, "__notes__", [])
    assert any("required-iam-permissions" in note for note in notes)
    assert any(f"Model id: {CLAUDE_ID}" in note for note in notes)


@pytest.mark.asyncio
async def test_stream_access_denied_adds_note_without_add_notes(bedrock_client):
    """When add_note is not available, the note text is still included in the error output."""
    with unittest.mock.patch.object(_exception_notes, "supports_add_note", False):
        bedrock_client.invoke_model_with_response_stream.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "You don't have access to the model"}},
            "InvokeModelWithResponseStream",
        )
        with pytest.raises(ClientError) as err:
            await _collect(BedrockInvokeModel(model_id=CLAUDE_ID), [{"role": "user", "content": [{"text": "x"}]}])

    error_str = "".join(traceback.format_exception(err.value))
    assert "required-iam-permissions" in error_str
    assert f"└ Model id: {CLAUDE_ID}" in error_str


# ---- cancellation


def _slow_then_raise(bedrock_client):
    """Make the streaming InvokeModel call block, then fail, mimicking a hung boto3 call."""

    def slow_invoke(**kwargs):
        time.sleep(0.1)
        raise RuntimeError("simulated boto3 timeout")

    bedrock_client.invoke_model_with_response_stream.side_effect = slow_invoke


@pytest.mark.asyncio
async def test_stream_cancellation_consumes_orphaned_task_exception(bedrock_client):
    """Orphaned background task exception is consumed when stream generator is cancelled."""
    _slow_then_raise(bedrock_client)

    loop = asyncio.get_running_loop()
    captured: list[dict] = []
    loop.set_exception_handler(lambda _loop, ctx: captured.append(ctx))

    gen = BedrockInvokeModel(model_id=CLAUDE_ID).stream([{"role": "user", "content": [{"text": "x"}]}])
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(gen.__anext__(), timeout=0.01)

    await gen.aclose()

    # Allow the background thread to finish and the done-callback to fire
    await asyncio.sleep(0.2)

    assert not captured, f"orphaned task exception was not consumed: {captured}"


@pytest.mark.asyncio
async def test_stream_cancellation_does_not_block_on_background_call(bedrock_client):
    """Cancelling the generator returns promptly instead of waiting for the blocking boto3 call."""
    _slow_then_raise(bedrock_client)

    gen = BedrockInvokeModel(model_id=CLAUDE_ID).stream([{"role": "user", "content": [{"text": "x"}]}])
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(gen.__anext__(), timeout=0.01)

    # The background thread still sleeps for ~0.1s; closing must not wait for it.
    await asyncio.wait_for(gen.aclose(), timeout=0.05)

    # Consume the orphaned task's exception so it doesn't leak into other tests.
    await asyncio.sleep(0.2)


# ---- structured output


@pytest.mark.asyncio
async def test_structured_output_yields_pydantic_model(bedrock_client):
    cb_start = {"type": "tool_use", "id": "tu1", "name": "Person", "input": {}}
    delta = {"type": "input_json_delta", "partial_json": '{"name":"Ada","age":36}'}
    bedrock_client.invoke_model_with_response_stream.return_value = {
        "body": _chunks([
            {"type": "message_start", "message": {"usage": {"input_tokens": 4, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 0, "content_block": cb_start},
            {"type": "content_block_delta", "index": 0, "delta": delta},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
            {"type": "message_stop"},
        ])
    }

    class Person(pydantic.BaseModel):
        name: str
        age: int

    m = BedrockInvokeModel(model_id=CLAUDE_ID)
    structured: list[dict] = []
    async for event in m.structured_output(Person, [{"role": "user", "content": [{"text": "?"}]}]):
        structured.append(event)
    assert structured[-1]["output"] == Person(name="Ada", age=36)
