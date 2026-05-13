"""Tests for ``BedrockModelInvoke``."""

import json
import unittest.mock

import pydantic
import pytest
from botocore.exceptions import ClientError

import strands
from strands.models.bedrock import DEFAULT_BEDROCK_MODEL_ID
from strands.models.bedrock_invoke import BedrockModelInvoke
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException

CLAUDE_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"


@pytest.fixture
def session_cls():
    with unittest.mock.patch.object(strands.models.bedrock_invoke.boto3, "Session") as mock_cls:
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
    return BedrockModelInvoke(model_id=CLAUDE_ID)


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


pytestmark = pytest.mark.usefixtures("bedrock_client")


def test_init_default_model_id():
    m = BedrockModelInvoke()
    assert m.get_config()["model_id"] == DEFAULT_BEDROCK_MODEL_ID
    assert m.get_config()["streaming"] is True


def test_init_explicit_model_id():
    m = BedrockModelInvoke(model_id="my-model", streaming=False)
    assert m.get_config()["model_id"] == "my-model"
    assert m.get_config()["streaming"] is False


def test_init_rejects_session_and_region():
    with pytest.raises(ValueError):
        BedrockModelInvoke(boto_session=unittest.mock.Mock(), region_name="us-east-1")


def test_update_config():
    m = BedrockModelInvoke(model_id="m")
    m.update_config(temperature=0.7, max_tokens=128)
    cfg = m.get_config()
    assert cfg["temperature"] == 0.7
    assert cfg["max_tokens"] == 128


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
    assert BedrockModelInvoke(model_id=model_id)._get_model_family() == expected


def test_model_family_override():
    m = BedrockModelInvoke(model_id="arn:aws:bedrock:us-east-1:123:imported-model/abc", model_family="anthropic")
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
    m = BedrockModelInvoke(model_id="meta.llama3-1-8b-instruct-v1:0")
    req = m._format_openai_request(
        [{"role": "user", "content": [{"text": "Hello"}]}], None, [{"text": "sys"}], None
    )
    assert req["model"] == "meta.llama3-1-8b-instruct-v1:0"
    assert req["messages"][0] == {"role": "system", "content": "sys"}
    assert req["messages"][1] == {"role": "user", "content": "Hello"}


def test_format_openai_request_tool_calls_and_results():
    m = BedrockModelInvoke(model_id="my-imported-model", model_family="openai")
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
    events = await _collect(BedrockModelInvoke(model_id=CLAUDE_ID), [{"role": "user", "content": [{"text": "hi"}]}])
    assert _texts(events) == "Hi there"
    assert _stop_reason(events) == "end_turn"
    metadata = next(e for e in events if "metadata" in e)
    assert metadata["metadata"]["usage"] == {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8}


@pytest.mark.asyncio
async def test_stream_anthropic_tool_use(bedrock_client):
    bedrock_client.invoke_model_with_response_stream.return_value = {
        "body": _chunks([
            {"type": "message_start", "message": {"usage": {"input_tokens": 7, "output_tokens": 0}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "tu1", "name": "weather", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '"Paris"}'},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 11}},
            {"type": "message_stop"},
        ])
    }
    events = await _collect(BedrockModelInvoke(model_id=CLAUDE_ID), [{"role": "user", "content": [{"text": "?"}]}])
    starts = [e["contentBlockStart"]["start"] for e in events if "contentBlockStart" in e]
    assert {"toolUse": {"toolUseId": "tu1", "name": "weather"}} in starts
    assert _tool_inputs(events) == '{"city":"Paris"}'
    assert _stop_reason(events) == "tool_use"


@pytest.mark.asyncio
async def test_stream_openai_text_and_tool(bedrock_client):
    bedrock_client.invoke_model_with_response_stream.return_value = {
        "body": _chunks([
            {"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
            {
                "choices": [{
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_abc", "function": {"name": "fn", "arguments": '{"x":'}}
                        ]
                    },
                    "finish_reason": None,
                }]
            },
            {
                "choices": [{
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]},
                    "finish_reason": "tool_calls",
                }]
            },
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}},
        ])
    }
    m = BedrockModelInvoke(model_id="meta.llama3-1-8b-instruct-v1:0")
    events = await _collect(m, [{"role": "user", "content": [{"text": "go"}]}])
    assert _texts(events) == "Hello world"
    assert _tool_inputs(events) == '{"x":1}'
    assert _stop_reason(events) == "tool_use"
    metadata = next(e for e in events if "metadata" in e)
    assert metadata["metadata"]["usage"]["totalTokens"] == 14


@pytest.mark.asyncio
async def test_stream_non_streaming_anthropic(bedrock_client):
    body = unittest.mock.Mock()
    body.read.return_value = json.dumps({
        "content": [{"type": "text", "text": "ack"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode("utf-8")
    bedrock_client.invoke_model.return_value = {"body": body}

    m = BedrockModelInvoke(model_id=CLAUDE_ID, streaming=False)
    events = await _collect(m, [{"role": "user", "content": [{"text": "hi"}]}])
    assert _texts(events) == "ack"


# ---- errors


@pytest.mark.asyncio
async def test_stream_throttling_raises(bedrock_client):
    bedrock_client.invoke_model_with_response_stream.side_effect = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        "InvokeModelWithResponseStream",
    )
    with pytest.raises(ModelThrottledException):
        await _collect(BedrockModelInvoke(model_id=CLAUDE_ID), [{"role": "user", "content": [{"text": "x"}]}])


@pytest.mark.asyncio
async def test_stream_context_window_overflow(bedrock_client):
    bedrock_client.invoke_model_with_response_stream.side_effect = ClientError(
        {"Error": {"Code": "ValidationException", "Message": "Input is too long for requested model"}},
        "InvokeModelWithResponseStream",
    )
    with pytest.raises(ContextWindowOverflowException):
        await _collect(BedrockModelInvoke(model_id=CLAUDE_ID), [{"role": "user", "content": [{"text": "x"}]}])


# ---- structured output


@pytest.mark.asyncio
async def test_structured_output_yields_pydantic_model(bedrock_client):
    bedrock_client.invoke_model_with_response_stream.return_value = {
        "body": _chunks([
            {"type": "message_start", "message": {"usage": {"input_tokens": 4, "output_tokens": 0}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "tu1", "name": "Person", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"name":"Ada","age":36}'},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
            {"type": "message_stop"},
        ])
    }

    class Person(pydantic.BaseModel):
        name: str
        age: int

    m = BedrockModelInvoke(model_id=CLAUDE_ID)
    structured: list[dict] = []
    async for event in m.structured_output(Person, [{"role": "user", "content": [{"text": "?"}]}]):
        structured.append(event)
    assert structured[-1]["output"] == Person(name="Ada", age=36)
