import time
import unittest.mock

import ai21
import pydantic
import pytest

import strands
from strands.models.ai21 import AI21Model
from strands.types.exceptions import ModelThrottledException


@pytest.fixture
def ai21_client():
    with unittest.mock.patch.object(strands.models.ai21, "AsyncAI21Client") as mock_client_cls:
        yield mock_client_cls.return_value


@pytest.fixture
def model_id():
    return "jamba-mini"


@pytest.fixture
def max_tokens():
    return 1


@pytest.fixture
def model(ai21_client, model_id, max_tokens):
    _ = ai21_client
    return AI21Model(client_args={}, model_id=model_id, params={"max_tokens": max_tokens})


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def tool_specs():
    return [
        {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                    },
                    "required": ["input"],
                },
            },
        },
    ]


@pytest.fixture
def system_prompt():
    return "You are a helpful assistant."


@pytest.fixture
def test_output_model_cls():
    class TestOutputModel(pydantic.BaseModel):
        name: str
        age: int

    return TestOutputModel


# ===== Initialization Tests =====


def test__init__model_configs(ai21_client, model_id):
    _ = ai21_client
    model = AI21Model(client_args={}, model_id=model_id, params={"temperature": 0.5})

    tru_temperature = model.get_config().get("params")
    exp_temperature = {"temperature": 0.5}

    assert tru_temperature == exp_temperature


def test__init__with_client_args(ai21_client, model_id):
    _ = ai21_client
    client_args = {"api_key": "test_key", "timeout": 30}
    model = AI21Model(client_args=client_args, model_id=model_id)

    assert model.client_args == client_args
    assert model.get_config().get("model_id") == model_id


def test__init__with_no_params(ai21_client, model_id):
    _ = ai21_client
    model = AI21Model(client_args={}, model_id=model_id)

    assert model.get_config().get("model_id") == model_id
    assert model.get_config().get("params") is None


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


# ===== Message Formatting Tests =====


def test_format_request_messages_with_system_prompt(model, system_prompt):
    messages = [{"role": "user", "content": [{"text": "hello"}]}]

    tru_messages = model._format_request_messages(messages, system_prompt)

    assert len(tru_messages) == 2
    assert tru_messages[0].role == "system"
    assert tru_messages[0].content == system_prompt
    assert tru_messages[1].role == "user"
    assert tru_messages[1].content == "hello"


def test_format_request_messages_without_system_prompt(model):
    messages = [{"role": "user", "content": [{"text": "hello"}]}]

    tru_messages = model._format_request_messages(messages)

    assert len(tru_messages) == 1
    assert tru_messages[0].role == "user"
    assert tru_messages[0].content == "hello"


def test_format_request_messages_with_tool_content(model):
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "c1",
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                    },
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "c1",
                        "status": "success",
                        "content": [{"text": "4"}],
                    }
                }
            ],
        },
    ]

    tru_messages = model._format_request_messages(messages)

    assert len(tru_messages) == 2
    assert tru_messages[0].role == "assistant"
    assert tru_messages[0].content is None  # No text content, only tool calls
    assert len(tru_messages[0].tool_calls) == 1
    tool_call = tru_messages[0].tool_calls[0]
    # AI21 returns dict format, not ToolCall objects
    if hasattr(tool_call, "function"):
        assert tool_call.function.name == "calculator"
    else:
        assert tool_call["function"]["name"] == "calculator"
    assert tru_messages[1].role == "tool"
    assert "4" in tru_messages[1].content
    assert tru_messages[1].tool_call_id == "c1"


def test_format_request_message_content_text(model):
    content = {"text": "hello world"}

    tru_result = model._format_request_message_content(content)
    exp_result = "hello world"

    assert tru_result == exp_result


def test_format_request_message_content_unsupported_type(model):
    content = {"unsupported": {}}

    with pytest.raises(TypeError, match="content_type=<unsupported> | unsupported type"):
        model._format_request_message_content(content)


def test_format_request_tool_spec(model, tool_specs):
    tool_spec = tool_specs[0]

    tru_result = model._format_request_tool_spec(tool_spec)
    exp_result = {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {"type": "string"},
                },
                "required": ["input"],
            },
        },
    }

    assert tru_result == exp_result


def test_format_request_message_tool_call_helper(model):
    """Test the _format_request_message_tool_call helper method directly."""
    tool_use = {"toolUseId": "test123", "name": "calculator", "input": {"expression": "2+2"}}

    tru_result = model._format_request_message_tool_call(tool_use)
    exp_result = {
        "id": "test123",
        "type": "function",
        "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'},
    }

    assert tru_result == exp_result


def test_format_request_message_tool_message_helper(model):
    """Test the _format_request_message_tool_message helper method directly."""
    tool_result = {"toolUseId": "test123", "content": [{"text": "result text"}, {"json": {"key": "value"}}]}

    tru_result = model._format_request_message_tool_message(tool_result)

    assert tru_result["role"] == "tool"
    assert "result text" in tru_result["content"]
    assert "key" in tru_result["content"]
    assert "value" in tru_result["content"]
    assert tru_result["tool_call_id"] == "test123"


# ===== Request Formatting Tests =====


def test_format_request_default(model, messages, model_id):
    tru_request = model.format_request(messages)

    assert tru_request["model"] == model_id
    assert tru_request["stream"] is True
    assert len(tru_request["messages"]) == 1


def test_format_request_with_params(model, messages, model_id):
    model.update_config(params={"temperature": 0.5})

    tru_request = model.format_request(messages)

    assert tru_request["model"] == model_id
    assert tru_request["temperature"] == 0.5


def test_format_request_with_system_prompt(model, messages, model_id, system_prompt):
    tru_request = model.format_request(messages, system_prompt=system_prompt)

    assert tru_request["model"] == model_id
    assert len(tru_request["messages"]) == 2
    assert tru_request["messages"][0].role == "system"
    assert tru_request["messages"][0].content == system_prompt


def test_format_request_with_tools(model, messages, tool_specs, model_id):
    tru_request = model.format_request(messages, tool_specs)

    assert tru_request["model"] == model_id
    assert tru_request["stream"] is True
    assert "tools" in tru_request
    assert len(tru_request["tools"]) == 1
    assert tru_request["tools"][0]["function"]["name"] == "test_tool"


# ===== Chunk Formatting Tests =====


@pytest.mark.parametrize(
    ("event", "exp_chunk"),
    [
        # Message start
        (
            {"chunk_type": "message_start"},
            {"messageStart": {"role": "assistant"}},
        ),
        # Content Start - Text
        (
            {"chunk_type": "content_start", "data_type": "text"},
            {"contentBlockStart": {"start": {}}},
        ),
        # Content Start - Tool Use
        (
            {
                "chunk_type": "content_start",
                "data_type": "tool",
                "data": {"function": {"name": "calculator"}, "id": "c1"},
            },
            {"contentBlockStart": {"start": {"toolUse": {"name": "calculator", "toolUseId": "c1"}}}},
        ),
        # Content Delta - Text
        (
            {"chunk_type": "content_delta", "data_type": "text", "data": "hello"},
            {"contentBlockDelta": {"delta": {"text": "hello"}}},
        ),
        # Content Delta - Tool Use
        (
            {
                "chunk_type": "content_delta",
                "data_type": "tool",
                "data": {"function": {"arguments": '{"expression": "2+2"}'}},
            },
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"expression": "2+2"}'}}}},
        ),
        # Content Stop
        (
            {"chunk_type": "content_stop"},
            {"contentBlockStop": {}},
        ),
        # Message Stop - End Turn
        (
            {"chunk_type": "message_stop", "data": "stop"},
            {"messageStop": {"stopReason": "end_turn"}},
        ),
        # Message Stop - Max Tokens
        (
            {"chunk_type": "message_stop", "data": "length"},
            {"messageStop": {"stopReason": "max_tokens"}},
        ),
        # Message Stop - Tool Use
        (
            {"chunk_type": "message_stop", "data": "tool_calls"},
            {"messageStop": {"stopReason": "tool_use"}},
        ),
        # Metadata
        (
            {
                "chunk_type": "metadata",
                "data": unittest.mock.Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150),
            },
            {
                "metadata": {
                    "usage": {
                        "inputTokens": 100,
                        "outputTokens": 50,
                        "totalTokens": 150,
                    },
                    "metrics": {
                        "latencyMs": 0,
                    },
                },
            },
        ),
    ],
)
def test_format_chunk(event, exp_chunk, model):
    tru_chunk = model.format_chunk(event)
    assert tru_chunk == exp_chunk


def test_format_chunk_unknown_type(model):
    event = {"chunk_type": "unknown"}

    with pytest.raises(RuntimeError, match="chunk_type=<unknown> | unknown type"):
        model.format_chunk(event)


def test_format_chunk_latency_measurement(model):
    start_time = time.time()
    time.sleep(0.01)  # 10ms

    event = {
        "chunk_type": "metadata",
        "data": unittest.mock.Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    }

    tru_chunk = model.format_chunk(event, start_time)

    assert "metadata" in tru_chunk
    assert "metrics" in tru_chunk["metadata"]
    assert "latencyMs" in tru_chunk["metadata"]["metrics"]

    latency = tru_chunk["metadata"]["metrics"]["latencyMs"]
    assert isinstance(latency, int)
    assert latency > 0
    assert latency < 1000


def test_format_chunk_content_delta_tool_empty_arguments(model):
    event = {"chunk_type": "content_delta", "data_type": "tool", "data": {"function": {"arguments": ""}}}

    tru_chunk = model.format_chunk(event)

    # Empty string arguments should return empty delta
    assert tru_chunk == {"contentBlockDelta": {"delta": {}}}


# ===== Streaming Tests =====


@pytest.mark.asyncio
async def test_stream_throttle_error(ai21_client, model, alist):
    ai21_client.chat.completions.create.side_effect = ai21.TooManyRequestsError("rate limit")

    messages = [{"role": "user", "content": [{"text": "hello"}]}]
    with pytest.raises(ModelThrottledException, match="rate limit"):
        await alist(model.stream(messages))


@pytest.mark.asyncio
async def test_stream_api_timeout_error(ai21_client, model):
    ai21_client.chat.completions.create.side_effect = ai21.APITimeoutError("timeout")

    messages = [{"role": "user", "content": [{"text": "hello"}]}]
    with pytest.raises(ai21.APITimeoutError, match="timeout"):
        await anext(model.stream(messages))


@pytest.mark.asyncio
async def test_stream_empty_choices(ai21_client, model, alist):
    # Test handling of responses with empty or missing choices
    mock_event_empty_choices = unittest.mock.Mock(choices=[])  # Empty choices
    mock_event_valid = unittest.mock.Mock(
        choices=[unittest.mock.Mock(delta=unittest.mock.Mock(content="hello", tool_calls=None), finish_reason=None)]
    )
    mock_event_final = unittest.mock.Mock(
        choices=[unittest.mock.Mock(delta=unittest.mock.Mock(content="", tool_calls=None), finish_reason="stop")]
    )
    mock_usage = unittest.mock.Mock(prompt_tokens=5, completion_tokens=3, total_tokens=8)

    async def mock_generator():
        yield mock_event_empty_choices
        yield mock_event_valid
        yield mock_event_final
        yield unittest.mock.Mock(usage=mock_usage)

    ai21_client.chat.completions.create = unittest.mock.AsyncMock(return_value=mock_generator())

    messages = [{"role": "user", "content": [{"text": "test"}]}]
    events = await alist(model.stream(messages))

    # Should handle empty choices gracefully and still produce valid output
    assert len(events) > 0
    assert any("messageStart" in event for event in events)
    assert any("contentBlockDelta" in event for event in events)


# ===== Structured Output Tests =====


@pytest.mark.asyncio
async def test_structured_output(ai21_client, model, test_output_model_cls, alist):
    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    # Mock streaming response for structured output with tool calls
    # Create initial tool call with metadata (name/id)
    mock_tool_call_start = {"function": {"name": "TestOutputModel"}, "id": "123", "index": 0}

    # Create subsequent tool call with arguments
    mock_tool_call_args = {"function": {"arguments": '{"name": "John", "age": 30}'}, "index": 0}

    mock_choice_start = unittest.mock.Mock()
    mock_choice_start.delta = unittest.mock.Mock(content=None)
    mock_choice_start.delta.tool_calls = [mock_tool_call_start]
    mock_choice_start.finish_reason = None

    mock_choice_args = unittest.mock.Mock()
    mock_choice_args.delta = unittest.mock.Mock(content=None)
    mock_choice_args.delta.tool_calls = [mock_tool_call_args]
    mock_choice_args.finish_reason = None

    mock_choice_final = unittest.mock.Mock()
    mock_choice_final.delta = unittest.mock.Mock(content=None)
    mock_choice_final.delta.tool_calls = None
    mock_choice_final.finish_reason = "tool_calls"

    mock_usage = unittest.mock.Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15

    async def mock_stream():
        yield unittest.mock.Mock(choices=[mock_choice_start])
        yield unittest.mock.Mock(choices=[mock_choice_args])
        yield unittest.mock.Mock(choices=[mock_choice_final])
        yield unittest.mock.Mock(usage=mock_usage)

    ai21_client.chat.completions.create = unittest.mock.AsyncMock(return_value=mock_stream())

    stream = model.structured_output(test_output_model_cls, messages)
    events = await alist(stream)

    # The last event should contain the structured output
    final_event = events[-1]
    assert "output" in final_event
    assert isinstance(final_event["output"], test_output_model_cls)
    assert final_event["output"].name == "John"
    assert final_event["output"].age == 30


@pytest.mark.asyncio
async def test_structured_output_no_tool_use_error(ai21_client, model, test_output_model_cls, alist):
    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    # Mock a streaming response that stops without tool use
    mock_choice_text = unittest.mock.Mock()
    mock_choice_text.delta = unittest.mock.Mock()
    mock_choice_text.delta.content = "I can't use tools"
    mock_choice_text.delta.tool_calls = None
    mock_choice_text.finish_reason = None

    mock_choice_final = unittest.mock.Mock()
    mock_choice_final.delta = unittest.mock.Mock()
    mock_choice_final.delta.content = None
    mock_choice_final.delta.tool_calls = None
    mock_choice_final.finish_reason = "stop"  # Stops without tool use

    mock_usage = unittest.mock.Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    mock_usage.total_tokens = 15

    async def mock_stream():
        yield unittest.mock.Mock(choices=[mock_choice_text])
        yield unittest.mock.Mock(choices=[mock_choice_final])
        yield unittest.mock.Mock(usage=mock_usage)

    ai21_client.chat.completions.create = unittest.mock.AsyncMock(return_value=mock_stream())

    stream = model.structured_output(test_output_model_cls, messages)

    with pytest.raises(ValueError, match="AI21 did not use tools"):
        await alist(stream)


# ===== Utility Functions for Tests =====


@pytest.fixture
def alist():
    """Convert async generator to list."""

    async def _alist(async_gen):
        result = []
        async for item in async_gen:
            result.append(item)
        return result

    return _alist
