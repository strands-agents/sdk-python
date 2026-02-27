import unittest.mock

import openai
import pydantic
import pytest

import strands
from strands.models.openai_responses import _MAX_MEDIA_SIZE_BYTES, OpenAIResponsesModel
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException


@pytest.fixture
def openai_client():
    with unittest.mock.patch.object(strands.models.openai_responses.openai, "AsyncOpenAI") as mock_client_cls:
        mock_client = unittest.mock.AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        yield mock_client


@pytest.fixture
def model_id():
    return "gpt-4o"


@pytest.fixture
def model(openai_client, model_id):
    _ = openai_client
    return OpenAIResponsesModel(model_id=model_id, params={"max_output_tokens": 100})


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
    return "s1"


@pytest.fixture
def test_output_model_cls():
    class TestOutputModel(pydantic.BaseModel):
        name: str
        age: int

    return TestOutputModel


def test__init__(model_id):
    model = OpenAIResponsesModel(model_id=model_id, params={"max_output_tokens": 100})

    tru_config = model.get_config()
    exp_config = {"model_id": "gpt-4o", "params": {"max_output_tokens": 100}}

    assert tru_config == exp_config


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


@pytest.mark.parametrize(
    "content, exp_result",
    [
        # Document
        (
            {
                "document": {
                    "format": "pdf",
                    "name": "test doc",
                    "source": {"bytes": b"document"},
                },
            },
            {
                "type": "input_file",
                "file_url": "data:application/pdf;base64,ZG9jdW1lbnQ=",
            },
        ),
        # Image
        (
            {
                "image": {
                    "format": "jpg",
                    "source": {"bytes": b"image"},
                },
            },
            {
                "type": "input_image",
                "image_url": "data:image/jpeg;base64,aW1hZ2U=",
            },
        ),
        # Text
        (
            {"text": "hello"},
            {"type": "input_text", "text": "hello"},
        ),
    ],
)
def test_format_request_message_content(content, exp_result):
    tru_result = OpenAIResponsesModel._format_request_message_content(content)
    assert tru_result == exp_result


def test_format_request_message_content_unsupported_type():
    content = {"unsupported": {}}

    with pytest.raises(TypeError, match="content_type=<unsupported> | unsupported type"):
        OpenAIResponsesModel._format_request_message_content(content)


def test_format_request_message_tool_call():
    tool_use = {
        "input": {"expression": "2+2"},
        "name": "calculator",
        "toolUseId": "c1",
    }

    tru_result = OpenAIResponsesModel._format_request_message_tool_call(tool_use)
    exp_result = {
        "type": "function_call",
        "call_id": "c1",
        "name": "calculator",
        "arguments": '{"expression": "2+2"}',
    }
    assert tru_result == exp_result


def test_format_request_tool_message():
    tool_result = {
        "content": [{"text": "4"}, {"json": ["4"]}],
        "status": "success",
        "toolUseId": "c1",
    }

    tru_result = OpenAIResponsesModel._format_request_tool_message(tool_result)
    exp_result = {
        "type": "function_call_output",
        "call_id": "c1",
        "output": '4\n["4"]',
    }
    assert tru_result == exp_result


def test_format_request_tool_message_with_image():
    """Test that tool results with images return an array output."""
    tool_result = {
        "content": [
            {"text": "Here is the image:"},
            {"image": {"format": "png", "source": {"bytes": b"fake_image_data"}}},
        ],
        "status": "success",
        "toolUseId": "c2",
    }

    tru_result = OpenAIResponsesModel._format_request_tool_message(tool_result)

    assert tru_result["type"] == "function_call_output"
    assert tru_result["call_id"] == "c2"
    # When images are present, output should be an array
    assert isinstance(tru_result["output"], list)
    assert len(tru_result["output"]) == 2
    assert tru_result["output"][0]["type"] == "input_text"
    assert tru_result["output"][0]["text"] == "Here is the image:"
    assert tru_result["output"][1]["type"] == "input_image"
    assert "image_url" in tru_result["output"][1]


def test_format_request_tool_message_with_document():
    """Test that tool results with documents return an array output."""
    tool_result = {
        "content": [
            {"document": {"format": "pdf", "name": "test.pdf", "source": {"bytes": b"fake_pdf_data"}}},
        ],
        "status": "success",
        "toolUseId": "c3",
    }

    tru_result = OpenAIResponsesModel._format_request_tool_message(tool_result)

    assert tru_result["type"] == "function_call_output"
    assert tru_result["call_id"] == "c3"
    # When documents are present, output should be an array
    assert isinstance(tru_result["output"], list)
    assert len(tru_result["output"]) == 1
    assert tru_result["output"][0]["type"] == "input_file"
    assert "file_url" in tru_result["output"][0]


def test_format_request_messages(system_prompt):
    messages = [
        {
            "content": [],
            "role": "user",
        },
        {
            "content": [{"text": "hello"}],
            "role": "user",
        },
        {
            "content": [
                {"text": "call tool"},
                {
                    "toolUse": {
                        "input": {"expression": "2+2"},
                        "name": "calculator",
                        "toolUseId": "c1",
                    },
                },
            ],
            "role": "assistant",
        },
        {
            "content": [{"toolResult": {"toolUseId": "c1", "status": "success", "content": [{"text": "4"}]}}],
            "role": "user",
        },
    ]

    tru_result = OpenAIResponsesModel._format_request_messages(messages)
    exp_result = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "input_text", "text": "call tool"}],
        },
        {
            "type": "function_call",
            "call_id": "c1",
            "name": "calculator",
            "arguments": '{"expression": "2+2"}',
        },
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": "4",
        },
    ]
    assert tru_result == exp_result


def test_format_request(model, messages, tool_specs, system_prompt):
    tru_request = model._format_request(messages, tool_specs, system_prompt)
    exp_request = {
        "model": "gpt-4o",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "test"}],
            }
        ],
        "stream": True,
        "instructions": system_prompt,
        "tools": [
            {
                "type": "function",
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
        ],
        "max_output_tokens": 100,
    }
    assert tru_request == exp_request


@pytest.mark.parametrize(
    ("event", "exp_chunk"),
    [
        # Message start
        (
            {"chunk_type": "message_start"},
            {"messageStart": {"role": "assistant"}},
        ),
        # Content Start - Tool Use
        (
            {
                "chunk_type": "content_start",
                "data_type": "tool",
                "data": unittest.mock.Mock(**{"function.name": "calculator", "id": "c1"}),
            },
            {"contentBlockStart": {"start": {"toolUse": {"name": "calculator", "toolUseId": "c1"}}}},
        ),
        # Content Start - Text
        (
            {"chunk_type": "content_start", "data_type": "text"},
            {"contentBlockStart": {"start": {}}},
        ),
        # Content Delta - Tool Use
        (
            {
                "chunk_type": "content_delta",
                "data_type": "tool",
                "data": unittest.mock.Mock(function=unittest.mock.Mock(arguments='{"expression": "2+2"}')),
            },
            {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"expression": "2+2"}'}}}},
        ),
        # Content Delta - Tool Use - None
        (
            {
                "chunk_type": "content_delta",
                "data_type": "tool",
                "data": unittest.mock.Mock(function=unittest.mock.Mock(arguments=None)),
            },
            {"contentBlockDelta": {"delta": {"toolUse": {"input": ""}}}},
        ),
        # Content Delta - Reasoning Text
        (
            {"chunk_type": "content_delta", "data_type": "reasoning_content", "data": "I'm thinking"},
            {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "I'm thinking"}}}},
        ),
        # Content Delta - Text
        (
            {"chunk_type": "content_delta", "data_type": "text", "data": "hello"},
            {"contentBlockDelta": {"delta": {"text": "hello"}}},
        ),
        # Content Stop
        (
            {"chunk_type": "content_stop"},
            {"contentBlockStop": {}},
        ),
        # Message Stop - Tool Use
        (
            {"chunk_type": "message_stop", "data": "tool_calls"},
            {"messageStop": {"stopReason": "tool_use"}},
        ),
        # Message Stop - Max Tokens
        (
            {"chunk_type": "message_stop", "data": "length"},
            {"messageStop": {"stopReason": "max_tokens"}},
        ),
        # Message Stop - End Turn
        (
            {"chunk_type": "message_stop", "data": "stop"},
            {"messageStop": {"stopReason": "end_turn"}},
        ),
        # Metadata
        (
            {
                "chunk_type": "metadata",
                "data": unittest.mock.Mock(input_tokens=100, output_tokens=50, total_tokens=150),
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
    tru_chunk = model._format_chunk(event)
    assert tru_chunk == exp_chunk


def test_format_chunk_unknown_type(model):
    event = {"chunk_type": "unknown"}

    with pytest.raises(RuntimeError, match="chunk_type=<unknown> | unknown type"):
        model._format_chunk(event)


@pytest.mark.asyncio
async def test_stream(openai_client, model_id, model, agenerator, alist):
    # Mock response events
    mock_text_event = unittest.mock.Mock(type="response.output_text.delta", delta="Hello")
    mock_complete_event = unittest.mock.Mock(
        type="response.completed",
        response=unittest.mock.Mock(usage=unittest.mock.Mock(input_tokens=10, output_tokens=5, total_tokens=15)),
    )

    openai_client.responses.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_text_event, mock_complete_event])
    )

    messages = [{"role": "user", "content": [{"text": "test"}]}]
    response = model.stream(messages)
    tru_events = await alist(response)

    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": "Hello"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                "metrics": {"latencyMs": 0},
            }
        },
    ]

    assert len(tru_events) == len(exp_events)
    expected_request = {
        "model": model_id,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "test"}]}],
        "stream": True,
        "max_output_tokens": 100,
    }
    openai_client.responses.create.assert_called_once_with(**expected_request)


@pytest.mark.asyncio
async def test_stream_with_tool_calls(openai_client, model, agenerator, alist):
    # Mock tool call events
    mock_tool_event = unittest.mock.Mock(
        type="response.output_item.added",
        item=unittest.mock.Mock(type="function_call", call_id="call_123", name="calculator", id="item_456"),
    )
    mock_args_event = unittest.mock.Mock(
        type="response.function_call_arguments.delta", delta='{"expression": "2+2"}', item_id="item_456"
    )
    mock_complete_event = unittest.mock.Mock(
        type="response.completed",
        response=unittest.mock.Mock(usage=unittest.mock.Mock(input_tokens=10, output_tokens=5, total_tokens=15)),
    )

    openai_client.responses.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_tool_event, mock_args_event, mock_complete_event])
    )

    messages = [{"role": "user", "content": [{"text": "calculate 2+2"}]}]
    response = model.stream(messages)
    tru_events = await alist(response)

    # Should include tool call events
    assert any("toolUse" in str(event) for event in tru_events)
    assert {"messageStop": {"stopReason": "tool_use"}} in tru_events


@pytest.mark.asyncio
async def test_stream_with_tool_calls_done_event(openai_client, model, agenerator, alist):
    """Test that response.function_call_arguments.done overwrites accumulated deltas."""
    mock_tool_event = unittest.mock.Mock(
        type="response.output_item.added",
        item=unittest.mock.Mock(type="function_call", call_id="call_1", name="calculator", id="item_1"),
    )
    # Simulate partial delta that would produce incomplete JSON
    mock_args_delta = unittest.mock.Mock(
        type="response.function_call_arguments.delta", delta='{"expr', item_id="item_1"
    )
    # The done event provides the complete, correct arguments
    mock_args_done = unittest.mock.Mock(
        type="response.function_call_arguments.done", arguments='{"expression": "2+2"}', item_id="item_1"
    )
    mock_complete_event = unittest.mock.Mock(
        type="response.completed",
        response=unittest.mock.Mock(usage=unittest.mock.Mock(input_tokens=10, output_tokens=5, total_tokens=15)),
    )

    openai_client.responses.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_tool_event, mock_args_delta, mock_args_done, mock_complete_event])
    )

    messages = [{"role": "user", "content": [{"text": "calculate 2+2"}]}]
    tru_events = await alist(model.stream(messages))

    # Find the tool use delta event and verify it has the final (done) arguments, not the partial delta
    tool_deltas = [e for e in tru_events if "contentBlockDelta" in e and "toolUse" in e["contentBlockDelta"]["delta"]]
    assert len(tool_deltas) == 1
    assert tool_deltas[0]["contentBlockDelta"]["delta"]["toolUse"]["input"] == '{"expression": "2+2"}'


@pytest.mark.asyncio
async def test_stream_response_incomplete(openai_client, model, agenerator, alist):
    """Test that response.incomplete sets stop_reason to length when max_output_tokens is reached."""
    mock_text_event = unittest.mock.Mock(type="response.output_text.delta", delta="Truncated resp")
    mock_incomplete_event = unittest.mock.Mock(
        type="response.incomplete",
        response=unittest.mock.Mock(
            usage=unittest.mock.Mock(input_tokens=10, output_tokens=100, total_tokens=110),
            incomplete_details=unittest.mock.Mock(reason="max_output_tokens"),
        ),
    )

    openai_client.responses.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_text_event, mock_incomplete_event])
    )

    messages = [{"role": "user", "content": [{"text": "write a long essay"}]}]
    tru_events = await alist(model.stream(messages))

    assert {"messageStop": {"stopReason": "max_tokens"}} in tru_events
    # Verify usage was still captured
    metadata_events = [e for e in tru_events if "metadata" in e]
    assert len(metadata_events) == 1
    assert metadata_events[0]["metadata"]["usage"]["inputTokens"] == 10
    assert metadata_events[0]["metadata"]["usage"]["outputTokens"] == 100


@pytest.mark.asyncio
async def test_stream_reasoning_content(openai_client, model, agenerator, alist):
    """Test that reasoning content (o1/o3 models) is streamed correctly."""
    mock_reasoning_event = unittest.mock.Mock(type="response.reasoning_text.delta", delta="Let me think...")
    mock_text_event = unittest.mock.Mock(type="response.output_text.delta", delta="The answer is 42")
    mock_complete_event = unittest.mock.Mock(
        type="response.completed",
        response=unittest.mock.Mock(usage=unittest.mock.Mock(input_tokens=10, output_tokens=20, total_tokens=30)),
    )

    openai_client.responses.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_reasoning_event, mock_text_event, mock_complete_event])
    )

    messages = [{"role": "user", "content": [{"text": "think step by step"}]}]
    tru_events = await alist(model.stream(messages))

    # Verify reasoning content block was emitted
    reasoning_deltas = [
        e for e in tru_events if "contentBlockDelta" in e and "reasoningContent" in e["contentBlockDelta"]["delta"]
    ]
    assert len(reasoning_deltas) == 1
    assert reasoning_deltas[0]["contentBlockDelta"]["delta"]["reasoningContent"]["text"] == "Let me think..."

    # Verify text content block was also emitted
    text_deltas = [e for e in tru_events if "contentBlockDelta" in e and "text" in e["contentBlockDelta"]["delta"]]
    assert len(text_deltas) == 1
    assert text_deltas[0]["contentBlockDelta"]["delta"]["text"] == "The answer is 42"

    # Verify content blocks were properly opened and closed (reasoning start/stop, then text start/stop)
    content_starts = [e for e in tru_events if "contentBlockStart" in e]
    content_stops = [e for e in tru_events if "contentBlockStop" in e]
    assert len(content_starts) == 2  # one for reasoning, one for text
    assert len(content_stops) == 2


@pytest.mark.asyncio
async def test_structured_output(openai_client, model, test_output_model_cls, alist):
    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    mock_parsed_instance = test_output_model_cls(name="John", age=30)
    mock_response = unittest.mock.Mock(output_parsed=mock_parsed_instance)

    openai_client.responses.parse = unittest.mock.AsyncMock(return_value=mock_response)

    stream = model.structured_output(test_output_model_cls, messages)
    events = await alist(stream)

    tru_result = events[-1]
    exp_result = {"output": test_output_model_cls(name="John", age=30)}
    assert tru_result == exp_result


@pytest.mark.asyncio
async def test_stream_context_overflow_exception(openai_client, model, messages):
    """Test that OpenAI context overflow errors are properly converted to ContextWindowOverflowException."""
    mock_error = openai.BadRequestError(
        message="This model's maximum context length is 4096 tokens.",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "context_length_exceeded"}},
    )
    mock_error.code = "context_length_exceeded"

    openai_client.responses.create.side_effect = mock_error

    with pytest.raises(ContextWindowOverflowException) as exc_info:
        async for _ in model.stream(messages):
            pass

    assert "maximum context length" in str(exc_info.value)
    assert exc_info.value.__cause__ == mock_error


@pytest.mark.asyncio
async def test_stream_rate_limit_as_throttle(openai_client, model, messages):
    """Test that rate limit errors are converted to ModelThrottledException."""
    mock_error = openai.RateLimitError(
        message="Rate limit exceeded",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "rate_limit_exceeded"}},
    )
    mock_error.code = "rate_limit_exceeded"

    openai_client.responses.create.side_effect = mock_error

    with pytest.raises(ModelThrottledException) as exc_info:
        async for _ in model.stream(messages):
            pass

    assert "Rate limit exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_stream_bad_request_non_context_overflow(openai_client, model, messages):
    """Test that non-context-overflow BadRequestErrors are re-raised."""
    mock_error = openai.BadRequestError(
        message="Invalid request format",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "invalid_request"}},
    )
    mock_error.code = "invalid_request"

    openai_client.responses.create.side_effect = mock_error

    with pytest.raises(openai.BadRequestError) as exc_info:
        async for _ in model.stream(messages):
            pass

    assert exc_info.value == mock_error


@pytest.mark.asyncio
async def test_stream_error_during_iteration(openai_client, model, messages, agenerator):
    """Test that errors during streaming iteration are properly handled."""
    mock_text_event = unittest.mock.Mock(type="response.output_text.delta", delta="Hello")

    async def error_generator():
        yield mock_text_event
        raise openai.RateLimitError(
            message="Rate limit during stream",
            response=unittest.mock.MagicMock(),
            body={"error": {"code": "rate_limit_exceeded"}},
        )

    openai_client.responses.create = unittest.mock.AsyncMock(return_value=error_generator())

    with pytest.raises(ModelThrottledException) as exc_info:
        async for _ in model.stream(messages):
            pass

    assert "Rate limit during stream" in str(exc_info.value)


@pytest.mark.asyncio
async def test_stream_context_overflow_during_iteration(openai_client, model, messages):
    """Test that context overflow during streaming iteration is properly handled."""
    mock_text_event = unittest.mock.Mock(type="response.output_text.delta", delta="Hello")

    async def error_generator():
        yield mock_text_event
        error = openai.BadRequestError(
            message="Context length exceeded during stream",
            response=unittest.mock.MagicMock(),
            body={"error": {"code": "context_length_exceeded"}},
        )
        error.code = "context_length_exceeded"
        raise error

    openai_client.responses.create = unittest.mock.AsyncMock(return_value=error_generator())

    with pytest.raises(ContextWindowOverflowException) as exc_info:
        async for _ in model.stream(messages):
            pass

    assert "Context length exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_structured_output_context_overflow_exception(openai_client, model, messages, test_output_model_cls):
    """Test that structured output handles context overflow properly."""
    mock_error = openai.BadRequestError(
        message="This model's maximum context length is 4096 tokens.",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "context_length_exceeded"}},
    )
    mock_error.code = "context_length_exceeded"

    openai_client.responses.parse.side_effect = mock_error

    with pytest.raises(ContextWindowOverflowException) as exc_info:
        async for _ in model.structured_output(test_output_model_cls, messages):
            pass

    assert "maximum context length" in str(exc_info.value)
    assert exc_info.value.__cause__ == mock_error


@pytest.mark.asyncio
async def test_structured_output_rate_limit_as_throttle(openai_client, model, messages, test_output_model_cls):
    """Test that structured output handles rate limit errors properly."""
    mock_error = openai.RateLimitError(
        message="Rate limit exceeded",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "rate_limit_exceeded"}},
    )
    mock_error.code = "rate_limit_exceeded"

    openai_client.responses.parse.side_effect = mock_error

    with pytest.raises(ModelThrottledException) as exc_info:
        async for _ in model.structured_output(test_output_model_cls, messages):
            pass

    assert "Rate limit exceeded" in str(exc_info.value)
    assert exc_info.value.__cause__ == mock_error


def test_config_validation_warns_on_unknown_keys(openai_client, captured_warnings):
    """Test that unknown config keys emit a warning."""
    OpenAIResponsesModel({"api_key": "test"}, model_id="test-model", invalid_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "invalid_param" in str(captured_warnings[0].message)


def test_update_config_validation_warns_on_unknown_keys(model, captured_warnings):
    """Test that update_config warns on unknown keys."""
    model.update_config(wrong_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "wrong_param" in str(captured_warnings[0].message)


@pytest.mark.parametrize(
    ("tool_choice", "expected"),
    [
        (None, {}),
        ({"auto": {}}, {"tool_choice": "auto"}),
        ({"any": {}}, {"tool_choice": "required"}),
        ({"tool": {"name": "calculator"}}, {"tool_choice": {"type": "function", "name": "calculator"}}),
        ({"unknown": {}}, {"tool_choice": "auto"}),  # Test default fallback
    ],
)
def test_format_request_tool_choice(tool_choice, expected):
    """Test that tool_choice is properly formatted for the Responses API."""
    result = OpenAIResponsesModel._format_request_tool_choice(tool_choice)
    assert result == expected


def test_format_request_with_tool_choice(model, messages, tool_specs):
    """Test that tool_choice is properly included in the request."""
    tool_choice = {"tool": {"name": "test_tool"}}
    request = model._format_request(messages, tool_specs, tool_choice=tool_choice)

    assert "tool_choice" in request
    assert request["tool_choice"] == {"type": "function", "name": "test_tool"}


def test_format_request_message_content_image_size_limit():
    """Test that oversized images raise ValueError."""
    oversized_data = b"x" * (_MAX_MEDIA_SIZE_BYTES + 1)
    content = {"image": {"format": "png", "source": {"bytes": oversized_data}}}

    with pytest.raises(ValueError, match="Image size .* exceeds maximum"):
        OpenAIResponsesModel._format_request_message_content(content)


def test_format_request_message_content_document_size_limit():
    """Test that oversized documents raise ValueError."""
    oversized_data = b"x" * (_MAX_MEDIA_SIZE_BYTES + 1)
    content = {"document": {"format": "pdf", "name": "large.pdf", "source": {"bytes": oversized_data}}}

    with pytest.raises(ValueError, match="Document size .* exceeds maximum"):
        OpenAIResponsesModel._format_request_message_content(content)


def test_format_request_tool_message_image_size_limit():
    """Test that oversized images in tool results raise ValueError."""
    oversized_data = b"x" * (_MAX_MEDIA_SIZE_BYTES + 1)
    tool_result = {
        "content": [{"image": {"format": "png", "source": {"bytes": oversized_data}}}],
        "status": "success",
        "toolUseId": "c1",
    }

    with pytest.raises(ValueError, match="Image size .* exceeds maximum"):
        OpenAIResponsesModel._format_request_tool_message(tool_result)


def test_format_request_tool_message_document_size_limit():
    """Test that oversized documents in tool results raise ValueError."""
    oversized_data = b"x" * (_MAX_MEDIA_SIZE_BYTES + 1)
    tool_result = {
        "content": [{"document": {"format": "pdf", "name": "large.pdf", "source": {"bytes": oversized_data}}}],
        "status": "success",
        "toolUseId": "c1",
    }

    with pytest.raises(ValueError, match="Document size .* exceeds maximum"):
        OpenAIResponsesModel._format_request_tool_message(tool_result)


def test_openai_version_check():
    """Test that module import fails with old OpenAI SDK version."""
    import importlib

    import strands.models.openai_responses as openai_responses_module

    def mock_old_version(package_name: str) -> str:
        if package_name == "openai":
            return "1.99.0"
        from importlib.metadata import version

        return version(package_name)

    def mock_valid_version(package_name: str) -> str:
        if package_name == "openai":
            return "2.0.0"
        from importlib.metadata import version

        return version(package_name)

    with unittest.mock.patch("importlib.metadata.version", mock_old_version):
        with pytest.raises(ImportError, match="OpenAIResponsesModel requires openai>=2.0.0"):
            importlib.reload(openai_responses_module)

    # Reload with valid version to restore module state
    with unittest.mock.patch("importlib.metadata.version", mock_valid_version):
        importlib.reload(openai_responses_module)
