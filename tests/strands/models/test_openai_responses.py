import unittest.mock

import openai
import pydantic
import pytest

import strands
from strands.models.openai_responses import OpenAIResponsesModel
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
    assert exc_info.value.__cause__ == mock_error


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
    from strands.models.openai_responses import MAX_MEDIA_SIZE_BYTES

    oversized_data = b"x" * (MAX_MEDIA_SIZE_BYTES + 1)
    content = {"image": {"format": "png", "source": {"bytes": oversized_data}}}

    with pytest.raises(ValueError, match="Image size .* exceeds maximum"):
        OpenAIResponsesModel._format_request_message_content(content)


def test_format_request_message_content_document_size_limit():
    """Test that oversized documents raise ValueError."""
    from strands.models.openai_responses import MAX_MEDIA_SIZE_BYTES

    oversized_data = b"x" * (MAX_MEDIA_SIZE_BYTES + 1)
    content = {"document": {"format": "pdf", "name": "large.pdf", "source": {"bytes": oversized_data}}}

    with pytest.raises(ValueError, match="Document size .* exceeds maximum"):
        OpenAIResponsesModel._format_request_message_content(content)


def test_format_request_tool_message_image_size_limit():
    """Test that oversized images in tool results raise ValueError."""
    from strands.models.openai_responses import MAX_MEDIA_SIZE_BYTES

    oversized_data = b"x" * (MAX_MEDIA_SIZE_BYTES + 1)
    tool_result = {
        "content": [{"image": {"format": "png", "source": {"bytes": oversized_data}}}],
        "status": "success",
        "toolUseId": "c1",
    }

    with pytest.raises(ValueError, match="Image size .* exceeds maximum"):
        OpenAIResponsesModel._format_request_tool_message(tool_result)


def test_format_request_tool_message_document_size_limit():
    """Test that oversized documents in tool results raise ValueError."""
    from strands.models.openai_responses import MAX_MEDIA_SIZE_BYTES

    oversized_data = b"x" * (MAX_MEDIA_SIZE_BYTES + 1)
    tool_result = {
        "content": [{"document": {"format": "pdf", "name": "large.pdf", "source": {"bytes": oversized_data}}}],
        "status": "success",
        "toolUseId": "c1",
    }

    with pytest.raises(ValueError, match="Document size .* exceeds maximum"):
        OpenAIResponsesModel._format_request_tool_message(tool_result)
