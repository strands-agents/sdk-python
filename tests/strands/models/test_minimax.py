import unittest.mock

import openai
import pydantic
import pytest

import strands
from strands.models.minimax import MinimaxModel
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException


@pytest.fixture
def openai_client():
    with unittest.mock.patch.object(strands.models.openai.openai, "AsyncOpenAI") as mock_client_cls:
        mock_client = unittest.mock.AsyncMock()
        # Make the mock client work as an async context manager
        mock_client.__aenter__ = unittest.mock.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = unittest.mock.AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def model_id():
    return "MiniMax-M2.7"


@pytest.fixture
def model(openai_client, model_id):
    _ = openai_client

    return MinimaxModel(model_id=model_id, params={"max_tokens": 1})


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


def test__init__(openai_client, model_id):
    _ = openai_client

    model = MinimaxModel(model_id=model_id, params={"max_tokens": 1})

    tru_config = model.get_config()
    exp_config = {"model_id": "MiniMax-M2.7", "params": {"max_tokens": 1}}

    assert tru_config == exp_config


def test__init__default_client_args(openai_client):
    """Test that default client_args include MiniMax base_url and API key."""
    _ = openai_client

    with unittest.mock.patch.dict("os.environ", {"MINIMAX_API_KEY": "test-key"}):
        model = MinimaxModel(model_id="MiniMax-M2.7")

    assert model.client_args["base_url"] == "https://api.minimax.io/v1"
    assert model.client_args["api_key"] == "test-key"


def test__init__custom_client_args(openai_client):
    """Test that custom client_args override defaults."""
    _ = openai_client

    model = MinimaxModel(
        client_args={"base_url": "https://custom.api.com/v1", "api_key": "custom-key"},
        model_id="MiniMax-M2.7",
    )

    assert model.client_args["base_url"] == "https://custom.api.com/v1"
    assert model.client_args["api_key"] == "custom-key"


def test__init__with_injected_client(openai_client):
    """Test initialization with a pre-configured client."""
    mock_client = unittest.mock.AsyncMock()

    model = MinimaxModel(client=mock_client, model_id="MiniMax-M2.7")

    assert model._custom_client is mock_client


def test__init__with_both_client_and_client_args_raises_error():
    """Test that providing both client and client_args raises ValueError."""
    mock_client = unittest.mock.AsyncMock()

    with pytest.raises(ValueError, match="Only one of 'client' or 'client_args' should be provided"):
        MinimaxModel(client=mock_client, client_args={"api_key": "test"}, model_id="MiniMax-M2.7")


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


def test_format_request_includes_stream_options(model, messages, tool_specs, system_prompt):
    """Test that stream_options is included in the request for usage tracking."""
    tru_request = model.format_request(messages, tool_specs, system_prompt)

    assert tru_request["stream_options"] == {"include_usage": True}
    assert tru_request["model"] == "MiniMax-M2.7"
    assert tru_request["stream"] is True


def test_format_request_no_empty_tools(model, messages, system_prompt):
    """Test that empty tools list is removed from the request."""
    tru_request = model.format_request(messages, None, system_prompt)

    assert "tools" not in tru_request


def test_format_request_with_tools(model, messages, tool_specs, system_prompt):
    """Test that non-empty tools list is preserved."""
    tru_request = model.format_request(messages, tool_specs, system_prompt)

    assert "tools" in tru_request
    assert len(tru_request["tools"]) == 1
    assert tru_request["tools"][0]["function"]["name"] == "test_tool"


def test_format_request(model, messages, tool_specs, system_prompt):
    tru_request = model.format_request(messages, tool_specs, system_prompt)
    exp_request = {
        "messages": [
            {
                "content": system_prompt,
                "role": "system",
            },
            {
                "content": [{"text": "test", "type": "text"}],
                "role": "user",
            },
        ],
        "model": "MiniMax-M2.7",
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [
            {
                "function": {
                    "description": "A test tool",
                    "name": "test_tool",
                    "parameters": {
                        "properties": {
                            "input": {"type": "string"},
                        },
                        "required": ["input"],
                        "type": "object",
                    },
                },
                "type": "function",
            },
        ],
        "max_tokens": 1,
    }
    assert tru_request == exp_request


@pytest.mark.asyncio
async def test_stream(openai_client, model_id, model, agenerator, alist):
    mock_delta = unittest.mock.Mock(content="Hello", tool_calls=None, reasoning_content=None)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_event_3 = unittest.mock.Mock()

    openai_client.chat.completions.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_event_1, mock_event_2, mock_event_3])
    )

    messages = [{"role": "user", "content": [{"text": "test"}]}]
    response = model.stream(messages)
    tru_events = await alist(response)

    # Should have: messageStart, contentBlockStart, 2x contentBlockDelta, contentBlockStop, messageStop
    assert len(tru_events) >= 4
    assert tru_events[0] == {"messageStart": {"role": "assistant"}}

    # Verify the request was made with stream_options for usage tracking
    call_kwargs = openai_client.chat.completions.create.call_args[1]
    assert call_kwargs["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_stream_context_overflow_exception(openai_client, model, messages):
    """Test that context overflow errors are properly converted."""
    mock_error = openai.BadRequestError(
        message="This model's maximum context length exceeded.",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "context_length_exceeded"}},
    )
    mock_error.code = "context_length_exceeded"

    openai_client.chat.completions.create.side_effect = mock_error

    with pytest.raises(ContextWindowOverflowException):
        async for _ in model.stream(messages):
            pass


@pytest.mark.asyncio
async def test_stream_rate_limit_as_throttle(openai_client, model, messages):
    """Test that rate limit errors are converted to ModelThrottledException."""
    mock_error = openai.RateLimitError(
        message="Rate limit reached.",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "rate_limit_exceeded"}},
    )
    mock_error.code = "rate_limit_exceeded"

    openai_client.chat.completions.create.side_effect = mock_error

    with pytest.raises(ModelThrottledException):
        async for _ in model.stream(messages):
            pass


@pytest.mark.asyncio
async def test_stream_with_tool_calls(openai_client, model, agenerator, alist):
    mock_tool_call = unittest.mock.Mock(index=0)
    mock_delta_1 = unittest.mock.Mock(content="I'll help", tool_calls=[mock_tool_call], reasoning_content=None)
    mock_delta_2 = unittest.mock.Mock(content="", tool_calls=None, reasoning_content=None)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_1)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="tool_calls", delta=mock_delta_2)])
    mock_event_3 = unittest.mock.Mock()

    openai_client.chat.completions.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_event_1, mock_event_2, mock_event_3])
    )

    messages = [{"role": "user", "content": [{"text": "use the tool"}]}]
    response = model.stream(messages)
    tru_events = await alist(response)

    # Should contain tool_use stop reason
    stop_events = [e for e in tru_events if "messageStop" in e]
    assert len(stop_events) == 1
    assert stop_events[0]["messageStop"]["stopReason"] == "tool_use"


@pytest.mark.asyncio
async def test_structured_output(openai_client, model, test_output_model_cls, alist):
    """Test structured output using regular completion with response_format."""
    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    mock_choice = unittest.mock.Mock()
    mock_choice.message.content = '{"name": "John", "age": 30}'
    mock_response = unittest.mock.Mock()
    mock_response.choices = [mock_choice]

    openai_client.chat.completions.create = unittest.mock.AsyncMock(return_value=mock_response)

    stream = model.structured_output(test_output_model_cls, messages)
    events = await alist(stream)

    tru_result = events[-1]
    exp_result = {"output": test_output_model_cls(name="John", age=30)}
    assert tru_result == exp_result


@pytest.mark.asyncio
async def test_structured_output_with_think_tags(openai_client, model, test_output_model_cls, alist):
    """Test structured output strips <think> tags from MiniMax response."""
    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    mock_choice = unittest.mock.Mock()
    mock_choice.message.content = '<think>\nLet me think about this.\n</think>\n{"name": "Alice", "age": 25}'
    mock_response = unittest.mock.Mock()
    mock_response.choices = [mock_choice]

    openai_client.chat.completions.create = unittest.mock.AsyncMock(return_value=mock_response)

    stream = model.structured_output(test_output_model_cls, messages)
    events = await alist(stream)

    tru_result = events[-1]
    exp_result = {"output": test_output_model_cls(name="Alice", age=25)}
    assert tru_result == exp_result


def test_config_validation_warns_on_unknown_keys(openai_client, captured_warnings):
    """Test that unknown config keys emit a warning."""
    MinimaxModel(model_id="MiniMax-M2.7", invalid_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "invalid_param" in str(captured_warnings[0].message)


def test_update_config_validation_warns_on_unknown_keys(model, captured_warnings):
    """Test that update_config warns on unknown keys."""
    model.update_config(wrong_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "wrong_param" in str(captured_warnings[0].message)


def test_format_request_messages(system_prompt):
    """Test message formatting inherits from OpenAI properly."""
    messages = [
        {
            "content": [{"text": "hello"}],
            "role": "user",
        },
    ]

    tru_result = MinimaxModel.format_request_messages(messages, system_prompt)
    exp_result = [
        {
            "content": system_prompt,
            "role": "system",
        },
        {
            "content": [{"text": "hello", "type": "text"}],
            "role": "user",
        },
    ]
    assert tru_result == exp_result


def test_format_request_message_content():
    """Test content formatting inherits from OpenAI properly."""
    content = {"text": "hello"}
    tru_result = MinimaxModel.format_request_message_content(content)
    exp_result = {"type": "text", "text": "hello"}
    assert tru_result == exp_result


def test_format_request_message_tool_call():
    """Test tool call formatting inherits from OpenAI properly."""
    tool_use = {
        "input": {"expression": "2+2"},
        "name": "calculator",
        "toolUseId": "c1",
    }

    tru_result = MinimaxModel.format_request_message_tool_call(tool_use)
    exp_result = {
        "function": {
            "arguments": '{"expression": "2+2"}',
            "name": "calculator",
        },
        "id": "c1",
        "type": "function",
    }
    assert tru_result == exp_result


@pytest.mark.parametrize(
    ("event", "exp_chunk"),
    [
        # Message start
        (
            {"chunk_type": "message_start"},
            {"messageStart": {"role": "assistant"}},
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
        # Message Stop - End Turn
        (
            {"chunk_type": "message_stop", "data": "stop"},
            {"messageStop": {"stopReason": "end_turn"}},
        ),
    ],
)
def test_format_chunk(event, exp_chunk, model):
    tru_chunk = model.format_chunk(event)
    assert tru_chunk == exp_chunk


@pytest.mark.parametrize(
    "input_text, expected",
    [
        ("<think>\nSome reasoning.\n</think>\n{}", "{}"),
        ("<think>quick thought</think>Hello", "Hello"),
        ("No think tags here", "No think tags here"),
        ('<think>\nMultiple\nlines\n</think>\n{"key": "value"}', '{"key": "value"}'),
        ('```json\n{"key": "value"}\n```', '{"key": "value"}'),
        ('```\n{"key": "value"}\n```', '{"key": "value"}'),
        ('<think>\nthinking\n</think>\n```json\n{"key": "value"}\n```', '{"key": "value"}'),
    ],
)
def test_clean_response_content(input_text, expected):
    """Test that _clean_response_content removes think tags and code blocks."""
    assert MinimaxModel._clean_response_content(input_text) == expected


@pytest.mark.asyncio
async def test_stream_with_injected_client(model_id, agenerator, alist):
    """Test that stream works with an injected client."""
    mock_injected_client = unittest.mock.AsyncMock()
    mock_injected_client.close = unittest.mock.AsyncMock()

    mock_delta = unittest.mock.Mock(content="Hello", tool_calls=None, reasoning_content=None)
    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_event_3 = unittest.mock.Mock()

    mock_injected_client.chat.completions.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_event_1, mock_event_2, mock_event_3])
    )

    model = MinimaxModel(client=mock_injected_client, model_id=model_id, params={"max_tokens": 1})

    messages = [{"role": "user", "content": [{"text": "test"}]}]
    response = model.stream(messages)
    tru_events = await alist(response)

    assert len(tru_events) > 0
    mock_injected_client.chat.completions.create.assert_called_once()
    mock_injected_client.close.assert_not_called()
