import unittest.mock

import openai
import pydantic
import pytest

from strands.models.portkey import PortkeyModel
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException


@pytest.fixture
def mock_portkey_client():
    """Create a mock AsyncPortkey client."""
    mock_client = unittest.mock.AsyncMock()
    mock_client.close = unittest.mock.AsyncMock()
    return mock_client


@pytest.fixture
def model_id():
    return "gpt-4o"


@pytest.fixture
def model(mock_portkey_client, model_id):
    return PortkeyModel(client=mock_portkey_client, model_id=model_id)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt():
    return "s1"


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
def test_output_model_cls():
    class TestOutputModel(pydantic.BaseModel):
        name: str
        age: int

    return TestOutputModel


def test__init__with_api_key_only():
    """Test simplest initialization with just api_key and model_id."""
    model = PortkeyModel(api_key="pk-test", model_id="gpt-4o")

    config = model.get_config()
    assert config["model_id"] == "gpt-4o"
    assert config["api_key"] == "pk-test"


def test__init__with_virtual_key():
    """Test initialization with api_key and virtual_key."""
    model = PortkeyModel(api_key="pk-test", virtual_key="vk-openai", model_id="gpt-4o")

    config = model.get_config()
    assert config["api_key"] == "pk-test"
    assert config["virtual_key"] == "vk-openai"
    assert config["model_id"] == "gpt-4o"


def test__init__with_config_slug():
    """Test initialization with a Portkey config slug for routing/fallbacks."""
    model = PortkeyModel(api_key="pk-test", config="cf-xxx", model_id="gpt-4o")

    config = model.get_config()
    assert config["config"] == "cf-xxx"


def test__init__with_provider():
    """Test initialization with an explicit provider."""
    model = PortkeyModel(api_key="pk-test", provider="anthropic", model_id="claude-sonnet-4-20250514")

    config = model.get_config()
    assert config["provider"] == "anthropic"
    assert config["model_id"] == "claude-sonnet-4-20250514"


def test__init__with_params():
    """Test initialization with model parameters."""
    model = PortkeyModel(api_key="pk-test", model_id="gpt-4o", params={"max_tokens": 100, "temperature": 0.7})

    config = model.get_config()
    assert config["params"] == {"max_tokens": 100, "temperature": 0.7}


def test__init__with_injected_client(mock_portkey_client):
    """Test initialization with a pre-configured client."""
    model = PortkeyModel(client=mock_portkey_client, model_id="gpt-4o")

    assert model._custom_client is mock_portkey_client
    assert model.get_config()["model_id"] == "gpt-4o"


def test_update_config(model):
    model.update_config(model_id="claude-sonnet-4-20250514")

    assert model.get_config()["model_id"] == "claude-sonnet-4-20250514"


def test_update_config_portkey_args(model):
    """Test that Portkey client args can be updated via update_config."""
    model.update_config(trace_id="trace-123", metadata={"user": "test"})

    config = model.get_config()
    assert config["trace_id"] == "trace-123"
    assert config["metadata"] == {"user": "test"}


def test_config_validation_warns_on_unknown_keys(mock_portkey_client, captured_warnings):
    """Test that unknown config keys emit a warning."""
    PortkeyModel(client=mock_portkey_client, model_id="gpt-4o", invalid_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "invalid_param" in str(captured_warnings[0].message)


def test_update_config_validation_warns_on_unknown_keys(model, captured_warnings):
    """Test that update_config warns on unknown keys."""
    model.update_config(wrong_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "wrong_param" in str(captured_warnings[0].message)


@pytest.mark.asyncio
async def test_stream(mock_portkey_client, model, model_id, agenerator, alist):
    mock_delta_1 = unittest.mock.Mock(content="Hello", tool_calls=None, reasoning_content=None)
    mock_delta_2 = unittest.mock.Mock(content=" world", tool_calls=None, reasoning_content=None)
    mock_delta_3 = unittest.mock.Mock(content="", tool_calls=None, reasoning_content=None)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_1)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_2)])
    mock_event_3 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta_3)])
    mock_event_4 = unittest.mock.Mock()

    mock_portkey_client.chat.completions.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_event_1, mock_event_2, mock_event_3, mock_event_4])
    )

    messages = [{"role": "user", "content": [{"text": "say hello"}]}]
    response = model.stream(messages)
    tru_events = await alist(response)

    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": "Hello"}}},
        {"contentBlockDelta": {"delta": {"text": " world"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": mock_event_4.usage.prompt_tokens,
                    "outputTokens": mock_event_4.usage.completion_tokens,
                    "totalTokens": mock_event_4.usage.total_tokens,
                },
                "metrics": {"latencyMs": 0},
            }
        },
    ]

    assert tru_events == exp_events

    mock_portkey_client.chat.completions.create.assert_called_once_with(
        messages=[{"role": "user", "content": [{"text": "say hello", "type": "text"}]}],
        model=model_id,
        stream=True,
        stream_options={"include_usage": True},
        tools=[],
    )


@pytest.mark.asyncio
async def test_stream_with_tool_calls(mock_portkey_client, model, agenerator, alist):
    mock_tool_call = unittest.mock.Mock(index=0)
    mock_delta_1 = unittest.mock.Mock(content="Let me check", tool_calls=[mock_tool_call], reasoning_content=None)
    mock_delta_2 = unittest.mock.Mock(content="", tool_calls=None, reasoning_content=None)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_1)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="tool_calls", delta=mock_delta_2)])
    mock_event_3 = unittest.mock.Mock(usage=None)

    mock_portkey_client.chat.completions.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_event_1, mock_event_2, mock_event_3])
    )

    messages = [{"role": "user", "content": [{"text": "use a tool"}]}]
    response = model.stream(messages)
    tru_events = await alist(response)

    assert tru_events[0] == {"messageStart": {"role": "assistant"}}
    assert {"messageStop": {"stopReason": "tool_use"}} in tru_events


@pytest.mark.asyncio
async def test_stream_empty(mock_portkey_client, model, model_id, agenerator, alist):
    mock_delta = unittest.mock.Mock(content=None, tool_calls=None, reasoning_content=None)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_event_3 = unittest.mock.Mock()
    mock_event_4 = unittest.mock.Mock(usage=None)

    mock_portkey_client.chat.completions.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_event_1, mock_event_2, mock_event_3, mock_event_4]),
    )

    messages = [{"role": "user", "content": []}]
    response = model.stream(messages)
    tru_events = await alist(response)

    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]

    assert len(tru_events) == len(exp_events)


@pytest.mark.asyncio
async def test_structured_output(mock_portkey_client, model, test_output_model_cls, alist):
    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    mock_parsed_instance = test_output_model_cls(name="John", age=30)
    mock_choice = unittest.mock.Mock()
    mock_choice.message.parsed = mock_parsed_instance
    mock_response = unittest.mock.Mock()
    mock_response.choices = [mock_choice]

    mock_portkey_client.beta.chat.completions.parse = unittest.mock.AsyncMock(return_value=mock_response)

    stream = model.structured_output(test_output_model_cls, messages)
    events = await alist(stream)

    tru_result = events[-1]
    exp_result = {"output": test_output_model_cls(name="John", age=30)}
    assert tru_result == exp_result


@pytest.mark.asyncio
async def test_stream_context_overflow_exception(mock_portkey_client, model, messages):
    """Test that context overflow errors are properly converted."""
    mock_error = openai.BadRequestError(
        message="This model's maximum context length is 4096 tokens.",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "context_length_exceeded"}},
    )
    mock_error.code = "context_length_exceeded"

    mock_portkey_client.chat.completions.create.side_effect = mock_error

    with pytest.raises(ContextWindowOverflowException):
        async for _ in model.stream(messages):
            pass


@pytest.mark.asyncio
async def test_stream_rate_limit_as_throttle(mock_portkey_client, model, messages):
    """Test that rate limit errors are converted to ModelThrottledException."""
    mock_error = openai.RateLimitError(
        message="Rate limit exceeded.",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "rate_limit_exceeded"}},
    )
    mock_error.code = "rate_limit_exceeded"

    mock_portkey_client.chat.completions.create.side_effect = mock_error

    with pytest.raises(ModelThrottledException):
        async for _ in model.stream(messages):
            pass


@pytest.mark.asyncio
async def test_get_client_creates_async_portkey_from_config():
    """Test that _get_client extracts Portkey args from config and creates an AsyncPortkey client."""
    mock_module = unittest.mock.MagicMock()
    mock_client = unittest.mock.AsyncMock()
    mock_client.close = unittest.mock.AsyncMock()
    mock_module.AsyncPortkey.return_value = mock_client

    with unittest.mock.patch.dict("sys.modules", {"portkey_ai": mock_module}):
        model = PortkeyModel(api_key="pk-test", virtual_key="vk-openai", model_id="gpt-4o")

        async with model._get_client() as client:
            assert client is mock_client

        # Only Portkey client keys should be passed, not model_id or params
        mock_module.AsyncPortkey.assert_called_once_with(api_key="pk-test", virtual_key="vk-openai")
        mock_client.close.assert_called_once()


@pytest.mark.asyncio
async def test_get_client_uses_injected_client(mock_portkey_client):
    """Test that _get_client uses the injected client and does not close it."""
    model = PortkeyModel(client=mock_portkey_client, model_id="gpt-4o")

    async with model._get_client() as client:
        assert client is mock_portkey_client

    mock_portkey_client.close.assert_not_called()


def test_format_request(model, messages, tool_specs, system_prompt):
    """Test that format_request produces OpenAI-compatible request (inherited)."""
    tru_request = model.format_request(messages, tool_specs, system_prompt)

    assert tru_request["model"] == "gpt-4o"
    assert tru_request["stream"] is True
    assert tru_request["stream_options"] == {"include_usage": True}
    assert len(tru_request["tools"]) == 1
    assert tru_request["tools"][0]["function"]["name"] == "test_tool"
    # Portkey client keys should NOT appear in the request
    assert "api_key" not in tru_request
    assert "virtual_key" not in tru_request


def test_format_request_does_not_leak_portkey_args():
    """Test that Portkey client args (api_key, virtual_key, etc.) don't leak into API requests."""
    model = PortkeyModel(
        api_key="pk-test",
        virtual_key="vk-test",
        provider="openai",
        trace_id="trace-123",
        model_id="gpt-4o",
    )

    messages = [{"role": "user", "content": [{"text": "hello"}]}]
    request = model.format_request(messages)

    assert "api_key" not in request
    assert "virtual_key" not in request
    assert "provider" not in request
    assert "trace_id" not in request
    assert request["model"] == "gpt-4o"
