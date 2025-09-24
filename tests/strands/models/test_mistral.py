import unittest.mock

import pydantic
import pytest

import strands
from strands.models.mistral import MistralModel
from strands.types.exceptions import ModelThrottledException


@pytest.fixture
def mistral_client():
    with unittest.mock.patch.object(strands.models.mistral.mistralai, "Mistral") as mock_client_cls:
        mock_client = unittest.mock.AsyncMock()
        mock_client_cls.return_value.__aenter__.return_value = mock_client
        yield mock_client


@pytest.fixture
def model_id():
    return "mistral-large-latest"


@pytest.fixture
def max_tokens():
    return 100


@pytest.fixture
def model(model_id, max_tokens):
    return MistralModel(model_id=model_id, max_tokens=max_tokens)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def tool_use_messages():
    return [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "calc_123",
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                    },
                },
            ],
        }
    ]


@pytest.fixture
def system_prompt():
    return "You are a helpful assistant"


@pytest.fixture
def test_output_model_cls():
    class TestOutputModel(pydantic.BaseModel):
        name: str
        age: int

    return TestOutputModel


def test__init__model_configs(mistral_client, model_id, max_tokens):
    _ = mistral_client

    model = MistralModel(model_id=model_id, max_tokens=max_tokens, temperature=0.7)

    actual_temperature = model.get_config().get("temperature")
    exp_temperature = 0.7

    assert actual_temperature == exp_temperature


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    actual_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert actual_model_id == exp_model_id


def test_format_request_default(model, messages, model_id):
    actual_request = model.format_request(messages)
    exp_request = {
        "model": model_id,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 100,
        "stream": True,
    }

    assert actual_request == exp_request


def test_format_request_with_temperature(model, messages, model_id):
    model.update_config(temperature=0.8)

    actual_request = model.format_request(messages)
    exp_request = {
        "model": model_id,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 100,
        "temperature": 0.8,
        "stream": True,
    }

    assert actual_request == exp_request


def test_format_request_with_system_prompt(model, messages, model_id, system_prompt):
    actual_request = model.format_request(messages, system_prompt=system_prompt)
    exp_request = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "test"},
        ],
        "max_tokens": 100,
        "stream": True,
    }

    assert actual_request == exp_request


def test_format_request_with_tool_use(model, model_id):
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "calc_123",
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                    },
                },
            ],
        },
    ]

    actual_request = model.format_request(messages)
    exp_request = {
        "model": model_id,
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "2+2"}',
                        },
                        "id": "calc_123",
                        "type": "function",
                    }
                ],
            }
        ],
        "max_tokens": 100,
        "stream": True,
    }

    assert actual_request == exp_request


def test_format_request_with_tool_result(model, model_id):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "calc_123",
                        "status": "success",
                        "content": [{"text": "4"}, {"json": {"result": 4}}],
                    }
                }
            ],
        }
    ]

    actual_request = model.format_request(messages)
    exp_request = {
        "model": model_id,
        "messages": [
            {
                "role": "tool",
                "name": "calc",
                "content": '4\n{"result": 4}',
                "tool_call_id": "calc_123",
            }
        ],
        "max_tokens": 100,
        "stream": True,
    }

    assert actual_request == exp_request


def test_format_request_with_tool_specs(model, messages, model_id):
    tool_specs = [
        {
            "name": "calculator",
            "description": "Calculate mathematical expressions",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                }
            },
        }
    ]

    actual_request = model.format_request(messages, tool_specs)
    exp_request = {
        "model": model_id,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 100,
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Calculate mathematical expressions",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                        "required": ["expression"],
                    },
                },
            }
        ],
    }

    assert actual_request == exp_request


def test_format_request_with_all_optional_params(model, messages, model_id):
    model.update_config(
        temperature=0.7,
        top_p=0.9,
    )

    tool_specs = [
        {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"json": {"type": "object"}},
        }
    ]

    actual_request = model.format_request(messages, tool_specs)
    exp_request = {
        "model": model_id,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": True,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object"},
                },
            }
        ],
    }

    assert actual_request == exp_request


def test_format_chunk_message_start(model):
    event = {"chunk_type": "message_start"}

    actual_chunk = model.format_chunk(event)
    exp_chunk = {"messageStart": {"role": "assistant"}}

    assert actual_chunk == exp_chunk


def test_format_chunk_content_start_text(model):
    event = {"chunk_type": "content_start", "data_type": "text"}

    actual_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStart": {"start": {}}}

    assert actual_chunk == exp_chunk


def test_format_chunk_content_start_tool(model):
    mock_tool_call = unittest.mock.Mock()
    mock_tool_call.function.name = "calculator"
    mock_tool_call.id = "calc_123"

    event = {"chunk_type": "content_start", "data_type": "tool", "data": mock_tool_call}

    actual_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStart": {"start": {"toolUse": {"name": "calculator", "toolUseId": "calc_123"}}}}

    assert actual_chunk == exp_chunk


def test_format_chunk_content_delta_text(model):
    event = {"chunk_type": "content_delta", "data_type": "text", "data": "Hello"}

    actual_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockDelta": {"delta": {"text": "Hello"}}}

    assert actual_chunk == exp_chunk


def test_format_chunk_content_delta_tool(model):
    event = {
        "chunk_type": "content_delta",
        "data_type": "tool",
        "data": '{"expression": "2+2"}',
    }

    actual_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"expression": "2+2"}'}}}}

    assert actual_chunk == exp_chunk


def test_format_chunk_content_stop(model):
    event = {"chunk_type": "content_stop"}

    actual_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStop": {}}

    assert actual_chunk == exp_chunk


def test_format_chunk_message_stop_end_turn(model):
    event = {"chunk_type": "message_stop", "data": "stop"}

    actual_chunk = model.format_chunk(event)
    exp_chunk = {"messageStop": {"stopReason": "end_turn"}}

    assert actual_chunk == exp_chunk


def test_format_chunk_message_stop_tool_use(model):
    event = {"chunk_type": "message_stop", "data": "tool_calls"}

    actual_chunk = model.format_chunk(event)
    exp_chunk = {"messageStop": {"stopReason": "tool_use"}}

    assert actual_chunk == exp_chunk


def test_format_chunk_message_stop_max_tokens(model):
    event = {"chunk_type": "message_stop", "data": "length"}

    actual_chunk = model.format_chunk(event)
    exp_chunk = {"messageStop": {"stopReason": "max_tokens"}}

    assert actual_chunk == exp_chunk


def test_format_chunk_metadata(model):
    mock_usage = unittest.mock.Mock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150

    event = {
        "chunk_type": "metadata",
        "data": mock_usage,
        "latency_ms": 250,
    }

    actual_chunk = model.format_chunk(event)
    exp_chunk = {
        "metadata": {
            "usage": {
                "inputTokens": 100,
                "outputTokens": 50,
                "totalTokens": 150,
            },
            "metrics": {
                "latencyMs": 250,
            },
        },
    }

    assert actual_chunk == exp_chunk


def test_format_chunk_metadata_no_latency(model):
    mock_usage = unittest.mock.Mock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150

    event = {
        "chunk_type": "metadata",
        "data": mock_usage,
    }

    actual_chunk = model.format_chunk(event)
    exp_chunk = {
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
    }

    assert actual_chunk == exp_chunk


def test_format_chunk_unknown(model):
    event = {"chunk_type": "unknown"}

    with pytest.raises(RuntimeError, match="chunk_type=<unknown> | unknown type"):
        model.format_chunk(event)


@pytest.mark.asyncio
async def test_stream(mistral_client, model, agenerator, alist, captured_warnings):
    mock_usage = unittest.mock.Mock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150

    mock_event = unittest.mock.Mock(
        data=unittest.mock.Mock(
            choices=[
                unittest.mock.Mock(
                    delta=unittest.mock.Mock(content="test stream", tool_calls=None),
                    finish_reason="end_turn",
                )
            ]
        ),
        usage=mock_usage,
    )

    mistral_client.chat.stream_async = unittest.mock.AsyncMock(return_value=agenerator([mock_event]))

    messages = [{"role": "user", "content": [{"text": "test"}]}]
    response = model.stream(messages, None, None)

    # Consume the response
    await alist(response)

    expected_request = {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 100,
        "stream": True,
    }

    mistral_client.chat.stream_async.assert_called_once_with(**expected_request)

    assert len(captured_warnings) == 0


@pytest.mark.asyncio
async def test_tool_choice_not_supported_warns(mistral_client, model, agenerator, alist, captured_warnings):
    tool_choice = {"auto": {}}

    mock_usage = unittest.mock.Mock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150

    mock_event = unittest.mock.Mock(
        data=unittest.mock.Mock(
            choices=[
                unittest.mock.Mock(
                    delta=unittest.mock.Mock(content="test stream", tool_calls=None),
                    finish_reason="end_turn",
                )
            ]
        ),
        usage=mock_usage,
    )

    mistral_client.chat.stream_async = unittest.mock.AsyncMock(return_value=agenerator([mock_event]))

    messages = [{"role": "user", "content": [{"text": "test"}]}]
    response = model.stream(messages, None, None, tool_choice=tool_choice)

    # Consume the response
    await alist(response)

    assert len(captured_warnings) == 1
    assert "ToolChoice was provided to this provider but is not supported" in str(captured_warnings[0].message)


@pytest.mark.asyncio
async def test_stream_rate_limit_error(mistral_client, model, alist):
    mistral_client.chat.stream_async.side_effect = Exception("rate limit exceeded (429)")

    messages = [{"role": "user", "content": [{"text": "test"}]}]
    with pytest.raises(ModelThrottledException, match="rate limit exceeded"):
        await alist(model.stream(messages))


@pytest.mark.asyncio
async def test_stream_other_error(mistral_client, model, alist):
    mistral_client.chat.stream_async.side_effect = Exception("some other error")

    messages = [{"role": "user", "content": [{"text": "test"}]}]
    with pytest.raises(Exception, match="some other error"):
        await alist(model.stream(messages))


@pytest.mark.asyncio
async def test_structured_output_success(mistral_client, model, test_output_model_cls, alist):
    messages = [{"role": "user", "content": [{"text": "Extract data"}]}]

    mock_response = unittest.mock.Mock()
    mock_response.choices = [unittest.mock.Mock()]
    mock_response.choices[0].message.tool_calls = [unittest.mock.Mock()]
    mock_response.choices[0].message.tool_calls[0].function.arguments = '{"name": "John", "age": 30}'

    mistral_client.chat.complete_async = unittest.mock.AsyncMock(return_value=mock_response)

    stream = model.structured_output(test_output_model_cls, messages)
    events = await alist(stream)

    tru_result = events[-1]
    exp_result = {"output": test_output_model_cls(name="John", age=30)}
    assert tru_result == exp_result


@pytest.mark.asyncio
async def test_structured_output_no_tool_calls(mistral_client, model, test_output_model_cls):
    mock_response = unittest.mock.Mock()
    mock_response.choices = [unittest.mock.Mock()]
    mock_response.choices[0].message.tool_calls = None

    mistral_client.chat.complete_async = unittest.mock.AsyncMock(return_value=mock_response)

    prompt = [{"role": "user", "content": [{"text": "Extract data"}]}]

    with pytest.raises(ValueError, match="No tool calls found in response"):
        stream = model.structured_output(test_output_model_cls, prompt)
        await anext(stream)


@pytest.mark.asyncio
async def test_structured_output_invalid_json(mistral_client, model, test_output_model_cls):
    mock_response = unittest.mock.Mock()
    mock_response.choices = [unittest.mock.Mock()]
    mock_response.choices[0].message.tool_calls = [unittest.mock.Mock()]
    mock_response.choices[0].message.tool_calls[0].function.arguments = "invalid json"

    mistral_client.chat.complete_async = unittest.mock.AsyncMock(return_value=mock_response)

    prompt = [{"role": "user", "content": [{"text": "Extract data"}]}]

    with pytest.raises(ValueError, match="Failed to parse tool call arguments into model"):
        stream = model.structured_output(test_output_model_cls, prompt)
        await anext(stream)


def test_config_validation_warns_on_unknown_keys(mistral_client, captured_warnings):
    """Test that unknown config keys emit a warning."""
    MistralModel(model_id="test-model", max_tokens=100, invalid_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "invalid_param" in str(captured_warnings[0].message)


def test_update_config_validation_warns_on_unknown_keys(model, captured_warnings):
    """Test that update_config warns on unknown keys."""
    model.update_config(wrong_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "wrong_param" in str(captured_warnings[0].message)
