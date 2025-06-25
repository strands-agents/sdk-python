import unittest.mock

import pytest

import strands
from strands.models.mistral import MistralModel
from strands.types.exceptions import ModelThrottledException


@pytest.fixture
def mistral_client():
    with unittest.mock.patch.object(strands.models.mistral, "Mistral") as mock_client_cls:
        yield mock_client_cls.return_value


@pytest.fixture
def model_id():
    return "mistral-large-latest"


@pytest.fixture
def max_tokens():
    return 100


@pytest.fixture
def model(mistral_client, model_id, max_tokens):
    _ = mistral_client

    return MistralModel(model_id=model_id, max_tokens=max_tokens)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt():
    return "You are a helpful assistant"


def test__init__model_configs(mistral_client, model_id, max_tokens):
    _ = mistral_client

    model = MistralModel(model_id=model_id, max_tokens=max_tokens, temperature=0.7)

    tru_temperature = model.get_config().get("temperature")
    exp_temperature = 0.7

    assert tru_temperature == exp_temperature


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


def test_format_request_default(model, messages, model_id):
    tru_request = model.format_request(messages)
    exp_request = {
        "model": model_id,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 100,
    }

    assert tru_request == exp_request


def test_format_request_with_temperature(model, messages, model_id):
    model.update_config(temperature=0.8)

    tru_request = model.format_request(messages)
    exp_request = {
        "model": model_id,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 100,
        "temperature": 0.8,
    }

    assert tru_request == exp_request


def test_format_request_with_system_prompt(model, messages, model_id, system_prompt):
    tru_request = model.format_request(messages, system_prompt=system_prompt)
    exp_request = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "test"},
        ],
        "max_tokens": 100,
    }

    assert tru_request == exp_request


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

    tru_request = model.format_request(messages)
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
    }

    assert tru_request == exp_request


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

    tru_request = model.format_request(messages)
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
    }

    assert tru_request == exp_request


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

    tru_request = model.format_request(messages, tool_specs)
    exp_request = {
        "model": model_id,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 100,
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

    assert tru_request == exp_request


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

    tru_request = model.format_request(messages, tool_specs)
    exp_request = {
        "model": model_id,
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.9,
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

    assert tru_request == exp_request


def test_format_chunk_message_start(model):
    event = {"chunk_type": "message_start"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStart": {"role": "assistant"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_start_text(model):
    event = {"chunk_type": "content_start", "data_type": "text"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStart": {"start": {}}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_start_tool(model):
    mock_tool_call = unittest.mock.Mock()
    mock_tool_call.function.name = "calculator"
    mock_tool_call.id = "calc_123"

    event = {"chunk_type": "content_start", "data_type": "tool", "data": mock_tool_call}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStart": {"start": {"toolUse": {"name": "calculator", "toolUseId": "calc_123"}}}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_delta_text(model):
    event = {"chunk_type": "content_delta", "data_type": "text", "data": "Hello"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockDelta": {"delta": {"text": "Hello"}}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_delta_tool(model):
    event = {
        "chunk_type": "content_delta",
        "data_type": "tool",
        "data": '{"expression": "2+2"}',
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"expression": "2+2"}'}}}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_stop(model):
    event = {"chunk_type": "content_stop"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStop": {}}

    assert tru_chunk == exp_chunk


def test_format_chunk_message_stop_end_turn(model):
    event = {"chunk_type": "message_stop", "data": "stop"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStop": {"stopReason": "end_turn"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_message_stop_tool_use(model):
    event = {"chunk_type": "message_stop", "data": "tool_calls"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStop": {"stopReason": "tool_use"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_message_stop_max_tokens(model):
    event = {"chunk_type": "message_stop", "data": "length"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStop": {"stopReason": "max_tokens"}}

    assert tru_chunk == exp_chunk


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

    tru_chunk = model.format_chunk(event)
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

    assert tru_chunk == exp_chunk


def test_format_chunk_metadata_no_latency(model):
    mock_usage = unittest.mock.Mock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150

    event = {
        "chunk_type": "metadata",
        "data": mock_usage,
    }

    tru_chunk = model.format_chunk(event)
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

    assert tru_chunk == exp_chunk


def test_format_chunk_unknown(model):
    event = {"chunk_type": "unknown"}

    with pytest.raises(RuntimeError, match="chunk_type=<unknown> | unknown type"):
        model.format_chunk(event)


def test_stream_rate_limit_error(mistral_client, model):
    mistral_client.chat.stream.side_effect = Exception("rate limit exceeded (429)")

    with pytest.raises(ModelThrottledException, match="rate limit exceeded"):
        list(model.stream({}))


def test_stream_other_error(mistral_client, model):
    mistral_client.chat.stream.side_effect = Exception("some other error")

    with pytest.raises(Exception, match="some other error"):
        list(model.stream({}))


def test_structured_output_success(mistral_client, model):
    from pydantic import BaseModel

    class TestModel(BaseModel):
        name: str
        age: int

    # Mock successful response
    mock_response = unittest.mock.Mock()
    mock_response.choices = [unittest.mock.Mock()]
    mock_response.choices[0].message.tool_calls = [unittest.mock.Mock()]
    mock_response.choices[0].message.tool_calls[0].function.arguments = '{"name": "John", "age": 30}'

    mistral_client.chat.complete.return_value = mock_response

    prompt = [{"role": "user", "content": [{"text": "Extract data"}]}]
    result = model.structured_output(TestModel, prompt)

    assert isinstance(result, TestModel)
    assert result.name == "John"
    assert result.age == 30


def test_structured_output_no_tool_calls(mistral_client, model):
    from pydantic import BaseModel

    class TestModel(BaseModel):
        name: str

    mock_response = unittest.mock.Mock()
    mock_response.choices = [unittest.mock.Mock()]
    mock_response.choices[0].message.tool_calls = None

    mistral_client.chat.complete.return_value = mock_response

    prompt = [{"role": "user", "content": [{"text": "Extract data"}]}]

    with pytest.raises(ValueError, match="No tool calls found in response"):
        model.structured_output(TestModel, prompt)


def test_structured_output_invalid_json(mistral_client, model):
    from pydantic import BaseModel

    class TestModel(BaseModel):
        name: str

    mock_response = unittest.mock.Mock()
    mock_response.choices = [unittest.mock.Mock()]
    mock_response.choices[0].message.tool_calls = [unittest.mock.Mock()]
    mock_response.choices[0].message.tool_calls[0].function.arguments = "invalid json"

    mistral_client.chat.complete.return_value = mock_response

    prompt = [{"role": "user", "content": [{"text": "Extract data"}]}]

    with pytest.raises(ValueError, match="Failed to parse tool call arguments into model"):
        model.structured_output(TestModel, prompt)
