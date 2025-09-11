import json
import unittest.mock

import pydantic
import pytest

import strands
from strands.models.ollama import OllamaModel
from strands.types.content import Messages


@pytest.fixture
def ollama_client():
    with unittest.mock.patch.object(strands.models.ollama.ollama, "AsyncClient") as mock_client_cls:
        yield mock_client_cls.return_value


@pytest.fixture
def model_id():
    return "m1"


@pytest.fixture
def host():
    return "h1"


@pytest.fixture
def model(model_id, host):
    return OllamaModel(host, model_id=model_id)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt():
    return "s1"


@pytest.fixture
def test_output_model_cls():
    class TestOutputModel(pydantic.BaseModel):
        name: str
        age: int

    return TestOutputModel


def test__init__model_configs(ollama_client, model_id, host):
    _ = ollama_client

    model = OllamaModel(host, model_id=model_id, max_tokens=1)

    tru_max_tokens = model.get_config().get("max_tokens")
    exp_max_tokens = 1

    assert tru_max_tokens == exp_max_tokens


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


def test_format_request_default(model, messages, model_id):
    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "options": {},
        "stream": True,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_override(model, messages, model_id):
    model.update_config(model_id=model_id)
    tru_request = model.format_request(messages, tool_specs=None)
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "options": {},
        "stream": True,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_system_prompt(model, messages, model_id, system_prompt):
    tru_request = model.format_request(messages, system_prompt=system_prompt)
    exp_request = {
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": "test"}],
        "model": model_id,
        "options": {},
        "stream": True,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_image(model, model_id):
    messages = [{"role": "user", "content": [{"image": {"source": {"bytes": "base64encodedimage"}}}]}]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "images": ["base64encodedimage"]}],
        "model": model_id,
        "options": {},
        "stream": True,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_tool_use(model, model_id):
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "calculator", "input": '{"expression": "2+2"}'}}]}
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "2+2"}',
                        }
                    }
                ],
            }
        ],
        "model": model_id,
        "options": {},
        "stream": True,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_tool_result(model, model_id):
    messages: Messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "calculator",
                        "status": "success",
                        "content": [
                            {"text": "4"},
                            {"image": {"source": {"bytes": b"image"}}},
                            {"json": ["4"]},
                        ],
                    },
                },
                {
                    "text": "see results",
                },
            ],
        },
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "tool",
                "content": "4",
            },
            {
                "role": "tool",
                "images": [b"image"],
            },
            {
                "role": "tool",
                "content": '["4"]',
            },
            {
                "role": "user",
                "content": "see results",
            },
        ],
        "model": model_id,
        "options": {},
        "stream": True,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_unsupported_type(model):
    messages = [
        {
            "role": "user",
            "content": [{"unsupported": {}}],
        },
    ]

    with pytest.raises(TypeError, match="content_type=<unsupported> | unsupported type"):
        model.format_request(messages)


def test_format_request_with_tool_specs(model, messages, model_id):
    tool_specs = [
        {
            "name": "calculator",
            "description": "Calculate mathematical expressions",
            "inputSchema": {
                "json": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
            },
        }
    ]

    tru_request = model.format_request(messages, tool_specs)
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "options": {},
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

    assert tru_request == exp_request


def test_format_request_with_inference_config(model, messages, model_id):
    inference_config = {
        "max_tokens": 1,
        "stop_sequences": ["stop"],
        "temperature": 1,
        "top_p": 1,
    }

    model.update_config(**inference_config)
    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "options": {
            "num_predict": inference_config["max_tokens"],
            "temperature": inference_config["temperature"],
            "top_p": inference_config["top_p"],
            "stop": inference_config["stop_sequences"],
        },
        "stream": True,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_options(model, messages, model_id):
    options = {"o1": 1}

    model.update_config(options=options)
    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "content": "test"}],
        "model": model_id,
        "options": {"o1": 1},
        "stream": True,
        "tools": [],
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
    mock_function = unittest.mock.Mock()
    mock_function.function.name = "calculator"

    event = {"chunk_type": "content_start", "data_type": "tool", "data": mock_function}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStart": {"start": {"toolUse": {"name": "calculator", "toolUseId": "calculator"}}}}

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
        "data": unittest.mock.Mock(function=unittest.mock.Mock(arguments={"expression": "2+2"})),
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps({"expression": "2+2"})}}}}

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
    event = {"chunk_type": "message_stop", "data": "tool_use"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStop": {"stopReason": "tool_use"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_message_stop_length(model):
    event = {"chunk_type": "message_stop", "data": "length"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStop": {"stopReason": "max_tokens"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_metadata(model):
    event = {
        "chunk_type": "metadata",
        "data": unittest.mock.Mock(eval_count=100, prompt_eval_count=50, total_duration=1000000),
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
                "latencyMs": 1.0,
            },
        },
    }

    assert tru_chunk == exp_chunk


def test_format_chunk_other(model):
    event = {"chunk_type": "other"}

    with pytest.raises(RuntimeError, match="chunk_type=<other> | unknown type"):
        model.format_chunk(event)


@pytest.mark.asyncio
async def test_stream(ollama_client, model, agenerator, alist, captured_warnings):
    mock_event = unittest.mock.Mock()
    mock_event.message.tool_calls = None
    mock_event.message.content = "Hello"
    mock_event.done_reason = "stop"
    mock_event.eval_count = 10
    mock_event.prompt_eval_count = 5
    mock_event.total_duration = 1000000  # 1ms in nanoseconds

    ollama_client.chat = unittest.mock.AsyncMock(return_value=agenerator([mock_event]))

    messages = [{"role": "user", "content": [{"text": "Hello"}]}]
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
                "metrics": {"latencyMs": 1.0},
            }
        },
    ]

    assert tru_events == exp_events
    expected_request = {
        "model": "m1",
        "messages": [{"role": "user", "content": "Hello"}],
        "options": {},
        "stream": True,
        "tools": [],
    }
    ollama_client.chat.assert_called_once_with(**expected_request)

    # Ensure no warnings emitted
    assert len(captured_warnings) == 0


@pytest.mark.asyncio
async def test_tool_choice_not_supported_warns(ollama_client, model, agenerator, alist, captured_warnings):
    """Test that non-None toolChoice emits warning for unsupported providers."""
    tool_choice = {"auto": {}}

    mock_event = unittest.mock.Mock()
    mock_event.message.tool_calls = None
    mock_event.message.content = "Hello"
    mock_event.done_reason = "stop"
    mock_event.eval_count = 10
    mock_event.prompt_eval_count = 5
    mock_event.total_duration = 1000000  # 1ms in nanoseconds

    ollama_client.chat = unittest.mock.AsyncMock(return_value=agenerator([mock_event]))

    messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    await alist(model.stream(messages, tool_choice=tool_choice))

    assert len(captured_warnings) == 1
    assert "ToolChoice was provided to this provider but is not supported" in str(captured_warnings[0].message)


@pytest.mark.asyncio
async def test_stream_with_tool_calls(ollama_client, model, agenerator, alist):
    mock_event = unittest.mock.Mock()
    mock_tool_call = unittest.mock.Mock()
    mock_tool_call.function.name = "calculator"
    mock_tool_call.function.arguments = {"expression": "2+2"}
    mock_event.message.tool_calls = [mock_tool_call]
    mock_event.message.content = "I'll calculate that for you"
    mock_event.done_reason = "stop"
    mock_event.eval_count = 15
    mock_event.prompt_eval_count = 8
    mock_event.total_duration = 2000000  # 2ms in nanoseconds

    ollama_client.chat = unittest.mock.AsyncMock(return_value=agenerator([mock_event]))

    messages = [{"role": "user", "content": [{"text": "Calculate 2+2"}]}]
    response = model.stream(messages)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockStart": {"start": {"toolUse": {"name": "calculator", "toolUseId": "calculator"}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"expression": "2+2"}'}}}},
        {"contentBlockStop": {}},
        {"contentBlockDelta": {"delta": {"text": "I'll calculate that for you"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
        {
            "metadata": {
                "usage": {"inputTokens": 15, "outputTokens": 8, "totalTokens": 23},
                "metrics": {"latencyMs": 2.0},
            }
        },
    ]

    assert tru_events == exp_events
    expected_request = {
        "model": "m1",
        "messages": [{"role": "user", "content": "Calculate 2+2"}],
        "options": {},
        "stream": True,
        "tools": [],
    }
    ollama_client.chat.assert_called_once_with(**expected_request)


@pytest.mark.asyncio
async def test_structured_output(ollama_client, model, test_output_model_cls, alist):
    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    mock_response = unittest.mock.Mock()
    mock_response.message.content = '{"name": "John", "age": 30}'

    ollama_client.chat = unittest.mock.AsyncMock(return_value=mock_response)

    stream = model.structured_output(test_output_model_cls, messages)
    events = await alist(stream)

    tru_result = events[-1]
    exp_result = {"output": test_output_model_cls(name="John", age=30)}
    assert tru_result == exp_result


def test_config_validation_warns_on_unknown_keys(ollama_client, captured_warnings):
    """Test that unknown config keys emit a warning."""
    OllamaModel("http://localhost:11434", model_id="test-model", invalid_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "invalid_param" in str(captured_warnings[0].message)


def test_update_config_validation_warns_on_unknown_keys(model, captured_warnings):
    """Test that update_config warns on unknown keys."""
    model.update_config(wrong_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "wrong_param" in str(captured_warnings[0].message)
