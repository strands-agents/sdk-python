import json
import unittest.mock

import pytest

import strands
from strands.models.ollama import OllamaModel
from strands.types.content import Messages


@pytest.fixture
def ollama_client():
    with unittest.mock.patch.object(strands.models.ollama, "OllamaClient") as mock_client_cls:
        yield mock_client_cls.return_value


@pytest.fixture
def model_id():
    return "m1"


@pytest.fixture
def host():
    return "h1"


@pytest.fixture
def model(ollama_client, model_id, host):
    _ = ollama_client

    return OllamaModel(host, model_id=model_id)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt():
    return "s1"


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
            "role": "tool",
            "content": [{"toolResult": {"toolUseId": "calculator", "status": "success", "content": [{"text": "4"}]}}],
        }
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "name": "calculator",
                        "result": "4",
                        "status": "success",
                    }
                ),
            }
        ],
        "model": model_id,
        "options": {},
        "stream": True,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_tool_result_json(model, model_id):
    messages: Messages = [
        {
            "role": "tool",
            "content": [
                {"toolResult": {"toolUseId": "calculator", "status": "success", "content": [{"json": {"result": 4}}]}}
            ],
        }
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "name": "calculator",
                        "result": {"result": 4},
                        "status": "success",
                    }
                ),
            }
        ],
        "model": model_id,
        "options": {},
        "stream": True,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_tool_result_image(model, model_id):
    messages: Messages = [
        {
            "role": "tool",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "image_generator",
                        "status": "success",
                        "content": [{"image": {"source": {"bytes": "base64encodedimage"}}}],
                    }
                }
            ],
        }
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "name": "image_generator",
                        "result": "see images",
                        "status": "success",
                    }
                ),
                "images": ["base64encodedimage"],
            }
        ],
        "model": model_id,
        "options": {},
        "stream": True,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_tool_result_other(model, model_id):
    messages = [
        {
            "role": "tool",
            "content": [{"toolResult": {"toolUseId": "other", "status": "success", "content": {"other": {"a": 1}}}}],
        }
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "name": "other",
                        "result": {"other": {"a": 1}},
                        "status": "success",
                    }
                ),
            }
        ],
        "model": model_id,
        "options": {},
        "stream": True,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_other(model, model_id):
    messages = [
        {
            "role": "user",
            "content": [{"other": {"a": 1}}],
        }
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "user",
                "content": json.dumps({"other": {"a": 1}}),
            }
        ],
        "model": model_id,
        "options": {},
        "stream": True,
        "tools": [],
    }

    assert tru_request == exp_request


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


def test_stream(ollama_client, model):
    mock_event = unittest.mock.Mock()
    mock_event.message.tool_calls = None
    mock_event.message.content = "Hello"
    mock_event.done_reason = "stop"

    ollama_client.chat.return_value = [mock_event]

    request = {"model": "m1", "messages": [{"role": "user", "content": "Hello"}]}
    response = model.stream(request)

    tru_events = list(response)
    exp_events = [
        {"chunk_type": "message_start"},
        {"chunk_type": "content_start", "data_type": "text"},
        {"chunk_type": "content_delta", "data_type": "text", "data": "Hello"},
        {"chunk_type": "content_stop", "data_type": "text"},
        {"chunk_type": "message_stop", "data": "stop"},
        {"chunk_type": "metadata", "data": mock_event},
    ]

    assert tru_events == exp_events
    ollama_client.chat.assert_called_once_with(**request)


def test_stream_with_tool_calls(ollama_client, model):
    mock_event = unittest.mock.Mock()
    mock_tool_call = unittest.mock.Mock()
    mock_event.message.tool_calls = [mock_tool_call]
    mock_event.message.content = "I'll calculate that for you"
    mock_event.done_reason = "stop"

    ollama_client.chat.return_value = [mock_event]

    request = {"model": "m1", "messages": [{"role": "user", "content": "Calculate 2+2"}]}
    response = model.stream(request)

    tru_events = list(response)
    exp_events = [
        {"chunk_type": "message_start"},
        {"chunk_type": "content_start", "data_type": "text"},
        {"chunk_type": "content_start", "data_type": "tool", "data": mock_tool_call},
        {"chunk_type": "content_delta", "data_type": "tool", "data": mock_tool_call},
        {"chunk_type": "content_stop", "data_type": "tool", "data": mock_tool_call},
        {"chunk_type": "content_delta", "data_type": "text", "data": "I'll calculate that for you"},
        {"chunk_type": "content_stop", "data_type": "text"},
        {"chunk_type": "message_stop", "data": "tool_use"},
        {"chunk_type": "metadata", "data": mock_event},
    ]

    assert tru_events == exp_events
    ollama_client.chat.assert_called_once_with(**request)
