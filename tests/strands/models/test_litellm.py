import json
import unittest.mock

import pytest
import strands
from strands.models.litellm import LiteLLMModel


@pytest.fixture
def litellm_client():
    with unittest.mock.patch.object(strands.models.litellm.litellm, "LiteLLM") as mock_client_cls:
        yield mock_client_cls.return_value


@pytest.fixture
def model_id():
    return "m1"


@pytest.fixture
def model(litellm_client, model_id):
    _ = litellm_client

    return LiteLLMModel(model_id=model_id)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt():
    return "s1"


def test__init__model_configs(litellm_client, model_id):
    _ = litellm_client

    model = LiteLLMModel(model_id=model_id, params={"max_tokens": 1})

    tru_max_tokens = model.get_config().get("params")
    exp_max_tokens = {"max_tokens": 1}

    assert tru_max_tokens == exp_max_tokens


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


def test_format_request_default(model, messages, model_id):
    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}],
        "model": model_id,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_params(model, messages, model_id):
    model.update_config(params={"max_tokens": 1})

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}],
        "model": model_id,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
        "tools": [],
        "max_tokens": 1,
    }

    assert tru_request == exp_request


def test_format_request_with_system_prompt(model, messages, model_id, system_prompt):
    tru_request = model.format_request(messages, system_prompt=system_prompt)
    exp_request = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "text", "text": "test"}]},
        ],
        "model": model_id,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_image(model, model_id):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": "jpg",
                        "source": {"bytes": b"base64encodedimage"},
                    },
                },
            ],
        },
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "image_url": {
                            "detail": "auto",
                            "format": "image/jpeg",
                            "url": "data:image/jpeg;base64,base64encodedimage",
                        },
                        "type": "image_url",
                    },
                ],
            },
        ],
        "model": model_id,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_reasoning(model, model_id):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "reasoningContent": {
                        "reasoningText": {
                            "signature": "reasoning_signature",
                            "text": "reasoning_text",
                        },
                    },
                },
            ],
        },
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "signature": "reasoning_signature",
                        "thinking": "reasoning_text",
                        "type": "thinking",
                    },
                ],
            },
        ],
        "model": model_id,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_video(model, model_id):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "video": {
                        "source": {"bytes": "base64encodedvideo"},
                    },
                },
            ],
        },
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "detail": "auto",
                            "url": "base64encodedvideo",
                        },
                    },
                ],
            },
        ],
        "model": model_id,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_other(model, model_id):
    messages = [
        {
            "role": "user",
            "content": [{"other": {"a": 1}}],
        },
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "text": json.dumps({"other": {"a": 1}}),
                        "type": "text",
                    },
                ],
            },
        ],
        "model": model_id,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_tool_result(model, model_id):
    messages = [
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "c1", "status": "success", "content": [{"value": 4}]}}],
        }
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "content": json.dumps(
                    {
                        "content": [{"value": 4}],
                        "status": "success",
                    }
                ),
                "role": "tool",
                "tool_call_id": "c1",
            },
        ],
        "model": model_id,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_tool_use(model, model_id):
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
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "messages": [
            {
                "content": [],
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "2+2"}',
                        },
                        "id": "c1",
                        "type": "function",
                    }
                ],
            }
        ],
        "model": model_id,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
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
        "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}],
        "model": model_id,
        "stream": True,
        "stream_options": {
            "include_usage": True,
        },
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
    mock_tool_use = unittest.mock.Mock()
    mock_tool_use.function.name = "calculator"
    mock_tool_use.id = "c1"

    event = {"chunk_type": "content_start", "data_type": "tool", "data": mock_tool_use}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStart": {"start": {"toolUse": {"name": "calculator", "toolUseId": "c1"}}}}

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
        "data": unittest.mock.Mock(function=unittest.mock.Mock(arguments='{"expression": "2+2"}')),
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
    event = {
        "chunk_type": "metadata",
        "data": unittest.mock.Mock(prompt_tokens=100, completion_tokens=50, total_tokens=150),
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


def test_format_chunk_other(model):
    event = {"chunk_type": "other"}

    with pytest.raises(RuntimeError, match="chunk_type=<other> | unknown type"):
        model.format_chunk(event)


def test_stream(litellm_client, model):
    mock_tool_call_1_part_1 = unittest.mock.Mock(index=0)
    mock_tool_call_2_part_1 = unittest.mock.Mock(index=1)
    mock_delta_1 = unittest.mock.Mock(
        content="I'll calculate", tool_calls=[mock_tool_call_1_part_1, mock_tool_call_2_part_1]
    )

    mock_tool_call_1_part_2 = unittest.mock.Mock(index=0)
    mock_tool_call_2_part_2 = unittest.mock.Mock(index=1)
    mock_delta_2 = unittest.mock.Mock(
        content="that for you", tool_calls=[mock_tool_call_1_part_2, mock_tool_call_2_part_2]
    )

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_1)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_2)])
    mock_event_3 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="tool_calls")])
    mock_event_4 = unittest.mock.Mock()

    litellm_client.chat.completions.create.return_value = iter([mock_event_1, mock_event_2, mock_event_3, mock_event_4])

    request = {"model": "m1", "messages": [{"role": "user", "content": [{"type": "text", "text": "calculate 2+2"}]}]}
    response = model.stream(request)

    tru_events = list(response)
    exp_events = [
        {"chunk_type": "message_start"},
        {"chunk_type": "content_start", "data_type": "text"},
        {"chunk_type": "content_delta", "data_type": "text", "data": "I'll calculate"},
        {"chunk_type": "content_delta", "data_type": "text", "data": "that for you"},
        {"chunk_type": "content_stop", "data_type": "text"},
        {"chunk_type": "content_start", "data_type": "tool", "data": mock_tool_call_1_part_1},
        {"chunk_type": "content_delta", "data_type": "tool", "data": mock_tool_call_1_part_2},
        {"chunk_type": "content_stop", "data_type": "tool"},
        {"chunk_type": "content_start", "data_type": "tool", "data": mock_tool_call_2_part_1},
        {"chunk_type": "content_delta", "data_type": "tool", "data": mock_tool_call_2_part_2},
        {"chunk_type": "content_stop", "data_type": "tool"},
        {"chunk_type": "message_stop", "data": "tool_calls"},
        {"chunk_type": "metadata", "data": mock_event_4.usage},
    ]

    assert tru_events == exp_events
    litellm_client.chat.completions.create.assert_called_once_with(**request)


def test_stream_empty(litellm_client, model):
    mock_delta = unittest.mock.Mock(content=None, tool_calls=None)
    mock_usage = unittest.mock.Mock(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop")])
    mock_event_3 = unittest.mock.Mock()
    mock_event_4 = unittest.mock.Mock(usage=mock_usage)

    litellm_client.chat.completions.create.return_value = iter([mock_event_1, mock_event_2, mock_event_3, mock_event_4])

    request = {"model": "m1", "messages": [{"role": "user", "content": []}]}
    response = model.stream(request)

    tru_events = list(response)
    exp_events = [
        {"chunk_type": "message_start"},
        {"chunk_type": "content_start", "data_type": "text"},
        {"chunk_type": "content_stop", "data_type": "text"},
        {"chunk_type": "message_stop", "data": "stop"},
        {"chunk_type": "metadata", "data": mock_usage},
    ]

    assert tru_events == exp_events
    litellm_client.chat.completions.create.assert_called_once_with(**request)
