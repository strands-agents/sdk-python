import unittest.mock

import pydantic
import pytest

import strands
from strands.models.openai import OpenAIModel


@pytest.fixture
def openai_client_cls():
    with unittest.mock.patch.object(strands.models.openai.openai, "AsyncOpenAI") as mock_client_cls:
        yield mock_client_cls


@pytest.fixture
def openai_client(openai_client_cls):
    return openai_client_cls.return_value


@pytest.fixture
def model_id():
    return "m1"


@pytest.fixture
def model(openai_client, model_id):
    _ = openai_client

    return OpenAIModel(model_id=model_id, params={"max_tokens": 1})


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


def test__init__(openai_client_cls, model_id):
    model = OpenAIModel({"api_key": "k1"}, model_id=model_id, params={"max_tokens": 1})

    tru_config = model.get_config()
    exp_config = {"model_id": "m1", "params": {"max_tokens": 1}}

    assert tru_config == exp_config

    openai_client_cls.assert_called_once_with(api_key="k1")


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
                "file": {
                    "file_data": "data:application/pdf;base64,ZG9jdW1lbnQ=",
                    "filename": "test doc",
                },
                "type": "file",
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
                "image_url": {
                    "detail": "auto",
                    "format": "image/jpeg",
                    "url": "data:image/jpeg;base64,aW1hZ2U=",
                },
                "type": "image_url",
            },
        ),
        # Text
        (
            {"text": "hello"},
            {"type": "text", "text": "hello"},
        ),
    ],
)
def test_format_request_message_content(content, exp_result):
    tru_result = OpenAIModel.format_request_message_content(content)
    assert tru_result == exp_result


def test_format_request_message_content_unsupported_type():
    content = {"unsupported": {}}

    with pytest.raises(TypeError, match="content_type=<unsupported> | unsupported type"):
        OpenAIModel.format_request_message_content(content)


def test_format_request_message_tool_call():
    tool_use = {
        "input": {"expression": "2+2"},
        "name": "calculator",
        "toolUseId": "c1",
    }

    tru_result = OpenAIModel.format_request_message_tool_call(tool_use)
    exp_result = {
        "function": {
            "arguments": '{"expression": "2+2"}',
            "name": "calculator",
        },
        "id": "c1",
        "type": "function",
    }
    assert tru_result == exp_result


def test_format_request_tool_message():
    tool_result = {
        "content": [{"text": "4"}, {"json": ["4"]}],
        "status": "success",
        "toolUseId": "c1",
    }

    tru_result = OpenAIModel.format_request_tool_message(tool_result)
    exp_result = {
        "content": [{"text": "4", "type": "text"}, {"text": '["4"]', "type": "text"}],
        "role": "tool",
        "tool_call_id": "c1",
    }
    assert tru_result == exp_result


def test_format_request_tool_choice_auto():
    tool_choice = {"auto": {}}

    tru_result = OpenAIModel._format_request_tool_choice(tool_choice)
    exp_result = {"tool_choice": "auto"}
    assert tru_result == exp_result


def test_format_request_tool_choice_any():
    tool_choice = {"any": {}}

    tru_result = OpenAIModel._format_request_tool_choice(tool_choice)
    exp_result = {"tool_choice": "required"}
    assert tru_result == exp_result


def test_format_request_tool_choice_tool():
    tool_choice = {"tool": {"name": "test_tool"}}

    tru_result = OpenAIModel._format_request_tool_choice(tool_choice)
    exp_result = {"tool_choice": {"type": "function", "function": {"name": "test_tool"}}}
    assert tru_result == exp_result


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

    tru_result = OpenAIModel.format_request_messages(messages, system_prompt)
    exp_result = [
        {
            "content": system_prompt,
            "role": "system",
        },
        {
            "content": [{"text": "hello", "type": "text"}],
            "role": "user",
        },
        {
            "content": [{"text": "call tool", "type": "text"}],
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
        },
        {
            "content": [{"text": "4", "type": "text"}],
            "role": "tool",
            "tool_call_id": "c1",
        },
    ]
    assert tru_result == exp_result


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
        "model": "m1",
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


def test_format_request_with_tool_choice_auto(model, messages, tool_specs, system_prompt):
    tool_choice = {"auto": {}}
    tru_request = model.format_request(messages, tool_specs, system_prompt, tool_choice)
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
        "model": "m1",
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
        "tool_choice": "auto",
        "max_tokens": 1,
    }
    assert tru_request == exp_request


def test_format_request_with_tool_choice_any(model, messages, tool_specs, system_prompt):
    tool_choice = {"any": {}}
    tru_request = model.format_request(messages, tool_specs, system_prompt, tool_choice)
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
        "model": "m1",
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
        "tool_choice": "required",
        "max_tokens": 1,
    }
    assert tru_request == exp_request


def test_format_request_with_tool_choice_tool(model, messages, tool_specs, system_prompt):
    tool_choice = {"tool": {"name": "test_tool"}}
    tru_request = model.format_request(messages, tool_specs, system_prompt, tool_choice)
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
        "model": "m1",
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
        "tool_choice": {"type": "function", "function": {"name": "test_tool"}},
        "max_tokens": 1,
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
    tru_chunk = model.format_chunk(event)
    assert tru_chunk == exp_chunk


def test_format_chunk_unknown_type(model):
    event = {"chunk_type": "unknown"}

    with pytest.raises(RuntimeError, match="chunk_type=<unknown> | unknown type"):
        model.format_chunk(event)


@pytest.mark.asyncio
async def test_stream(openai_client, model_id, model, agenerator, alist):
    mock_tool_call_1_part_1 = unittest.mock.Mock(index=0)
    mock_tool_call_2_part_1 = unittest.mock.Mock(index=1)
    mock_delta_1 = unittest.mock.Mock(
        reasoning_content="",
        content=None,
        tool_calls=None,
    )
    mock_delta_2 = unittest.mock.Mock(
        reasoning_content="\nI'm thinking",
        content=None,
        tool_calls=None,
    )
    mock_delta_3 = unittest.mock.Mock(
        content="I'll calculate", tool_calls=[mock_tool_call_1_part_1, mock_tool_call_2_part_1], reasoning_content=None
    )

    mock_tool_call_1_part_2 = unittest.mock.Mock(index=0)
    mock_tool_call_2_part_2 = unittest.mock.Mock(index=1)
    mock_delta_4 = unittest.mock.Mock(
        content="that for you", tool_calls=[mock_tool_call_1_part_2, mock_tool_call_2_part_2], reasoning_content=None
    )

    mock_delta_5 = unittest.mock.Mock(content="", tool_calls=None, reasoning_content=None)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_1)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_2)])
    mock_event_3 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_3)])
    mock_event_4 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta_4)])
    mock_event_5 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="tool_calls", delta=mock_delta_5)])
    mock_event_6 = unittest.mock.Mock()

    openai_client.chat.completions.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_event_1, mock_event_2, mock_event_3, mock_event_4, mock_event_5, mock_event_6])
    )

    messages = [{"role": "user", "content": [{"text": "calculate 2+2"}]}]
    response = model.stream(messages)
    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "\nI'm thinking"}}}},
        {"contentBlockDelta": {"delta": {"text": "I'll calculate"}}},
        {"contentBlockDelta": {"delta": {"text": "that for you"}}},
        {"contentBlockStop": {}},
        {
            "contentBlockStart": {
                "start": {
                    "toolUse": {"toolUseId": mock_tool_call_1_part_1.id, "name": mock_tool_call_1_part_1.function.name}
                }
            }
        },
        {"contentBlockDelta": {"delta": {"toolUse": {"input": mock_tool_call_1_part_1.function.arguments}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": mock_tool_call_1_part_2.function.arguments}}}},
        {"contentBlockStop": {}},
        {
            "contentBlockStart": {
                "start": {
                    "toolUse": {"toolUseId": mock_tool_call_2_part_1.id, "name": mock_tool_call_2_part_1.function.name}
                }
            }
        },
        {"contentBlockDelta": {"delta": {"toolUse": {"input": mock_tool_call_2_part_1.function.arguments}}}},
        {"contentBlockDelta": {"delta": {"toolUse": {"input": mock_tool_call_2_part_2.function.arguments}}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "tool_use"}},
        {
            "metadata": {
                "usage": {
                    "inputTokens": mock_event_6.usage.prompt_tokens,
                    "outputTokens": mock_event_6.usage.completion_tokens,
                    "totalTokens": mock_event_6.usage.total_tokens,
                },
                "metrics": {"latencyMs": 0},
            }
        },
    ]

    assert len(tru_events) == len(exp_events)
    # Verify that format_request was called with the correct arguments
    expected_request = {
        "max_tokens": 1,
        "model": model_id,
        "messages": [{"role": "user", "content": [{"text": "calculate 2+2", "type": "text"}]}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [],
    }
    openai_client.chat.completions.create.assert_called_once_with(**expected_request)


@pytest.mark.asyncio
async def test_stream_empty(openai_client, model_id, model, agenerator, alist):
    mock_delta = unittest.mock.Mock(content=None, tool_calls=None, reasoning_content=None)

    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_event_3 = unittest.mock.Mock()
    mock_event_4 = unittest.mock.Mock(usage=None)

    openai_client.chat.completions.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_event_1, mock_event_2, mock_event_3, mock_event_4]),
    )

    messages = [{"role": "user", "content": []}]
    response = model.stream(messages)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
    ]

    assert len(tru_events) == len(exp_events)
    expected_request = {
        "max_tokens": 1,
        "model": model_id,
        "messages": [],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [],
    }
    openai_client.chat.completions.create.assert_called_once_with(**expected_request)


@pytest.mark.asyncio
async def test_stream_with_empty_choices(openai_client, model, agenerator, alist):
    mock_delta = unittest.mock.Mock(content="content", tool_calls=None, reasoning_content=None)
    mock_usage = unittest.mock.Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    # Event with no choices attribute
    mock_event_1 = unittest.mock.Mock(spec=[])

    # Event with empty choices list
    mock_event_2 = unittest.mock.Mock(choices=[])

    # Valid event with content
    mock_event_3 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])

    # Event with finish reason
    mock_event_4 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])

    # Final event with usage info
    mock_event_5 = unittest.mock.Mock(usage=mock_usage)

    openai_client.chat.completions.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_event_1, mock_event_2, mock_event_3, mock_event_4, mock_event_5])
    )

    messages = [{"role": "user", "content": [{"text": "test"}]}]
    response = model.stream(messages)

    tru_events = await alist(response)
    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": "content"}}},
        {"contentBlockDelta": {"delta": {"text": "content"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
                "metrics": {"latencyMs": 0},
            }
        },
    ]

    assert len(tru_events) == len(exp_events)
    expected_request = {
        "max_tokens": 1,
        "model": "m1",
        "messages": [{"role": "user", "content": [{"text": "test", "type": "text"}]}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": [],
    }
    openai_client.chat.completions.create.assert_called_once_with(**expected_request)


@pytest.mark.asyncio
async def test_structured_output(openai_client, model, test_output_model_cls, alist):
    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    mock_parsed_instance = test_output_model_cls(name="John", age=30)
    mock_choice = unittest.mock.Mock()
    mock_choice.message.parsed = mock_parsed_instance
    mock_response = unittest.mock.Mock()
    mock_response.choices = [mock_choice]

    openai_client.beta.chat.completions.parse = unittest.mock.AsyncMock(return_value=mock_response)

    stream = model.structured_output(test_output_model_cls, messages)
    events = await alist(stream)

    tru_result = events[-1]
    exp_result = {"output": test_output_model_cls(name="John", age=30)}
    assert tru_result == exp_result


def test_config_validation_warns_on_unknown_keys(openai_client, captured_warnings):
    """Test that unknown config keys emit a warning."""
    OpenAIModel({"api_key": "test"}, model_id="test-model", invalid_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "invalid_param" in str(captured_warnings[0].message)


def test_update_config_validation_warns_on_unknown_keys(model, captured_warnings):
    """Test that update_config warns on unknown keys."""
    model.update_config(wrong_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "wrong_param" in str(captured_warnings[0].message)


def test_tool_choice_supported_no_warning(model, messages, captured_warnings):
    """Test that toolChoice doesn't emit warning for supported providers."""
    tool_choice = {"auto": {}}
    model.format_request(messages, tool_choice=tool_choice)

    assert len(captured_warnings) == 0


def test_tool_choice_none_no_warning(model, messages, captured_warnings):
    """Test that None toolChoice doesn't emit warning."""
    model.format_request(messages, tool_choice=None)

    assert len(captured_warnings) == 0
