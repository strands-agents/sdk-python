import logging
import unittest.mock

import openai
import pydantic
import pytest

import strands
from strands.models.openai import OpenAIModel
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


def test__init__(model_id):
    model = OpenAIModel(model_id=model_id, params={"max_tokens": 1})

    tru_config = model.get_config()
    exp_config = {"model_id": "m1", "params": {"max_tokens": 1}}

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


def test_format_request_tool_message_single_text_returns_string():
    """Test that single text content is returned as string for model compatibility."""
    tool_result = {
        "content": [{"text": '{"result": "success"}'}],
        "status": "success",
        "toolUseId": "c1",
    }

    tru_result = OpenAIModel.format_request_tool_message(tool_result)
    exp_result = {
        "content": '{"result": "success"}',
        "role": "tool",
        "tool_call_id": "c1",
    }
    assert tru_result == exp_result


def test_split_tool_message_images_with_image():
    """Test that images are extracted from tool messages."""
    tool_message = {
        "role": "tool",
        "tool_call_id": "c1",
        "content": [
            {"type": "text", "text": "Result"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo=", "detail": "auto", "format": "image/png"},
            },
        ],
    }

    tool_clean, user_with_image = OpenAIModel._split_tool_message_images(tool_message)

    # Tool message should now have the original text plus the appended informational text
    assert tool_clean["role"] == "tool"
    assert tool_clean["tool_call_id"] == "c1"
    assert len(tool_clean["content"]) == 2
    assert tool_clean["content"][0]["type"] == "text"
    assert tool_clean["content"][0]["text"] == "Result"
    assert "Tool successfully returned an image" in tool_clean["content"][1]["text"]

    # User message should have the image
    assert user_with_image is not None
    assert user_with_image["role"] == "user"
    assert len(user_with_image["content"]) == 1
    assert user_with_image["content"][0]["type"] == "image_url"


def test_split_tool_message_images_without_image():
    """Test that tool messages without images are unchanged."""
    tool_message = {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "Result"}]}

    tool_clean, user_with_image = OpenAIModel._split_tool_message_images(tool_message)

    assert tool_clean == tool_message
    assert user_with_image is None


def test_split_tool_message_images_only_image():
    """Test tool message with only image content."""
    tool_message = {
        "role": "tool",
        "tool_call_id": "c1",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}],
    }

    tool_clean, user_with_image = OpenAIModel._split_tool_message_images(tool_message)

    # Tool message should have default text
    assert tool_clean["role"] == "tool"
    assert len(tool_clean["content"]) == 1
    assert "successfully" in tool_clean["content"][0]["text"].lower()

    # User message should have the image
    assert user_with_image is not None
    assert user_with_image["role"] == "user"
    assert len(user_with_image["content"]) == 1


def test_split_tool_message_images_non_tool_role():
    """Test that messages with roles other than 'tool' are ignored."""
    user_msg = {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    clean, extra = OpenAIModel._split_tool_message_images(user_msg)
    assert clean == user_msg
    assert extra is None


def test_split_tool_message_images_invalid_content_type():
    """Test that messages with non-list content are ignored."""
    invalid_msg = {"role": "tool", "content": "not a list"}
    clean, extra = OpenAIModel._split_tool_message_images(invalid_msg)
    assert clean == invalid_msg
    assert extra is None


def test_format_request_messages_with_tool_result_containing_image():
    """Test that tool results with images are properly split into tool and user messages."""
    messages = [
        {
            "content": [{"text": "Run the tool"}],
            "role": "user",
        },
        {
            "content": [
                {
                    "toolUse": {
                        "input": {},
                        "name": "image_tool",
                        "toolUseId": "t1",
                    },
                },
            ],
            "role": "assistant",
        },
        {
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "t1",
                        "status": "success",
                        "content": [
                            {"text": "Image generated"},
                            {
                                "image": {
                                    "format": "png",
                                    "source": {"bytes": b"fake_image_data"},
                                }
                            },
                        ],
                    }
                }
            ],
            "role": "user",
        },
    ]

    formatted = OpenAIModel.format_request_messages(messages)

    # Find the tool message
    tool_messages = [msg for msg in formatted if msg.get("role") == "tool"]
    assert len(tool_messages) == 1

    # Tool message should only have text content
    tool_msg = tool_messages[0]
    assert all(c.get("type") != "image_url" for c in tool_msg["content"])

    # There should be a user message right after the tool message with the image
    tool_msg_idx = formatted.index(tool_msg)
    assert tool_msg_idx + 1 < len(formatted)
    user_msg = formatted[tool_msg_idx + 1]
    assert user_msg["role"] == "user"
    assert any(c.get("type") == "image_url" for c in user_msg["content"])


def test_format_request_messages_with_multiple_images_in_tool_result():
    """Test tool result with multiple images."""
    messages = [
        {
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "t1",
                        "status": "success",
                        "content": [
                            {"text": "Two images generated"},
                            {
                                "image": {
                                    "format": "png",
                                    "source": {"bytes": b"image1"},
                                }
                            },
                            {
                                "image": {
                                    "format": "jpg",
                                    "source": {"bytes": b"image2"},
                                }
                            },
                        ],
                    }
                }
            ],
            "role": "user",
        },
    ]

    formatted = OpenAIModel.format_request_messages(messages)

    # Find user message with images
    user_image_msgs = [
        msg
        for msg in formatted
        if msg.get("role") == "user" and any(c.get("type") == "image_url" for c in msg.get("content", []))
    ]
    assert len(user_image_msgs) == 1

    # Should have both images
    image_contents = [c for c in user_image_msgs[0]["content"] if c.get("type") == "image_url"]
    assert len(image_contents) == 2


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
            "content": "4",
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
        {"contentBlockStart": {"start": {}}},  # reasoning_content starts
        {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "\nI'm thinking"}}}},
        {"contentBlockStop": {}},  # reasoning_content ends
        {"contentBlockStart": {"start": {}}},  # text starts
        {"contentBlockDelta": {"delta": {"text": "I'll calculate"}}},
        {"contentBlockDelta": {"delta": {"text": "that for you"}}},
        {"contentBlockStop": {}},  # text ends
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
async def test_stream_respects_streaming_flag(openai_client, model_id, alist):
    """Test that the streaming flag in config is respected."""
    # Model configured to NOT stream
    model = OpenAIModel(client_args={}, model_id=model_id, params={"max_tokens": 1}, streaming=False)

    # Mock a non-streaming response object
    mock_choice = unittest.mock.Mock()
    mock_choice.finish_reason = "stop"
    mock_choice.message = unittest.mock.Mock()
    mock_choice.message.content = "non-stream result"
    mock_choice.message.tool_calls = None
    mock_response = unittest.mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = unittest.mock.Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)

    openai_client.chat.completions.create = unittest.mock.AsyncMock(return_value=mock_response)

    # Consume the generator and verify the events
    response_gen = model.stream([{"role": "user", "content": [{"text": "hi"}]}])
    tru_events = await alist(response_gen)

    expected_request = {
        "max_tokens": 1,
        "model": model_id,
        "messages": [{"role": "user", "content": [{"text": "hi", "type": "text"}]}],
        "stream": False,
        "tools": [],
    }
    openai_client.chat.completions.create.assert_called_once_with(**expected_request)

    exp_events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"start": {}}},
        {"contentBlockDelta": {"delta": {"text": "non-stream result"}}},
        {"contentBlockStop": {}},
        {"messageStop": {"stopReason": "end_turn"}},
        {
            "metadata": {
                "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30},
                "metrics": {"latencyMs": 0},
            }
        },
    ]
    assert tru_events == exp_events


@pytest.mark.asyncio
async def test_stream_non_streaming_tool_call(openai_client, model_id, alist):
    """Test that tool calls work correctly in non-streaming mode (Fix for #778)."""
    # Mock a tool call in a non-streaming response
    mock_tool_call = unittest.mock.Mock()
    mock_tool_call.id = "t1"
    mock_tool_call.function.name = "get_weather"
    mock_tool_call.function.arguments = '{"location": "London"}'

    mock_choice = unittest.mock.Mock()
    mock_choice.message.content = None
    mock_choice.message.tool_calls = [mock_tool_call]
    mock_choice.finish_reason = "tool_calls"

    mock_response = unittest.mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    openai_client.chat.completions.create = unittest.mock.AsyncMock(return_value=mock_response)

    model = OpenAIModel(model_id=model_id, streaming=False)
    messages = [{"role": "user", "content": [{"text": "What is the weather?"}]}]

    events = await alist(model.stream(messages))

    # Verify tool use events were emitted
    tool_starts = [e for e in events if "contentBlockStart" in e and "toolUse" in e["contentBlockStart"]["start"]]
    assert len(tool_starts) == 1
    assert tool_starts[0]["contentBlockStart"]["start"]["toolUse"]["name"] == "get_weather"

    tool_deltas = [e for e in events if "contentBlockDelta" in e and "toolUse" in e["contentBlockDelta"]["delta"]]
    assert len(tool_deltas) == 1
    assert tool_deltas[0]["contentBlockDelta"]["delta"]["toolUse"]["input"] == '{"location": "London"}'

    # Verify stop reason
    stop_event = next(e for e in events if "messageStop" in e)
    assert stop_event["messageStop"]["stopReason"] == "tool_use"


@pytest.mark.asyncio
async def test_stream_non_streaming_multi_block_response(openai_client, model_id, alist):
    """Test non-streaming response with multiple content blocks (text and reasoning)."""
    # Mock a response with a list of blocks
    mock_choice = unittest.mock.Mock()
    mock_choice.message.content = [
        {"reasoningContent": {"reasoningText": {"text": "I will check the weather."}}},
        {"text": "Checking weather now..."},
        "Additional string block",
    ]
    mock_choice.message.tool_calls = None
    mock_choice.finish_reason = "stop"

    mock_response = unittest.mock.Mock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    openai_client.chat.completions.create = unittest.mock.AsyncMock(return_value=mock_response)

    model = OpenAIModel(model_id=model_id, streaming=False)
    events = await alist(model.stream([{"role": "user", "content": [{"text": "hi"}]}]))

    # Verify both types of content were captured
    event_contents = [e["contentBlockDelta"]["delta"] for e in events if "contentBlockDelta" in e]

    assert any(
        "reasoningContent" in c and c["reasoningContent"]["text"] == "I will check the weather." for c in event_contents
    )
    assert any("text" in c and c["text"] == "Checking weather now..." for c in event_contents)
    assert any("text" in c and c["text"] == "Additional string block" for c in event_contents)


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
        {"messageStop": {"stopReason": "end_turn"}},  # No content blocks when no content
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
        {"contentBlockStart": {"start": {}}},  # text content starts
        {"contentBlockDelta": {"delta": {"text": "content"}}},
        {"contentBlockDelta": {"delta": {"text": "content"}}},
        {"contentBlockStop": {}},  # text content ends
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


@pytest.mark.parametrize(
    "new_data_type, prev_data_type, expected_chunks, expected_data_type",
    [
        ("text", None, [{"contentBlockStart": {"start": {}}}], "text"),
        (
            "reasoning_content",
            "text",
            [{"contentBlockStop": {}}, {"contentBlockStart": {"start": {}}}],
            "reasoning_content",
        ),
        ("text", "text", [], "text"),
    ],
)
def test__stream_switch_content(model, new_data_type, prev_data_type, expected_chunks, expected_data_type):
    """Test _stream_switch_content method for content type switching."""
    chunks, data_type = model._stream_switch_content(new_data_type, prev_data_type)
    assert chunks == expected_chunks
    assert data_type == expected_data_type


def test_format_request_messages_excludes_reasoning_content():
    """Test that reasoningContent is excluded from formatted messages."""
    messages = [
        {
            "content": [
                {"text": "Hello"},
                {"reasoningContent": {"reasoningText": {"text": "excluded"}}},
            ],
            "role": "user",
        },
    ]

    tru_result = OpenAIModel.format_request_messages(messages)

    # Only text content should be included
    exp_result = [
        {
            "content": [{"text": "Hello", "type": "text"}],
            "role": "user",
        },
    ]
    assert tru_result == exp_result


@pytest.mark.asyncio
async def test_structured_output_context_overflow_exception(openai_client, model, messages, test_output_model_cls):
    """Test that structured output also handles context overflow properly."""
    # Create a mock OpenAI BadRequestError with context_length_exceeded code
    mock_error = openai.BadRequestError(
        message="This model's maximum context length is 4096 tokens. However, your messages resulted in 5000 tokens.",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "context_length_exceeded"}},
    )
    mock_error.code = "context_length_exceeded"

    # Configure the mock client to raise the context overflow error
    openai_client.beta.chat.completions.parse.side_effect = mock_error

    # Test that the structured_output method converts the error properly
    with pytest.raises(ContextWindowOverflowException) as exc_info:
        async for _ in model.structured_output(test_output_model_cls, messages):
            pass

    # Verify the exception message contains the original error
    assert "maximum context length" in str(exc_info.value)
    assert exc_info.value.__cause__ == mock_error


@pytest.mark.asyncio
async def test_stream_context_overflow_exception(openai_client, model, messages):
    """Test that OpenAI context overflow errors are properly converted to ContextWindowOverflowException."""
    # Create a mock OpenAI BadRequestError with context_length_exceeded code
    mock_error = openai.BadRequestError(
        message="This model's maximum context length is 4096 tokens. However, your messages resulted in 5000 tokens.",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "context_length_exceeded"}},
    )
    mock_error.code = "context_length_exceeded"

    # Configure the mock client to raise the context overflow error
    openai_client.chat.completions.create.side_effect = mock_error

    # Test that the stream method converts the error properly
    with pytest.raises(ContextWindowOverflowException) as exc_info:
        async for _ in model.stream(messages):
            pass

    # Verify the exception message contains the original error
    assert "maximum context length" in str(exc_info.value)
    assert exc_info.value.__cause__ == mock_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_message",
    [
        "Input is too long for requested model",
        "input length and `max_tokens` exceed context limit",
        "too many total text bytes",
    ],
)
async def test_stream_alternative_context_overflow_messages(openai_client, model, messages, error_message):
    """Test that alternative context overflow messages in APIError are properly converted."""
    # Create a mock OpenAI APIError with alternative context overflow message
    mock_error = openai.APIError(
        message=error_message,
        request=unittest.mock.MagicMock(),
        body={"error": {"message": error_message}},
    )

    # Configure the mock client to raise the APIError
    openai_client.chat.completions.create.side_effect = mock_error

    # Test that the stream method converts the error properly
    with pytest.raises(ContextWindowOverflowException) as exc_info:
        async for _ in model.stream(messages):
            pass

    # Verify the exception message contains the original error
    assert error_message in str(exc_info.value)
    assert exc_info.value.__cause__ == mock_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_message",
    [
        "Input is too long for requested model",
        "input length and `max_tokens` exceed context limit",
        "too many total text bytes",
    ],
)
async def test_structured_output_alternative_context_overflow_messages(
    openai_client, model, messages, test_output_model_cls, error_message
):
    """Test that alternative context overflow messages in APIError are properly converted in structured output."""
    # Create a mock OpenAI APIError with alternative context overflow message
    mock_error = openai.APIError(
        message=error_message,
        request=unittest.mock.MagicMock(),
        body={"error": {"message": error_message}},
    )

    # Configure the mock client to raise the APIError
    openai_client.beta.chat.completions.parse.side_effect = mock_error

    # Test that the structured_output method converts the error properly
    with pytest.raises(ContextWindowOverflowException) as exc_info:
        async for _ in model.structured_output(test_output_model_cls, messages):
            pass

    # Verify the exception message contains the original error
    assert error_message in str(exc_info.value)
    assert exc_info.value.__cause__ == mock_error


@pytest.mark.asyncio
async def test_stream_api_error_passthrough(openai_client, model, messages):
    """Test that APIError without overflow messages passes through unchanged."""
    # Create a mock OpenAI APIError without overflow message
    mock_error = openai.APIError(
        message="Some other API error",
        request=unittest.mock.MagicMock(),
        body={"error": {"message": "Some other API error"}},
    )

    # Configure the mock client to raise the APIError
    openai_client.chat.completions.create.side_effect = mock_error

    # Test that APIError without overflow messages passes through
    with pytest.raises(openai.APIError) as exc_info:
        async for _ in model.stream(messages):
            pass

    # Verify the original exception is raised, not ContextWindowOverflowException
    assert exc_info.value == mock_error


@pytest.mark.asyncio
async def test_stream_other_bad_request_errors_passthrough(openai_client, model, messages):
    """Test that other BadRequestError exceptions are not converted to ContextWindowOverflowException."""
    # Create a mock OpenAI BadRequestError with a different error code
    mock_error = openai.BadRequestError(
        message="Invalid parameter value",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "invalid_parameter"}},
    )
    mock_error.code = "invalid_parameter"

    # Configure the mock client to raise the non-context error
    openai_client.chat.completions.create.side_effect = mock_error

    # Test that other BadRequestError exceptions pass through unchanged
    with pytest.raises(openai.BadRequestError) as exc_info:
        async for _ in model.stream(messages):
            pass

    # Verify the original exception is raised, not ContextWindowOverflowException
    assert exc_info.value == mock_error


@pytest.mark.asyncio
async def test_stream_rate_limit_as_throttle(openai_client, model, messages):
    """Test that all rate limit errors are converted to ModelThrottledException."""

    # Create a mock OpenAI RateLimitError (any type of rate limit)
    mock_error = openai.RateLimitError(
        message="Request too large for gpt-4o on tokens per min (TPM): Limit 30000, Requested 117505.",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "rate_limit_exceeded"}},
    )
    mock_error.code = "rate_limit_exceeded"

    # Configure the mock client to raise the rate limit error
    openai_client.chat.completions.create.side_effect = mock_error

    # Test that the stream method converts the error properly
    with pytest.raises(ModelThrottledException) as exc_info:
        async for _ in model.stream(messages):
            pass

    # Verify the exception message contains the original error
    assert "tokens per min" in str(exc_info.value)
    assert exc_info.value.__cause__ == mock_error


@pytest.mark.asyncio
async def test_stream_request_rate_limit_as_throttle(openai_client, model, messages):
    """Test that request-based rate limit errors are converted to ModelThrottledException."""

    # Create a mock OpenAI RateLimitError for request-based rate limiting
    mock_error = openai.RateLimitError(
        message="Rate limit reached for requests per minute.",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "rate_limit_exceeded"}},
    )
    mock_error.code = "rate_limit_exceeded"

    # Configure the mock client to raise the request rate limit error
    openai_client.chat.completions.create.side_effect = mock_error

    # Test that the stream method converts the error properly
    with pytest.raises(ModelThrottledException) as exc_info:
        async for _ in model.stream(messages):
            pass

    # Verify the exception message contains the original error
    assert "Rate limit reached" in str(exc_info.value)
    assert exc_info.value.__cause__ == mock_error


@pytest.mark.asyncio
async def test_structured_output_rate_limit_as_throttle(openai_client, model, messages, test_output_model_cls):
    """Test that structured output handles rate limit errors properly."""

    # Create a mock OpenAI RateLimitError
    mock_error = openai.RateLimitError(
        message="Request too large for gpt-4o on tokens per min (TPM): Limit 30000, Requested 117505.",
        response=unittest.mock.MagicMock(),
        body={"error": {"code": "rate_limit_exceeded"}},
    )
    mock_error.code = "rate_limit_exceeded"

    # Configure the mock client to raise the rate limit error
    openai_client.beta.chat.completions.parse.side_effect = mock_error

    # Test that the structured_output method converts the error properly
    with pytest.raises(ModelThrottledException) as exc_info:
        async for _ in model.structured_output(test_output_model_cls, messages):
            pass

    # Verify the exception message contains the original error
    assert "tokens per min" in str(exc_info.value)
    assert exc_info.value.__cause__ == mock_error


def test_format_request_messages_with_system_prompt_content():
    """Test format_request_messages with system_prompt_content parameter."""
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    system_prompt_content = [{"text": "You are a helpful assistant."}]

    result = OpenAIModel.format_request_messages(messages, system_prompt_content=system_prompt_content)

    expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"text": "Hello", "type": "text"}]},
    ]

    assert result == expected


def test_format_request_messages_with_none_system_prompt_content():
    """Test format_request_messages with system_prompt_content parameter."""
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]

    result = OpenAIModel.format_request_messages(messages)

    expected = [{"role": "user", "content": [{"text": "Hello", "type": "text"}]}]

    assert result == expected


def test_format_request_messages_drops_cache_points():
    """Test that cache points are dropped in OpenAI format_request_messages."""
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    system_prompt_content = [{"text": "You are a helpful assistant."}, {"cachePoint": {"type": "default"}}]

    result = OpenAIModel.format_request_messages(messages, system_prompt_content=system_prompt_content)

    # Cache points should be dropped, only text content included
    expected = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"text": "Hello", "type": "text"}]},
    ]

    assert result == expected


@pytest.mark.asyncio
async def test_stream_with_injected_client(model_id, agenerator, alist):
    """Test that stream works with an injected client and doesn't close it."""
    # Create a mock injected client
    mock_injected_client = unittest.mock.AsyncMock()
    mock_injected_client.close = unittest.mock.AsyncMock()

    mock_delta = unittest.mock.Mock(content="Hello", tool_calls=None, reasoning_content=None)
    mock_event_1 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason=None, delta=mock_delta)])
    mock_event_2 = unittest.mock.Mock(choices=[unittest.mock.Mock(finish_reason="stop", delta=mock_delta)])
    mock_event_3 = unittest.mock.Mock()

    mock_injected_client.chat.completions.create = unittest.mock.AsyncMock(
        return_value=agenerator([mock_event_1, mock_event_2, mock_event_3])
    )

    # Create model with injected client
    model = OpenAIModel(client=mock_injected_client, model_id=model_id, params={"max_tokens": 1})

    messages = [{"role": "user", "content": [{"text": "test"}]}]
    response = model.stream(messages)
    tru_events = await alist(response)

    # Verify events were generated
    assert len(tru_events) > 0

    # Verify the injected client was used
    mock_injected_client.chat.completions.create.assert_called_once()

    # Verify the injected client was NOT closed
    mock_injected_client.close.assert_not_called()


@pytest.mark.asyncio
async def test_structured_output_with_injected_client(model_id, test_output_model_cls, alist):
    """Test that structured_output works with an injected client and doesn't close it."""
    # Create a mock injected client
    mock_injected_client = unittest.mock.AsyncMock()
    mock_injected_client.close = unittest.mock.AsyncMock()

    mock_parsed_instance = test_output_model_cls(name="John", age=30)
    mock_choice = unittest.mock.Mock()
    mock_choice.message.parsed = mock_parsed_instance
    mock_response = unittest.mock.Mock()
    mock_response.choices = [mock_choice]

    mock_injected_client.beta.chat.completions.parse = unittest.mock.AsyncMock(return_value=mock_response)

    # Create model with injected client
    model = OpenAIModel(client=mock_injected_client, model_id=model_id, params={"max_tokens": 1})

    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]
    stream = model.structured_output(test_output_model_cls, messages)
    events = await alist(stream)

    # Verify output was generated
    assert len(events) == 1
    assert events[0] == {"output": test_output_model_cls(name="John", age=30)}

    # Verify the injected client was used
    mock_injected_client.beta.chat.completions.parse.assert_called_once()

    # Verify the injected client was NOT closed
    mock_injected_client.close.assert_not_called()


def test_init_with_both_client_and_client_args_raises_error():
    """Test that providing both client and client_args raises ValueError."""
    mock_client = unittest.mock.AsyncMock()

    with pytest.raises(ValueError, match="Only one of 'client' or 'client_args' should be provided"):
        OpenAIModel(client=mock_client, client_args={"api_key": "test"}, model_id="test-model")


def test_format_request_filters_s3_source_image(model, caplog):
    """Test that images with Location sources are filtered out with warning."""
    caplog.set_level(logging.WARNING, logger="strands.models.openai")

    messages = [
        {
            "role": "user",
            "content": [
                {"text": "look at this image"},
                {
                    "image": {
                        "format": "png",
                        "source": {"location": {"type": "s3", "uri": "s3://my-bucket/image.png"}},
                    },
                },
            ],
        },
    ]

    request = model.format_request(messages)

    # Image with S3 source should be filtered, text should remain
    formatted_content = request["messages"][0]["content"]
    assert len(formatted_content) == 1
    assert formatted_content[0]["type"] == "text"
    assert "Location sources are not supported by OpenAI" in caplog.text


def test_format_request_filters_location_source_document(model, caplog):
    """Test that documents with Location sources are filtered out with warning."""
    caplog.set_level(logging.WARNING, logger="strands.models.openai")

    messages = [
        {
            "role": "user",
            "content": [
                {"text": "analyze this document"},
                {
                    "document": {
                        "format": "pdf",
                        "name": "report.pdf",
                        "source": {"location": {"type": "s3", "uri": "s3://my-bucket/report.pdf"}},
                    },
                },
                {
                    "document": {
                        "format": "pdf",
                        "name": "report.pdf",
                        "source": {"location": {"type": "s3", "uri": "s3://my-bucket/report.pdf"}},
                    },
                },
            ],
        },
    ]

    request = model.format_request(messages)

    # Document with S3 source should be filtered, text should remain
    formatted_content = request["messages"][0]["content"]
    assert len(formatted_content) == 1
    assert formatted_content[0]["type"] == "text"
    assert "Location sources are not supported by OpenAI" in caplog.text


def test_format_request_messages_with_tool_calls_no_content():
    """Test that assistant messages with only tool calls are included and have no content field."""
    messages = [
        {"role": "user", "content": [{"text": "Use the calculator"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "input": {"expression": "2+2"},
                        "name": "calculator",
                        "toolUseId": "c1",
                    },
                },
            ],
        },
    ]

    tru_result = OpenAIModel.format_request_messages(messages)

    exp_result = [
        {"role": "user", "content": [{"text": "Use the calculator", "type": "text"}]},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {"arguments": '{"expression": "2+2"}', "name": "calculator"},
                    "id": "c1",
                    "type": "function",
                }
            ],
        },
    ]
    assert tru_result == exp_result


def test_format_request_messages_multiple_tool_calls_with_images():
    """Test that multiple tool calls with image results are formatted correctly.

    OpenAI requires all tool response messages to immediately follow the assistant
    message with tool_calls, before any other messages. When tools return images,
    the images are moved to user messages, but these must come after ALL tool messages.
    """
    messages = [
        {"role": "user", "content": [{"text": "Run the tools"}]},
        {
            "role": "assistant",
            "content": [
                {"toolUse": {"input": {}, "name": "tool1", "toolUseId": "call_1"}},
                {"toolUse": {"input": {}, "name": "tool2", "toolUseId": "call_2"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "call_1",
                        "content": [{"image": {"format": "png", "source": {"bytes": b"img1"}}}],
                        "status": "success",
                    }
                },
                {
                    "toolResult": {
                        "toolUseId": "call_2",
                        "content": [{"image": {"format": "png", "source": {"bytes": b"img2"}}}],
                        "status": "success",
                    }
                },
            ],
        },
    ]

    tru_result = OpenAIModel.format_request_messages(messages)

    image_placeholder = (
        "Tool successfully returned an image. The image is being provided in the following user message."
    )
    exp_result = [
        {"role": "user", "content": [{"text": "Run the tools", "type": "text"}]},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"arguments": "{}", "name": "tool1"}, "id": "call_1", "type": "function"},
                {"function": {"arguments": "{}", "name": "tool2"}, "id": "call_2", "type": "function"},
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": image_placeholder}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": [{"type": "text", "text": image_placeholder}],
        },
        {
            "role": "user",
            "content": [
                {
                    "image_url": {"detail": "auto", "format": "image/png", "url": "data:image/png;base64,aW1nMQ=="},
                    "type": "image_url",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "image_url": {"detail": "auto", "format": "image/png", "url": "data:image/png;base64,aW1nMg=="},
                    "type": "image_url",
                }
            ],
        },
    ]
    assert tru_result == exp_result
