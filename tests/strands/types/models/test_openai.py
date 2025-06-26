import unittest.mock

import pytest

from strands.types.models import OpenAIModel as SAOpenAIModel


class TestOpenAIModel(SAOpenAIModel):
    def __init__(self):
        self.config = {"model_id": "m1", "params": {"max_tokens": 1}}

    def update_config(self, **model_config):
        return model_config

    def get_config(self):
        return

    def stream(self, request):
        yield {"request": request}


@pytest.fixture
def model():
    return TestOpenAIModel()


@pytest.fixture
def messages():
    return [
        {
            "role": "user",
            "content": [{"text": "hello"}],
        },
    ]


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
                    "url": "data:image/jpeg;base64,image",
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
    tru_result = SAOpenAIModel.format_request_message_content(content)
    assert tru_result == exp_result


def test_format_request_message_tool_call():
    tool_use = {
        "input": {"expression": "2+2"},
        "name": "calculator",
        "toolUseId": "c1",
    }

    tru_result = SAOpenAIModel.format_request_message_tool_call(tool_use)
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

    tru_result = SAOpenAIModel.format_request_tool_message(tool_result)
    exp_result = {
        "content": [{"text": "4", "type": "text"}, {"text": '["4"]', "type": "text"}],
        "role": "tool",
        "tool_call_id": "c1",
    }
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

    tru_result = SAOpenAIModel.format_request_messages(messages, system_prompt)
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
                "content": [{"text": "hello", "type": "text"}],
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
