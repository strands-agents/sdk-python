import unittest.mock

import anthropic
import pydantic
import pytest

import strands
from strands.models.anthropic import AnthropicModel
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException


@pytest.fixture
def anthropic_client():
    with unittest.mock.patch.object(strands.models.anthropic.anthropic, "Anthropic") as mock_client_cls:
        yield mock_client_cls.return_value


@pytest.fixture
def model_id():
    return "m1"


@pytest.fixture
def max_tokens():
    return 1


@pytest.fixture
def model(anthropic_client, model_id, max_tokens):
    _ = anthropic_client

    return AnthropicModel(model_id=model_id, max_tokens=max_tokens)


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


def test__init__model_configs(anthropic_client, model_id, max_tokens):
    _ = anthropic_client

    model = AnthropicModel(model_id=model_id, max_tokens=max_tokens, params={"temperature": 1})

    tru_temperature = model.get_config().get("params")
    exp_temperature = {"temperature": 1}

    assert tru_temperature == exp_temperature


def test_update_config(model, model_id):
    model.update_config(model_id=model_id)

    tru_model_id = model.get_config().get("model_id")
    exp_model_id = model_id

    assert tru_model_id == exp_model_id


def test_format_request_default(model, messages, model_id, max_tokens):
    tru_request = model.format_request(messages)
    exp_request = {
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}],
        "model": model_id,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_params(model, messages, model_id, max_tokens):
    model.update_config(params={"temperature": 1})

    tru_request = model.format_request(messages)
    exp_request = {
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}],
        "model": model_id,
        "tools": [],
        "temperature": 1,
    }

    assert tru_request == exp_request


def test_format_request_with_system_prompt(model, messages, model_id, max_tokens, system_prompt):
    tru_request = model.format_request(messages, system_prompt=system_prompt)
    exp_request = {
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": [{"type": "text", "text": "test"}]}],
        "model": model_id,
        "system": system_prompt,
        "tools": [],
    }

    assert tru_request == exp_request


@pytest.mark.parametrize(
    ("content", "formatted_content"),
    [
        # PDF
        (
            {
                "document": {"format": "pdf", "name": "test doc", "source": {"bytes": b"pdf"}},
            },
            {
                "source": {
                    "data": "cGRm",
                    "media_type": "application/pdf",
                    "type": "base64",
                },
                "title": "test doc",
                "type": "document",
            },
        ),
        # Plain text
        (
            {
                "document": {"format": "txt", "name": "test doc", "source": {"bytes": b"txt"}},
            },
            {
                "source": {
                    "data": "txt",
                    "media_type": "text/plain",
                    "type": "text",
                },
                "title": "test doc",
                "type": "document",
            },
        ),
    ],
)
def test_format_request_with_document(content, formatted_content, model, model_id, max_tokens):
    messages = [
        {
            "role": "user",
            "content": [content],
        },
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [formatted_content],
            },
        ],
        "model": model_id,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_image(model, model_id, max_tokens):
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
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "source": {
                            "data": "YmFzZTY0ZW5jb2RlZGltYWdl",
                            "media_type": "image/jpeg",
                            "type": "base64",
                        },
                        "type": "image",
                    },
                ],
            },
        ],
        "model": model_id,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_reasoning(model, model_id, max_tokens):
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
        "max_tokens": max_tokens,
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
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_tool_use(model, model_id, max_tokens):
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
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "id": "c1",
                        "input": {"expression": "2+2"},
                        "name": "calculator",
                        "type": "tool_use",
                    },
                ],
            },
        ],
        "model": model_id,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_tool_results(model, model_id, max_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "c1",
                        "status": "success",
                        "content": [
                            {"text": "see image"},
                            {"json": ["see image"]},
                            {
                                "image": {
                                    "format": "jpg",
                                    "source": {"bytes": b"base64encodedimage"},
                                },
                            },
                        ],
                    }
                }
            ],
        }
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "content": [
                            {
                                "text": "see image",
                                "type": "text",
                            },
                            {
                                "text": '["see image"]',
                                "type": "text",
                            },
                            {
                                "source": {
                                    "data": "YmFzZTY0ZW5jb2RlZGltYWdl",
                                    "media_type": "image/jpeg",
                                    "type": "base64",
                                },
                                "type": "image",
                            },
                        ],
                        "is_error": False,
                        "tool_use_id": "c1",
                        "type": "tool_result",
                    },
                ],
            },
        ],
        "model": model_id,
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


def test_format_request_with_cache_point(model, model_id, max_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "cache me"},
                {"cachePoint": {"type": "default"}},
            ],
        },
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "cache_control": {"type": "ephemeral"},
                        "text": "cache me",
                        "type": "text",
                    },
                ],
            },
        ],
        "model": model_id,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_request_with_empty_content(model, model_id, max_tokens):
    messages = [
        {
            "role": "user",
            "content": [],
        },
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "max_tokens": max_tokens,
        "messages": [],
        "model": model_id,
        "tools": [],
    }

    assert tru_request == exp_request


def test_format_chunk_message_start(model):
    event = {"type": "message_start"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStart": {"role": "assistant"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_block_start_tool_use(model):
    event = {
        "content_block": {
            "id": "c1",
            "name": "calculator",
            "type": "tool_use",
        },
        "index": 0,
        "type": "content_block_start",
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {
        "contentBlockStart": {
            "contentBlockIndex": 0,
            "start": {"toolUse": {"name": "calculator", "toolUseId": "c1"}},
        },
    }

    assert tru_chunk == exp_chunk


def test_format_chunk_content_block_start_other(model):
    event = {
        "content_block": {
            "type": "text",
        },
        "index": 0,
        "type": "content_block_start",
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {
        "contentBlockStart": {
            "contentBlockIndex": 0,
            "start": {},
        },
    }

    assert tru_chunk == exp_chunk


def test_format_chunk_content_block_delta_signature_delta(model):
    event = {
        "delta": {
            "type": "signature_delta",
            "signature": "s1",
        },
        "index": 0,
        "type": "content_block_delta",
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {
                "reasoningContent": {
                    "signature": "s1",
                },
            },
        },
    }

    assert tru_chunk == exp_chunk


def test_format_chunk_content_block_delta_thinking_delta(model):
    event = {
        "delta": {
            "type": "thinking_delta",
            "thinking": "t1",
        },
        "index": 0,
        "type": "content_block_delta",
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {
                "reasoningContent": {
                    "text": "t1",
                },
            },
        },
    }

    assert tru_chunk == exp_chunk


def test_format_chunk_content_block_delta_input_json_delta_delta(model):
    event = {
        "delta": {
            "type": "input_json_delta",
            "partial_json": "{",
        },
        "index": 0,
        "type": "content_block_delta",
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {
                "toolUse": {
                    "input": "{",
                },
            },
        },
    }

    assert tru_chunk == exp_chunk


def test_format_chunk_content_block_delta_text_delta(model):
    event = {
        "delta": {
            "type": "text_delta",
            "text": "hello",
        },
        "index": 0,
        "type": "content_block_delta",
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {
        "contentBlockDelta": {
            "contentBlockIndex": 0,
            "delta": {"text": "hello"},
        },
    }

    assert tru_chunk == exp_chunk


def test_format_chunk_content_block_delta_unknown(model):
    event = {
        "delta": {
            "type": "unknown",
        },
        "type": "content_block_delta",
    }

    with pytest.raises(RuntimeError, match="chunk_type=<content_block_delta>, delta=<unknown> | unknown type"):
        model.format_chunk(event)


def test_format_chunk_content_block_stop(model):
    event = {"type": "content_block_stop", "index": 0}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStop": {"contentBlockIndex": 0}}

    assert tru_chunk == exp_chunk


def test_format_chunk_message_stop(model):
    event = {"type": "message_stop", "message": {"stop_reason": "end_turn"}}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStop": {"stopReason": "end_turn"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_metadata(model):
    event = {
        "type": "metadata",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {
        "metadata": {
            "usage": {
                "inputTokens": 1,
                "outputTokens": 2,
                "totalTokens": 3,
            },
            "metrics": {
                "latencyMs": 0,
            },
        },
    }

    assert tru_chunk == exp_chunk


def test_format_chunk_unknown(model):
    event = {"type": "unknown"}

    with pytest.raises(RuntimeError, match="chunk_type=<unknown> | unknown type"):
        model.format_chunk(event)


def test_stream(anthropic_client, model):
    mock_event_1 = unittest.mock.Mock(
        type="message_start",
        dict=lambda: {"type": "message_start"},
        model_dump=lambda: {"type": "message_start"},
    )
    mock_event_2 = unittest.mock.Mock(
        type="unknown",
        dict=lambda: {"type": "unknown"},
        model_dump=lambda: {"type": "unknown"},
    )
    mock_event_3 = unittest.mock.Mock(
        type="metadata",
        message=unittest.mock.Mock(
            usage=unittest.mock.Mock(
                dict=lambda: {"input_tokens": 1, "output_tokens": 2},
                model_dump=lambda: {"input_tokens": 1, "output_tokens": 2},
            )
        ),
    )

    mock_stream = unittest.mock.MagicMock()
    mock_stream.__iter__.return_value = iter([mock_event_1, mock_event_2, mock_event_3])
    anthropic_client.messages.stream.return_value.__enter__.return_value = mock_stream

    request = {"model": "m1"}
    response = model.stream(request)

    tru_events = list(response)
    exp_events = [
        {"type": "message_start"},
        {
            "type": "metadata",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        },
    ]

    assert tru_events == exp_events
    anthropic_client.messages.stream.assert_called_once_with(**request)


def test_stream_rate_limit_error(anthropic_client, model):
    anthropic_client.messages.stream.side_effect = anthropic.RateLimitError(
        "rate limit", response=unittest.mock.Mock(), body=None
    )

    with pytest.raises(ModelThrottledException, match="rate limit"):
        next(model.stream({}))


@pytest.mark.parametrize(
    "overflow_message",
    [
        "...input is too long...",
        "...input length exceeds context window...",
        "...input and output tokens exceed your context limit...",
    ],
)
def test_stream_bad_request_overflow_error(overflow_message, anthropic_client, model):
    anthropic_client.messages.stream.side_effect = anthropic.BadRequestError(
        overflow_message, response=unittest.mock.Mock(), body=None
    )

    with pytest.raises(ContextWindowOverflowException):
        next(model.stream({}))


def test_stream_bad_request_error(anthropic_client, model):
    anthropic_client.messages.stream.side_effect = anthropic.BadRequestError(
        "bad", response=unittest.mock.Mock(), body=None
    )

    with pytest.raises(anthropic.BadRequestError, match="bad"):
        next(model.stream({}))


def test_structured_output(anthropic_client, model, test_output_model_cls):
    messages = [{"role": "user", "content": [{"text": "Generate a person"}]}]

    events = [
        unittest.mock.Mock(type="message_start", model_dump=unittest.mock.Mock(return_value={"type": "message_start"})),
        unittest.mock.Mock(
            type="content_block_start",
            model_dump=unittest.mock.Mock(
                return_value={
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "123", "name": "TestOutputModel"},
                }
            ),
        ),
        unittest.mock.Mock(
            type="content_block_delta",
            model_dump=unittest.mock.Mock(
                return_value={
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"name": "John", "age": 30}'},
                },
            ),
        ),
        unittest.mock.Mock(
            type="content_block_stop",
            model_dump=unittest.mock.Mock(return_value={"type": "content_block_stop", "index": 0}),
        ),
        unittest.mock.Mock(
            type="message_stop",
            model_dump=unittest.mock.Mock(
                return_value={"type": "message_stop", "message": {"stop_reason": "tool_use"}}
            ),
        ),
        unittest.mock.Mock(
            message=unittest.mock.Mock(
                usage=unittest.mock.Mock(
                    model_dump=unittest.mock.Mock(return_value={"input_tokens": 0, "output_tokens": 0})
                ),
            ),
        ),
    ]

    mock_stream = unittest.mock.MagicMock()
    mock_stream.__iter__.return_value = iter(events)
    anthropic_client.messages.stream.return_value.__enter__.return_value = mock_stream

    stream = model.structured_output(test_output_model_cls, messages)

    tru_result = list(stream)[-1]
    exp_result = {"output": test_output_model_cls(name="John", age=30)}
    assert tru_result == exp_result
