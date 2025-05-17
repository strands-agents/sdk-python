"""Tests for the Gemini model provider."""

import json
import unittest.mock

import pytest

import strands
from strands.models.gemini import GeminiModel
from strands.types.exceptions import ContextWindowOverflowException, ModelThrottledException


@pytest.fixture
def gemini_client():
    with unittest.mock.patch.object(strands.models.gemini.genai, "GenerativeModel") as mock_client_cls:
        yield mock_client_cls.return_value


@pytest.fixture
def model_id():
    return "gemini-pro"


@pytest.fixture
def max_tokens():
    return 1000


@pytest.fixture
def model(gemini_client, model_id, max_tokens):
    _ = gemini_client
    return GeminiModel(model_id=model_id, max_tokens=max_tokens)


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "test"}]}]


@pytest.fixture
def system_prompt():
    return "s1"


def test__init__model_configs(gemini_client, model_id, max_tokens):
    _ = gemini_client

    model = GeminiModel(model_id=model_id, max_tokens=max_tokens, params={"temperature": 1})

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
        "contents": [{"role": "user", "parts": [{"text": "test"}]}],
        "generation_config": {"max_output_tokens": max_tokens},
        "tools": None,
        "system_instruction": None,
    }

    assert tru_request == exp_request


def test_format_request_with_params(model, messages, model_id, max_tokens):
    model.update_config(params={"temperature": 1})

    tru_request = model.format_request(messages)
    exp_request = {
        "contents": [{"role": "user", "parts": [{"text": "test"}]}],
        "generation_config": {
            "max_output_tokens": max_tokens,
            "temperature": 1,
        },
        "tools": None,
        "system_instruction": None,
    }

    assert tru_request == exp_request


def test_format_request_with_system_prompt(model, messages, model_id, max_tokens, system_prompt):
    tru_request = model.format_request(messages, system_prompt=system_prompt)
    exp_request = {
        "contents": [{"role": "user", "parts": [{"text": "test"}]}],
        "generation_config": {"max_output_tokens": max_tokens},
        "tools": None,
        "system_instruction": system_prompt,
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
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "data": "YmFzZTY0ZW5jb2RlZGltYWdl",
                            "mime_type": "image/jpeg",
                        }
                    }
                ],
            }
        ],
        "generation_config": {"max_output_tokens": max_tokens},
        "tools": None,
        "system_instruction": None,
    }

    assert tru_request == exp_request


def test_format_request_with_other(model, model_id, max_tokens):
    messages = [
        {
            "role": "user",
            "content": [{"other": {"a": 1}}],
        },
    ]

    tru_request = model.format_request(messages)
    exp_request = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": json.dumps({"other": {"a": 1}}),
                    }
                ],
            }
        ],
        "generation_config": {"max_output_tokens": max_tokens},
        "tools": None,
        "system_instruction": None,
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
        "contents": [],
        "generation_config": {"max_output_tokens": max_tokens},
        "tools": None,
        "system_instruction": None,
    }

    assert tru_request == exp_request


def test_format_chunk_message_start(model):
    event = {"type": "message_start"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStart": {"role": "assistant"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_content_block_start(model):
    event = {
        "type": "content_block_start",
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {
        "contentBlockStart": {
            "start": {},
        },
    }

    assert tru_chunk == exp_chunk


def test_format_chunk_content_block_delta(model):
    event = {
        "type": "content_block_delta",
        "text": "hello",
    }

    tru_chunk = model.format_chunk(event)
    exp_chunk = {
        "contentBlockDelta": {
            "delta": {"text": "hello"},
        },
    }

    assert tru_chunk == exp_chunk


def test_format_chunk_content_block_stop(model):
    event = {"type": "content_block_stop"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"contentBlockStop": {}}

    assert tru_chunk == exp_chunk


def test_format_chunk_message_stop(model):
    event = {"type": "message_stop", "stop_reason": "end_turn"}

    tru_chunk = model.format_chunk(event)
    exp_chunk = {"messageStop": {"stopReason": "end_turn"}}

    assert tru_chunk == exp_chunk


def test_format_chunk_metadata(model):
    event = {
        "type": "metadata",
        "usage": {
            "prompt_token_count": 1,
            "candidates_token_count": 2,
            "total_token_count": 3,
        },
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

    with pytest.raises(RuntimeError, match="event_type=<unknown> | unknown type"):
        model.format_chunk(event)


def test_stream(gemini_client, model):
    mock_chunk = unittest.mock.Mock(text="test")
    mock_response = unittest.mock.MagicMock()
    mock_response.__iter__.return_value = iter([mock_chunk])
    mock_response.usage_metadata = unittest.mock.Mock(
        prompt_token_count=1,
        candidates_token_count=2,
        total_token_count=3,
    )
    gemini_client.generate_content.return_value = mock_response

    request = {"model": "gemini-pro"}
    response = model.stream(request)

    tru_events = list(response)
    exp_events = [
        {"type": "message_start"},
        {"type": "content_block_start"},
        {"type": "content_block_delta", "text": "test"},
        {"type": "content_block_stop"},
        {"type": "message_stop", "stop_reason": "end_turn"},
        {
            "type": "metadata",
            "usage": {
                "prompt_token_count": 1,
                "candidates_token_count": 2,
                "total_token_count": 3,
            },
        },
    ]

    assert tru_events == exp_events
    gemini_client.generate_content.assert_called_once_with(**request, stream=True)


def test_stream_quota_error(gemini_client, model):
    gemini_client.generate_content.side_effect = Exception("quota exceeded")

    with pytest.raises(ModelThrottledException, match="quota exceeded"):
        next(model.stream({}))


@pytest.mark.parametrize(
    "overflow_message",
    [
        "...input is too long...",
        "...input length exceeds context window...",
        "...input and output tokens exceed your context limit...",
    ],
)
def test_stream_context_window_overflow_error(overflow_message, gemini_client, model):
    gemini_client.generate_content.side_effect = Exception(overflow_message)

    with pytest.raises(ContextWindowOverflowException):
        next(model.stream({}))


def test_stream_other_error(gemini_client, model):
    gemini_client.generate_content.side_effect = Exception("other error")

    with pytest.raises(Exception, match="other error"):
        next(model.stream({}))
