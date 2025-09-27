"""Unit tests for llama.cpp model provider."""

import base64
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import BaseModel

from strands.models.llamacpp import LlamaCppModel
from strands.types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
)


def test_init_default_config() -> None:
    """Test initialization with default configuration."""
    model = LlamaCppModel()

    assert model.config["model_id"] == "default"
    assert isinstance(model.client, httpx.AsyncClient)
    assert model.base_url == "http://localhost:8080"


def test_init_custom_config() -> None:
    """Test initialization with custom configuration."""
    model = LlamaCppModel(
        base_url="http://example.com:8081",
        model_id="llama-3-8b",
        params={"temperature": 0.7, "max_tokens": 100},
    )

    assert model.config["model_id"] == "llama-3-8b"
    assert model.config["params"]["temperature"] == 0.7
    assert model.config["params"]["max_tokens"] == 100
    assert model.base_url == "http://example.com:8081"


def test_format_request_basic() -> None:
    """Test basic request formatting."""
    model = LlamaCppModel(model_id="test-model")

    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
    ]

    request = model._format_request(messages)

    assert request["model"] == "test-model"
    assert request["messages"][0]["role"] == "user"
    assert request["messages"][0]["content"][0]["type"] == "text"
    assert request["messages"][0]["content"][0]["text"] == "Hello"
    assert request["stream"] is True
    assert "extra_body" not in request


def test_format_request_with_system_prompt() -> None:
    """Test request formatting with system prompt."""
    model = LlamaCppModel()

    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
    ]

    request = model._format_request(messages, system_prompt="You are a helpful assistant")

    assert request["messages"][0]["role"] == "system"
    assert request["messages"][0]["content"] == "You are a helpful assistant"
    assert request["messages"][1]["role"] == "user"


def test_format_request_with_llamacpp_params() -> None:
    """Test request formatting with llama.cpp specific parameters."""
    model = LlamaCppModel(
        params={
            "temperature": 0.8,
            "max_tokens": 50,
            "repeat_penalty": 1.1,
            "top_k": 40,
            "min_p": 0.05,
            "grammar": "root ::= 'yes' | 'no'",
        }
    )

    messages = [
        {"role": "user", "content": [{"text": "Is the sky blue?"}]},
    ]

    request = model._format_request(messages)

    # Standard OpenAI params
    assert request["temperature"] == 0.8
    assert request["max_tokens"] == 50

    # Grammar and json_schema go directly in request for llama.cpp
    assert request["grammar"] == "root ::= 'yes' | 'no'"

    # Other llama.cpp specific params should be in extra_body
    assert "extra_body" in request
    assert request["extra_body"]["repeat_penalty"] == 1.1
    assert request["extra_body"]["top_k"] == 40
    assert request["extra_body"]["min_p"] == 0.05


def test_format_request_with_all_new_params() -> None:
    """Test request formatting with all new llama.cpp parameters."""
    model = LlamaCppModel(
        params={
            # OpenAI params
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "seed": 42,
            # All llama.cpp specific params
            "repeat_penalty": 1.1,
            "top_k": 40,
            "min_p": 0.05,
            "typical_p": 0.95,
            "tfs_z": 0.97,
            "top_a": 0.1,
            "mirostat": 2,
            "mirostat_lr": 0.1,
            "mirostat_ent": 5.0,
            "grammar": "root ::= answer",
            "json_schema": {"type": "object"},
            "penalty_last_n": 256,
            "n_probs": 5,
            "min_keep": 1,
            "ignore_eos": False,
            "logit_bias": {100: 5.0, 200: -5.0},
            "cache_prompt": True,
            "slot_id": 1,
            "samplers": ["top_k", "tfs_z", "typical_p"],
        }
    )

    messages = [{"role": "user", "content": [{"text": "Test"}]}]
    request = model._format_request(messages)

    # Check OpenAI params are in root
    assert request["temperature"] == 0.7
    assert request["max_tokens"] == 100
    assert request["top_p"] == 0.9
    assert request["seed"] == 42

    # Grammar and json_schema go directly in request for llama.cpp
    assert request["grammar"] == "root ::= answer"
    assert request["json_schema"] == {"type": "object"}

    # Check all other llama.cpp params are in extra_body
    assert "extra_body" in request
    extra = request["extra_body"]
    assert extra["repeat_penalty"] == 1.1
    assert extra["top_k"] == 40
    assert extra["min_p"] == 0.05
    assert extra["typical_p"] == 0.95
    assert extra["tfs_z"] == 0.97
    assert extra["top_a"] == 0.1
    assert extra["mirostat"] == 2
    assert extra["mirostat_lr"] == 0.1
    assert extra["mirostat_ent"] == 5.0
    assert extra["penalty_last_n"] == 256
    assert extra["n_probs"] == 5
    assert extra["min_keep"] == 1
    assert extra["ignore_eos"] is False
    assert extra["logit_bias"] == {100: 5.0, 200: -5.0}
    assert extra["cache_prompt"] is True
    assert extra["slot_id"] == 1
    assert extra["samplers"] == ["top_k", "tfs_z", "typical_p"]


def test_format_request_with_tools() -> None:
    """Test request formatting with tool specifications."""
    model = LlamaCppModel()

    messages = [
        {"role": "user", "content": [{"text": "What's the weather?"}]},
    ]

    tool_specs = [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                }
            },
        }
    ]

    request = model._format_request(messages, tool_specs=tool_specs)

    assert "tools" in request
    assert len(request["tools"]) == 1
    assert request["tools"][0]["function"]["name"] == "get_weather"


def test_update_config() -> None:
    """Test configuration update."""
    model = LlamaCppModel(model_id="initial-model")

    assert model.config["model_id"] == "initial-model"

    model.update_config(model_id="updated-model", params={"temperature": 0.5})

    assert model.config["model_id"] == "updated-model"
    assert model.config["params"]["temperature"] == 0.5


def test_get_config() -> None:
    """Test configuration retrieval."""
    config = {
        "model_id": "test-model",
        "params": {"temperature": 0.9},
    }
    model = LlamaCppModel(**config)

    retrieved_config = model.get_config()

    assert retrieved_config["model_id"] == "test-model"
    assert retrieved_config["params"]["temperature"] == 0.9


@pytest.mark.asyncio
async def test_stream_basic() -> None:
    """Test basic streaming functionality."""
    model = LlamaCppModel()

    # Mock HTTP response with Server-Sent Events format
    mock_response_lines = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}',
        'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]}',
        'data: {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}',
        "data: [DONE]",
    ]

    async def mock_aiter_lines():
        for line in mock_response_lines:
            yield line

    mock_response = AsyncMock()
    mock_response.aiter_lines = mock_aiter_lines
    mock_response.raise_for_status = AsyncMock()

    with patch.object(model.client, "post", return_value=mock_response):
        messages = [{"role": "user", "content": [{"text": "Hi"}]}]

        chunks = []
        async for chunk in model.stream(messages):
            chunks.append(chunk)

        # Verify we got the expected chunks
        assert any("messageStart" in chunk for chunk in chunks)
        assert any(
            "contentBlockDelta" in chunk and chunk["contentBlockDelta"]["delta"]["text"] == "Hello" for chunk in chunks
        )
        assert any(
            "contentBlockDelta" in chunk and chunk["contentBlockDelta"]["delta"]["text"] == " world" for chunk in chunks
        )
        assert any("messageStop" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_structured_output() -> None:
    """Test structured output functionality."""

    class TestOutput(BaseModel):
        """Test output model for structured output testing."""

        answer: str
        confidence: float

    model = LlamaCppModel()

    # Mock successful JSON response using the new structured_output implementation
    mock_response_text = '{"answer": "yes", "confidence": 0.95}'

    # Create mock stream that returns JSON
    async def mock_stream(*_args, **_kwargs):
        # Verify json_schema was set
        assert "json_schema" in model.config.get("params", {})

        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": mock_response_text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}

    with patch.object(model, "stream", side_effect=mock_stream):
        messages = [{"role": "user", "content": [{"text": "Is the earth round?"}]}]

        events = []
        async for event in model.structured_output(TestOutput, messages):
            events.append(event)

        # Check we got the output
        output_event = next((e for e in events if "output" in e), None)
        assert output_event is not None
        assert output_event["output"].answer == "yes"
        assert output_event["output"].confidence == 0.95


def test_timeout_configuration() -> None:
    """Test timeout configuration."""
    # Test that timeout configuration is accepted without error
    model = LlamaCppModel(timeout=30.0)
    assert model.client.timeout is not None

    # Test with tuple timeout
    model2 = LlamaCppModel(timeout=(10.0, 60.0))
    assert model2.client.timeout is not None


def test_max_retries_configuration() -> None:
    """Test max retries configuration is handled gracefully."""
    # Since httpx doesn't use max_retries in the same way,
    # we just test that the model initializes without error
    model = LlamaCppModel()
    assert model.config["model_id"] == "default"


def test_grammar_constraint_via_params() -> None:
    """Test grammar constraint via params."""
    grammar = """
    root ::= answer
    answer ::= "yes" | "no"
    """
    model = LlamaCppModel(params={"grammar": grammar})

    assert model.config["params"]["grammar"] == grammar

    # Update grammar via update_config
    new_grammar = "root ::= [0-9]+"
    model.update_config(params={"grammar": new_grammar})

    assert model.config["params"]["grammar"] == new_grammar


def test_json_schema_via_params() -> None:
    """Test JSON schema constraint via params."""
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    model = LlamaCppModel(params={"json_schema": schema})

    assert model.config["params"]["json_schema"] == schema


@pytest.mark.asyncio
async def test_stream_with_context_overflow_error() -> None:
    """Test stream handling of context overflow errors."""
    model = LlamaCppModel()

    # Create HTTP error response
    error_response = httpx.Response(
        status_code=400,
        json={"error": {"message": "Context window exceeded. Max context length is 4096 tokens"}},
        request=httpx.Request("POST", "http://test.com"),
    )
    error = httpx.HTTPStatusError("Bad Request", request=error_response.request, response=error_response)

    # Mock the client to raise the error
    with patch.object(model.client, "post", side_effect=error):
        messages = [{"role": "user", "content": [{"text": "Very long message"}]}]

        with pytest.raises(ContextWindowOverflowException) as exc_info:
            async for _ in model.stream(messages):
                pass

        assert "Context window exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_stream_with_server_overload_error() -> None:
    """Test stream handling of server overload errors."""
    model = LlamaCppModel()

    # Create HTTP error response for 503
    error_response = httpx.Response(
        status_code=503,
        text="Server is busy",
        request=httpx.Request("POST", "http://test.com"),
    )
    error = httpx.HTTPStatusError(
        "Service Unavailable",
        request=error_response.request,
        response=error_response,
    )

    # Mock the client to raise the error
    with patch.object(model.client, "post", side_effect=error):
        messages = [{"role": "user", "content": [{"text": "Test"}]}]

        with pytest.raises(ModelThrottledException) as exc_info:
            async for _ in model.stream(messages):
                pass

        assert "server is busy or overloaded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_structured_output_with_json_schema() -> None:
    """Test structured output using JSON schema."""

    class TestOutput(BaseModel):
        """Test output model for JSON schema testing."""

        answer: str
        confidence: float

    model = LlamaCppModel()

    # Mock successful JSON response
    mock_response_text = '{"answer": "yes", "confidence": 0.95}'

    # Create mock stream that returns JSON
    async def mock_stream(*_args, **_kwargs):
        # Check that json_schema was set correctly
        assert model.config["params"]["json_schema"] == TestOutput.model_json_schema()

        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": mock_response_text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}

    with patch.object(model, "stream", side_effect=mock_stream):
        messages = [{"role": "user", "content": [{"text": "Is the earth round?"}]}]

        events = []
        async for event in model.structured_output(TestOutput, messages):
            events.append(event)

        # Check we got the output
        output_event = next((e for e in events if "output" in e), None)
        assert output_event is not None
        assert output_event["output"].answer == "yes"
        assert output_event["output"].confidence == 0.95


@pytest.mark.asyncio
async def test_structured_output_invalid_json_error() -> None:
    """Test structured output raises error for invalid JSON."""

    class TestOutput(BaseModel):
        """Test output model for invalid JSON testing."""

        value: int

    model = LlamaCppModel()

    # Mock stream that returns invalid JSON
    async def mock_stream(*_args, **_kwargs):
        # Check that json_schema was set correctly
        assert model.config["params"]["json_schema"] == TestOutput.model_json_schema()

        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": "This is not valid JSON"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}

    with patch.object(model, "stream", side_effect=mock_stream):
        messages = [{"role": "user", "content": [{"text": "Give me a number"}]}]

        with pytest.raises(json.JSONDecodeError):
            async for _ in model.structured_output(TestOutput, messages):
                pass


def test_format_audio_content() -> None:
    """Test formatting of audio content for llama.cpp multimodal models."""
    model = LlamaCppModel()

    # Create test audio data
    audio_bytes = b"fake audio data"
    audio_content = {"audio": {"source": {"bytes": audio_bytes}, "format": "wav"}}

    # Format the content
    result = model._format_message_content(audio_content)

    # Verify the structure
    assert result["type"] == "input_audio"
    assert "input_audio" in result
    assert "data" in result["input_audio"]
    assert "format" in result["input_audio"]

    # Verify the data is base64 encoded
    decoded = base64.b64decode(result["input_audio"]["data"])
    assert decoded == audio_bytes

    # Verify format is preserved
    assert result["input_audio"]["format"] == "wav"


def test_format_audio_content_default_format() -> None:
    """Test audio content formatting uses wav as default format."""
    model = LlamaCppModel()

    audio_content = {
        "audio": {"source": {"bytes": b"test audio"}}
        # No format specified
    }

    result = model._format_message_content(audio_content)

    # Should default to wav
    assert result["input_audio"]["format"] == "wav"


def test_format_messages_with_audio() -> None:
    """Test that _format_messages properly handles audio content."""
    model = LlamaCppModel()

    # Create messages with audio content
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Listen to this audio:"},
                {"audio": {"source": {"bytes": b"audio data"}, "format": "mp3"}},
            ],
        }
    ]

    # Format the messages
    result = model._format_messages(messages)

    # Check structure
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert len(result[0]["content"]) == 2

    # Check text content
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][0]["text"] == "Listen to this audio:"

    # Check audio content
    assert result[0]["content"][1]["type"] == "input_audio"
    assert "input_audio" in result[0]["content"][1]
    assert result[0]["content"][1]["input_audio"]["format"] == "mp3"


def test_format_messages_with_system_prompt() -> None:
    """Test _format_messages includes system prompt."""
    model = LlamaCppModel()

    messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    system_prompt = "You are a helpful assistant"

    result = model._format_messages(messages, system_prompt)

    # Should have system message first
    assert len(result) == 2
    assert result[0]["role"] == "system"
    assert result[0]["content"] == system_prompt
    assert result[1]["role"] == "user"


def test_format_messages_with_image() -> None:
    """Test that _format_messages properly handles image content."""
    model = LlamaCppModel()

    # Create messages with image content
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Describe this image:"},
                {"image": {"source": {"bytes": b"image data"}, "format": "png"}},
            ],
        }
    ]

    # Format the messages
    result = model._format_messages(messages)

    # Check structure
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert len(result[0]["content"]) == 2

    # Check text content
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][0]["text"] == "Describe this image:"

    # Check image content uses standard format
    assert result[0]["content"][1]["type"] == "image_url"
    assert "image_url" in result[0]["content"][1]
    assert "url" in result[0]["content"][1]["image_url"]
    assert result[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_format_messages_with_mixed_content() -> None:
    """Test that _format_messages handles mixed audio and image content correctly."""
    model = LlamaCppModel()

    # Create messages with both audio and image content
    messages = [
        {
            "role": "user",
            "content": [
                {"text": "Analyze this media:"},
                {"audio": {"source": {"bytes": b"audio data"}, "format": "wav"}},
                {"image": {"source": {"bytes": b"image data"}, "format": "jpg"}},
            ],
        }
    ]

    # Format the messages
    result = model._format_messages(messages)

    # Check structure
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert len(result[0]["content"]) == 3

    # Check text content
    assert result[0]["content"][0]["type"] == "text"
    assert result[0]["content"][0]["text"] == "Analyze this media:"

    # Check audio content uses llama.cpp specific format
    assert result[0]["content"][1]["type"] == "input_audio"
    assert "input_audio" in result[0]["content"][1]
    assert result[0]["content"][1]["input_audio"]["format"] == "wav"

    # Check image content uses standard OpenAI format
    assert result[0]["content"][2]["type"] == "image_url"
    assert "image_url" in result[0]["content"][2]
    assert result[0]["content"][2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
