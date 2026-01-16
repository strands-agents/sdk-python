"""Unit tests for the xAI model provider."""

import unittest.mock
from contextlib import contextmanager
from typing import Any, Generator

import pytest

import strands
from strands.models.xai import xAIModel


@contextmanager
def mock_xai_sdk() -> Generator[dict[str, unittest.mock.Mock], None, None]:
    """Context manager to mock the xAI SDK components."""
    with (
        unittest.mock.patch.object(strands.models.xai, "AsyncClient") as mock_client_cls,
        unittest.mock.patch.object(strands.models.xai, "xai_tool") as mock_xai_tool,
        unittest.mock.patch.object(strands.models.xai, "xai_system") as mock_xai_system,
        unittest.mock.patch.object(strands.models.xai, "xai_user") as mock_xai_user,
        unittest.mock.patch.object(strands.models.xai, "xai_tool_result") as mock_xai_tool_result,
        unittest.mock.patch.object(
            strands.models.xai, "get_tool_call_type", return_value="client_side_tool"
        ) as mock_get_tool_call_type,
    ):
        mock_client = mock_client_cls.return_value

        def create_tool_mock(name: str, description: str, parameters: dict) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {"name": name, "description": description, "parameters": parameters},
            }

        mock_xai_tool.side_effect = create_tool_mock

        yield {
            "client": mock_client,
            "client_cls": mock_client_cls,
            "xai_tool": mock_xai_tool,
            "xai_system": mock_xai_system,
            "xai_user": mock_xai_user,
            "xai_tool_result": mock_xai_tool_result,
            "get_tool_call_type": mock_get_tool_call_type,
        }


@contextmanager
def mock_xai_client() -> Generator[unittest.mock.Mock, None, None]:
    """Context manager to mock the xAI AsyncClient."""
    with mock_xai_sdk() as mocks:
        yield mocks["client"]


@pytest.fixture
def mock_xai_client_fixture() -> Generator[unittest.mock.Mock, None, None]:
    """Pytest fixture to mock the xAI AsyncClient."""
    with mock_xai_client() as client:
        yield client


@pytest.fixture
def mock_xai_sdk_fixture() -> Generator[dict[str, unittest.mock.Mock], None, None]:
    """Pytest fixture to mock the full xAI SDK."""
    with mock_xai_sdk() as mocks:
        yield mocks


@pytest.fixture
def model_id() -> str:
    """Default model ID for tests."""
    return "grok-4"


@pytest.fixture
def model(mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str) -> xAIModel:
    """Create a xAIModel instance with mocked SDK."""
    _ = mock_xai_sdk_fixture
    return xAIModel(model_id=model_id)


class TestxAIConfigRoundTrip:
    """Tests for configuration round-trip consistency."""

    @pytest.mark.parametrize(
        "model_id",
        [
            "grok-3-mini-fast-latest",
            "grok-2-latest",
            "test-model-123",
            "model_with_underscores",
            "model-with-dashes",
        ],
    )
    def test_config_round_trip_model_id_only(self, model_id: str) -> None:
        """For any valid model_id, get_config returns equivalent config."""
        with mock_xai_client():
            model = xAIModel(model_id=model_id)
            config = model.get_config()
            assert config["model_id"] == model_id

    @pytest.mark.parametrize(
        "model_id,params",
        [
            ("grok-3-mini-fast-latest", {"temperature": 0.7}),
            ("grok-2-latest", {"max_tokens": 1000}),
            ("test-model", {"temperature": 1.5, "max_tokens": 2048}),
            ("model-123", {}),
        ],
    )
    def test_config_round_trip_with_params(self, model_id: str, params: dict) -> None:
        """For any valid model_id and params, config round-trip preserves values."""
        with mock_xai_client():
            model = xAIModel(model_id=model_id, params=params)
            config = model.get_config()
            assert config["model_id"] == model_id
            if params:
                assert config["params"] == params

    @pytest.mark.parametrize(
        "model_id,reasoning_effort",
        [
            ("grok-3-mini-fast-latest", "low"),
            ("grok-2-latest", "high"),
        ],
    )
    def test_config_round_trip_with_reasoning_effort(self, model_id: str, reasoning_effort: str) -> None:
        """For any valid model_id and reasoning_effort, config round-trip preserves values."""
        with mock_xai_client():
            model = xAIModel(model_id=model_id, reasoning_effort=reasoning_effort)
            config = model.get_config()
            assert config["model_id"] == model_id
            assert config["reasoning_effort"] == reasoning_effort

    @pytest.mark.parametrize(
        "model_id,include",
        [
            ("grok-3-mini-fast-latest", ["verbose_streaming"]),
            ("grok-2-latest", ["inline_citations"]),
            ("test-model", ["verbose_streaming", "inline_citations"]),
            ("model-123", []),
        ],
    )
    def test_config_round_trip_with_include(self, model_id: str, include: list) -> None:
        """For any valid model_id and include list, config round-trip preserves values."""
        with mock_xai_client():
            model = xAIModel(model_id=model_id, include=include)
            config = model.get_config()
            assert config["model_id"] == model_id
            if include:
                assert config["include"] == include


class TestxAIModelInit:
    """Unit tests for xAIModel initialization."""

    def test_init_with_model_id(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test initialization with just model_id."""
        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id)
        assert model.get_config()["model_id"] == model_id

    def test_init_with_params(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test initialization with params."""
        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id, params={"temperature": 0.7})
        config = model.get_config()
        assert config["model_id"] == model_id
        assert config["params"] == {"temperature": 0.7}

    def test_init_with_client_args(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test initialization with client_args."""
        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id, client_args={"api_key": "test-key"})
        assert model.client_args == {"api_key": "test-key"}

    def test_init_with_custom_client(self, model_id: str) -> None:
        """Test initialization with a custom client."""
        mock_client = unittest.mock.Mock()
        model = xAIModel(client=mock_client, model_id=model_id)
        assert model._custom_client is mock_client

    def test_init_with_both_client_and_client_args_raises_error(self, model_id: str) -> None:
        """Test that providing both client and client_args raises ValueError."""
        mock_client = unittest.mock.Mock()
        with pytest.raises(ValueError, match="Only one of 'client' or 'client_args' should be provided"):
            xAIModel(client=mock_client, client_args={"api_key": "test"}, model_id=model_id)


class TestxAIModelGetClient:
    """Unit tests for xAIModel._get_client method."""

    def test_get_client_returns_custom_client(self, model_id: str) -> None:
        """Test that _get_client returns the injected client when provided."""
        mock_client = unittest.mock.Mock()
        model = xAIModel(client=mock_client, model_id=model_id)
        result = model._get_client()
        assert result is mock_client

    def test_get_client_creates_new_client(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test that _get_client creates a new client when no custom client is provided."""
        model = xAIModel(model_id=model_id, client_args={"api_key": "test-key"})
        model._get_client()
        strands.models.xai.AsyncClient.assert_called_with(api_key="test-key")


class TestGrokToolsValidation:
    """Unit tests for xai_tools validation."""

    def test_validate_xai_tools_rejects_function_tools(
        self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str
    ) -> None:
        """Test that function-based tools (dicts with type=function) are rejected in xai_tools."""
        _ = mock_xai_client_fixture
        # Client-side tools created via xai_tool() are dicts with "type": "function"
        mock_function_tool = {"type": "function", "function": {"name": "test_function"}}
        with pytest.raises(ValueError, match="xai_tools should not contain function-based tools"):
            xAIModel(model_id=model_id, xai_tools=[mock_function_tool])

    def test_validate_xai_tools_accepts_server_side_tools(
        self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str
    ) -> None:
        """Test that server-side tools (protobuf objects) are accepted in xai_tools."""
        _ = mock_xai_client_fixture
        # Server-side tools like web_search() are protobuf objects, not dicts
        mock_server_tool = unittest.mock.Mock(spec=[])
        model = xAIModel(model_id=model_id, xai_tools=[mock_server_tool])
        assert "xai_tools" in model.get_config()

    def test_validate_xai_tools_on_update_config(
        self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str
    ) -> None:
        """Test that xai_tools validation runs on update_config."""
        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id)
        # Client-side tools created via xai_tool() are dicts with "type": "function"
        mock_function_tool = {"type": "function", "function": {"name": "test_function"}}
        with pytest.raises(ValueError, match="xai_tools should not contain function-based tools"):
            model.update_config(xai_tools=[mock_function_tool])


class TestFormatRequestTools:
    """Unit tests for _format_request_tools method."""

    def test_format_empty_tools(self, model: xAIModel) -> None:
        """Test formatting with no tools."""
        result = model._format_request_tools(None)
        assert result == []

    def test_format_single_tool(self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str) -> None:
        """Test formatting a single tool spec."""
        model = xAIModel(model_id=model_id)
        tool_specs = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "inputSchema": {"json": {"type": "object", "properties": {"location": {"type": "string"}}}},
            }
        ]
        result = model._format_request_tools(tool_specs)
        assert len(result) == 1
        mock_xai_sdk_fixture["xai_tool"].assert_called_once_with(
            name="get_weather",
            description="Get weather for a location",
            parameters={"type": "object", "properties": {"location": {"type": "string"}}},
        )

    def test_format_multiple_tools(self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str) -> None:
        """Test formatting multiple tool specs."""
        model = xAIModel(model_id=model_id)
        tool_specs = [
            {"name": "tool1", "description": "First tool", "inputSchema": {"json": {"type": "object"}}},
            {"name": "tool2", "description": "Second tool", "inputSchema": {"json": {"type": "object"}}},
        ]
        result = model._format_request_tools(tool_specs)
        assert len(result) == 2
        assert mock_xai_sdk_fixture["xai_tool"].call_count == 2

    def test_format_tools_with_xai_tools(
        self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str
    ) -> None:
        """Test that xai_tools are appended to formatted tools."""
        mock_server_tool = unittest.mock.Mock(spec=[])
        model = xAIModel(model_id=model_id, xai_tools=[mock_server_tool])
        tool_specs = [{"name": "tool1", "description": "Tool", "inputSchema": {"json": {"type": "object"}}}]
        result = model._format_request_tools(tool_specs)
        assert len(result) == 2
        assert mock_server_tool in result


class TestFormatChunk:
    """Unit tests for _format_chunk method."""

    def test_format_message_start(self, model: xAIModel) -> None:
        """Test formatting message_start chunk."""
        result = model._format_chunk({"chunk_type": "message_start"})
        assert result == {"messageStart": {"role": "assistant"}}

    def test_format_content_start_text(self, model: xAIModel) -> None:
        """Test formatting content_start chunk for text."""
        result = model._format_chunk({"chunk_type": "content_start", "data_type": "text"})
        assert result == {"contentBlockStart": {"start": {}}}

    def test_format_content_start_tool(self, model: xAIModel) -> None:
        """Test formatting content_start chunk for tool."""
        result = model._format_chunk(
            {
                "chunk_type": "content_start",
                "data_type": "tool",
                "data": {"name": "get_weather", "id": "tool-123"},
            }
        )
        assert result == {"contentBlockStart": {"start": {"toolUse": {"name": "get_weather", "toolUseId": "tool-123"}}}}

    def test_format_content_delta_text(self, model: xAIModel) -> None:
        """Test formatting content_delta chunk for text."""
        result = model._format_chunk({"chunk_type": "content_delta", "data_type": "text", "data": "Hello"})
        assert result == {"contentBlockDelta": {"delta": {"text": "Hello"}}}

    def test_format_content_delta_tool(self, model: xAIModel) -> None:
        """Test formatting content_delta chunk for tool."""
        result = model._format_chunk(
            {
                "chunk_type": "content_delta",
                "data_type": "tool",
                "data": {"arguments": '{"location": "Paris"}'},
            }
        )
        assert result == {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"location": "Paris"}'}}}}

    def test_format_content_delta_reasoning(self, model: xAIModel) -> None:
        """Test formatting content_delta chunk for reasoning content."""
        result = model._format_chunk(
            {"chunk_type": "content_delta", "data_type": "reasoning_content", "data": "Thinking..."}
        )
        assert result == {"contentBlockDelta": {"delta": {"reasoningContent": {"text": "Thinking..."}}}}

    def test_format_content_stop(self, model: xAIModel) -> None:
        """Test formatting content_stop chunk."""
        result = model._format_chunk({"chunk_type": "content_stop"})
        assert result == {"contentBlockStop": {}}

    def test_format_message_stop_end_turn(self, model: xAIModel) -> None:
        """Test formatting message_stop chunk with end_turn."""
        result = model._format_chunk({"chunk_type": "message_stop", "data": "end_turn"})
        assert result == {"messageStop": {"stopReason": "end_turn"}}

    def test_format_message_stop_tool_use(self, model: xAIModel) -> None:
        """Test formatting message_stop chunk with tool_use."""
        result = model._format_chunk({"chunk_type": "message_stop", "data": "tool_use"})
        assert result == {"messageStop": {"stopReason": "tool_use"}}

    def test_format_message_stop_max_tokens(self, model: xAIModel) -> None:
        """Test formatting message_stop chunk with max_tokens."""
        result = model._format_chunk({"chunk_type": "message_stop", "data": "max_tokens"})
        assert result == {"messageStop": {"stopReason": "max_tokens"}}

    def test_format_metadata(self, model: xAIModel) -> None:
        """Test formatting metadata chunk."""
        mock_usage = unittest.mock.Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        mock_usage.reasoning_tokens = None
        result = model._format_chunk({"chunk_type": "metadata", "data": mock_usage})
        assert result["metadata"]["usage"]["inputTokens"] == 100
        assert result["metadata"]["usage"]["outputTokens"] == 50
        assert result["metadata"]["usage"]["totalTokens"] == 150

    def test_format_metadata_with_reasoning_tokens(self, model: xAIModel) -> None:
        """Test formatting metadata chunk with reasoning tokens."""
        mock_usage = unittest.mock.Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        mock_usage.reasoning_tokens = 25
        result = model._format_chunk({"chunk_type": "metadata", "data": mock_usage})
        assert result["metadata"]["usage"]["reasoningTokens"] == 25

    def test_format_metadata_with_citations(self, model: xAIModel) -> None:
        """Test formatting metadata chunk with citations."""
        mock_usage = unittest.mock.Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        mock_usage.reasoning_tokens = None
        citations = [{"url": "https://example.com"}]
        result = model._format_chunk({"chunk_type": "metadata", "data": mock_usage, "citations": citations})
        assert result["metadata"]["citations"] == citations

    def test_format_unknown_chunk_raises_error(self, model: xAIModel) -> None:
        """Test that unknown chunk types raise RuntimeError."""
        with pytest.raises(RuntimeError, match="chunk_type=<unknown> | unknown type"):
            model._format_chunk({"chunk_type": "unknown"})


class TestHandleStreamError:
    """Unit tests for _handle_stream_error method."""

    def test_rate_limit_error(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test that rate limit errors raise ModelThrottledException."""
        from strands.types.exceptions import ModelThrottledException

        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id)
        error = Exception("Rate limit exceeded")
        with pytest.raises(ModelThrottledException, match="Rate limit"):
            model._handle_stream_error(error)

    def test_rate_limit_error_429(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test that 429 errors raise ModelThrottledException."""
        from strands.types.exceptions import ModelThrottledException

        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id)
        error = Exception("Error 429: Too many requests")
        with pytest.raises(ModelThrottledException, match="429"):
            model._handle_stream_error(error)

    def test_too_many_requests_error(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test that 'too many requests' errors raise ModelThrottledException."""
        from strands.types.exceptions import ModelThrottledException

        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id)
        error = Exception("Too many requests, please slow down")
        with pytest.raises(ModelThrottledException, match="Too many requests"):
            model._handle_stream_error(error)

    def test_context_length_error(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test that context length errors raise ContextWindowOverflowException."""
        from strands.types.exceptions import ContextWindowOverflowException

        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id)
        error = Exception("Context length exceeded")
        with pytest.raises(ContextWindowOverflowException, match="Context length"):
            model._handle_stream_error(error)

    def test_maximum_context_error(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test that maximum context errors raise ContextWindowOverflowException."""
        from strands.types.exceptions import ContextWindowOverflowException

        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id)
        error = Exception("Maximum context length reached")
        with pytest.raises(ContextWindowOverflowException, match="Maximum context"):
            model._handle_stream_error(error)

    def test_token_limit_error(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test that token limit errors raise ContextWindowOverflowException."""
        from strands.types.exceptions import ContextWindowOverflowException

        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id)
        error = Exception("Token limit exceeded")
        with pytest.raises(ContextWindowOverflowException, match="Token limit"):
            model._handle_stream_error(error)

    def test_other_error_reraises(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test that other errors are re-raised unchanged."""
        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id)
        error = Exception("Some other error")
        with pytest.raises(Exception, match="Some other error"):
            model._handle_stream_error(error)


class TestBuildChat:
    """Unit tests for _build_chat method."""

    def test_build_chat_basic(self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str) -> None:
        """Test building a basic chat."""
        model = xAIModel(model_id=model_id)
        mock_client = mock_xai_sdk_fixture["client"]
        mock_chat = unittest.mock.Mock()
        mock_client.chat.create.return_value = mock_chat

        result = model._build_chat(mock_client)

        mock_client.chat.create.assert_called_once_with(model=model_id, store_messages=False)
        assert result is mock_chat

    def test_build_chat_with_tools(self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str) -> None:
        """Test building a chat with tools."""
        model = xAIModel(model_id=model_id)
        mock_client = mock_xai_sdk_fixture["client"]
        mock_chat = unittest.mock.Mock()
        mock_client.chat.create.return_value = mock_chat
        tool_specs = [{"name": "tool1", "description": "Tool", "inputSchema": {"json": {"type": "object"}}}]

        model._build_chat(mock_client, tool_specs)

        call_kwargs = mock_client.chat.create.call_args[1]
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1

    def test_build_chat_with_reasoning_effort(
        self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str
    ) -> None:
        """Test building a chat with reasoning_effort."""
        model = xAIModel(model_id=model_id, reasoning_effort="high")
        mock_client = mock_xai_sdk_fixture["client"]
        mock_chat = unittest.mock.Mock()
        mock_client.chat.create.return_value = mock_chat

        model._build_chat(mock_client)

        call_kwargs = mock_client.chat.create.call_args[1]
        assert call_kwargs["reasoning_effort"] == "high"

    def test_build_chat_with_include(self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str) -> None:
        """Test building a chat with include options."""
        model = xAIModel(model_id=model_id, include=["verbose_streaming"])
        mock_client = mock_xai_sdk_fixture["client"]
        mock_chat = unittest.mock.Mock()
        mock_client.chat.create.return_value = mock_chat

        model._build_chat(mock_client)

        call_kwargs = mock_client.chat.create.call_args[1]
        assert call_kwargs["include"] == ["verbose_streaming"]

    def test_build_chat_with_params(self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str) -> None:
        """Test building a chat with additional params."""
        model = xAIModel(model_id=model_id, params={"temperature": 0.7, "max_tokens": 1000})
        mock_client = mock_xai_sdk_fixture["client"]
        mock_chat = unittest.mock.Mock()
        mock_client.chat.create.return_value = mock_chat

        model._build_chat(mock_client)

        call_kwargs = mock_client.chat.create.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1000


class TestAppendMessagesToChat:
    """Unit tests for _append_messages_to_chat method."""

    def test_append_system_prompt(self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str) -> None:
        """Test appending system prompt."""
        model = xAIModel(model_id=model_id)
        mock_chat = unittest.mock.Mock()
        mock_xai_sdk_fixture["xai_system"].return_value = "system_msg"

        model._append_messages_to_chat(mock_chat, [], system_prompt="You are helpful")

        mock_xai_sdk_fixture["xai_system"].assert_called_once_with("You are helpful")
        mock_chat.append.assert_called_once_with("system_msg")

    def test_append_user_message_with_text(
        self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str
    ) -> None:
        """Test appending user message with text."""
        model = xAIModel(model_id=model_id)
        mock_chat = unittest.mock.Mock()
        mock_xai_sdk_fixture["xai_user"].return_value = "user_msg"
        messages = [{"role": "user", "content": [{"text": "Hello"}]}]

        model._append_messages_to_chat(mock_chat, messages)

        mock_xai_sdk_fixture["xai_user"].assert_called_once_with("Hello")
        mock_chat.append.assert_called_once_with("user_msg")

    def test_append_user_message_with_tool_result(
        self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str
    ) -> None:
        """Test appending user message with tool result."""
        model = xAIModel(model_id=model_id)
        mock_chat = unittest.mock.Mock()
        mock_xai_sdk_fixture["xai_tool_result"].return_value = "tool_result_msg"
        messages = [
            {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [{"text": "Result"}]}}]}
        ]

        model._append_messages_to_chat(mock_chat, messages)

        mock_xai_sdk_fixture["xai_tool_result"].assert_called_once_with("Result")
        mock_chat.append.assert_called_once_with("tool_result_msg")

    def test_append_user_message_with_json_tool_result(
        self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str
    ) -> None:
        """Test appending user message with JSON tool result."""
        model = xAIModel(model_id=model_id)
        mock_chat = unittest.mock.Mock()
        mock_xai_sdk_fixture["xai_tool_result"].return_value = "tool_result_msg"
        messages = [
            {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [{"json": {"key": "value"}}]}}]}
        ]

        model._append_messages_to_chat(mock_chat, messages)

        mock_xai_sdk_fixture["xai_tool_result"].assert_called_once_with('{"key": "value"}')

    def test_append_assistant_message_with_text(
        self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str
    ) -> None:
        """Test that assistant messages are reconstructed as protobuf messages."""
        model = xAIModel(model_id=model_id)
        mock_chat = unittest.mock.Mock()
        messages = [{"role": "assistant", "content": [{"text": "Hello"}]}]

        model._append_messages_to_chat(mock_chat, messages)

        # Assistant messages should be appended as protobuf Message objects
        mock_chat.append.assert_called_once()
        # Verify the appended message is a protobuf Message with correct content
        appended_msg = mock_chat.append.call_args[0][0]
        assert appended_msg.role == 2  # ROLE_ASSISTANT
        assert len(appended_msg.content) == 1
        assert appended_msg.content[0].text == "Hello"


class TestUpdateConfig:
    """Unit tests for update_config method."""

    def test_update_model_id(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test updating model_id."""
        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id)
        model.update_config(model_id="grok-4-fast")
        assert model.get_config()["model_id"] == "grok-4-fast"

    def test_update_params(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test updating params."""
        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id)
        model.update_config(params={"temperature": 0.5})
        assert model.get_config()["params"] == {"temperature": 0.5}

    def test_update_reasoning_effort(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test updating reasoning_effort."""
        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id)
        model.update_config(reasoning_effort="low")
        assert model.get_config()["reasoning_effort"] == "low"

    def test_update_include(self, mock_xai_client_fixture: unittest.mock.Mock, model_id: str) -> None:
        """Test updating include."""
        _ = mock_xai_client_fixture
        model = xAIModel(model_id=model_id)
        model.update_config(include=["inline_citations"])
        assert model.get_config()["include"] == ["inline_citations"]


class TestStream:
    """Unit tests for stream method."""

    @pytest.mark.asyncio
    async def test_stream_basic_response(
        self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str
    ) -> None:
        """Test streaming a basic response."""
        model = xAIModel(model_id=model_id)
        mock_client = mock_xai_sdk_fixture["client"]
        mock_chat = unittest.mock.Mock()
        mock_client.chat.create.return_value = mock_chat

        # Create mock response and chunk
        mock_response = unittest.mock.Mock()
        mock_response.usage = unittest.mock.Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.reasoning_tokens = None
        mock_response.citations = None
        mock_response.encrypted_content = None  # Explicitly set to avoid xAI state capture

        mock_chunk = unittest.mock.Mock()
        mock_chunk.content = "Hello"
        mock_chunk.reasoning_content = None
        mock_chunk.tool_calls = None

        async def mock_stream():
            yield mock_response, mock_chunk

        mock_chat.stream.return_value = mock_stream()

        events = []
        async for event in model.stream(messages=[], system_prompt="Test"):
            events.append(event)

        # Should have: message_start, content_start, content_delta, content_stop, message_stop, metadata
        assert len(events) >= 5
        assert events[0] == {"messageStart": {"role": "assistant"}}

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(
        self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str
    ) -> None:
        """Test streaming a response with client-side tool calls."""
        model = xAIModel(model_id=model_id)
        mock_client = mock_xai_sdk_fixture["client"]
        mock_chat = unittest.mock.Mock()
        mock_client.chat.create.return_value = mock_chat

        mock_response = unittest.mock.Mock()
        mock_response.usage = unittest.mock.Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.reasoning_tokens = None
        mock_response.citations = None
        mock_response.encrypted_content = None

        mock_tool_call = unittest.mock.Mock()
        mock_tool_call.id = "tool-123"
        mock_tool_call.function = unittest.mock.Mock()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "Paris"}'

        mock_chunk = unittest.mock.Mock()
        mock_chunk.content = None
        mock_chunk.reasoning_content = None
        mock_chunk.tool_calls = [mock_tool_call]

        async def mock_stream():
            yield mock_response, mock_chunk

        mock_chat.stream.return_value = mock_stream()

        events = []
        async for event in model.stream(messages=[]):
            events.append(event)

        # Should have tool_use stop reason (get_tool_call_type is mocked to return "client_side_tool")
        stop_events = [e for e in events if "messageStop" in e]
        assert len(stop_events) == 1
        assert stop_events[0]["messageStop"]["stopReason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_stream_with_reasoning_content(
        self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str
    ) -> None:
        """Test streaming a response with reasoning content."""
        model = xAIModel(model_id=model_id)
        mock_client = mock_xai_sdk_fixture["client"]
        mock_chat = unittest.mock.Mock()
        mock_client.chat.create.return_value = mock_chat

        mock_response = unittest.mock.Mock()
        mock_response.usage = unittest.mock.Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.reasoning_tokens = 20
        mock_response.citations = None
        mock_response.encrypted_content = None  # Explicitly set to None to avoid Mock auto-attribute

        mock_chunk = unittest.mock.Mock()
        mock_chunk.content = None
        mock_chunk.reasoning_content = "Thinking..."
        mock_chunk.tool_calls = None

        async def mock_stream():
            yield mock_response, mock_chunk

        mock_chat.stream.return_value = mock_stream()

        events = []
        async for event in model.stream(messages=[]):
            events.append(event)

        # Should have reasoning content delta with text (not encrypted)
        reasoning_text_events = [
            e
            for e in events
            if "contentBlockDelta" in e
            and "reasoningContent" in e.get("contentBlockDelta", {}).get("delta", {})
            and "text" in e.get("contentBlockDelta", {}).get("delta", {}).get("reasoningContent", {})
        ]
        assert len(reasoning_text_events) == 1
        assert reasoning_text_events[0]["contentBlockDelta"]["delta"]["reasoningContent"]["text"] == "Thinking..."


class TestStructuredOutput:
    """Unit tests for structured_output method."""

    @pytest.mark.asyncio
    async def test_structured_output_basic(
        self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str
    ) -> None:
        """Test structured output with a Pydantic model."""
        import pydantic

        class Weather(pydantic.BaseModel):
            temperature: int
            condition: str

        model = xAIModel(model_id=model_id)
        mock_client = mock_xai_sdk_fixture["client"]
        mock_chat = unittest.mock.Mock()
        mock_client.chat.create.return_value = mock_chat

        parsed_output = Weather(temperature=25, condition="sunny")
        mock_response = unittest.mock.Mock()

        async def mock_parse(output_model):
            return mock_response, parsed_output

        mock_chat.parse = mock_parse

        messages = [{"role": "user", "content": [{"text": "What's the weather?"}]}]
        results = []
        async for result in model.structured_output(Weather, messages):
            results.append(result)

        assert len(results) == 1
        assert results[0]["output"] == parsed_output
        assert results[0]["output"].temperature == 25
        assert results[0]["output"].condition == "sunny"

    @pytest.mark.asyncio
    async def test_structured_output_with_system_prompt(
        self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str
    ) -> None:
        """Test structured output with system prompt."""
        import pydantic

        class Result(pydantic.BaseModel):
            value: str

        model = xAIModel(model_id=model_id)
        mock_client = mock_xai_sdk_fixture["client"]
        mock_chat = unittest.mock.Mock()
        mock_client.chat.create.return_value = mock_chat

        parsed_output = Result(value="test")
        mock_response = unittest.mock.Mock()

        async def mock_parse(output_model):
            return mock_response, parsed_output

        mock_chat.parse = mock_parse
        mock_xai_sdk_fixture["xai_system"].return_value = "system_msg"

        messages = [{"role": "user", "content": [{"text": "Test"}]}]
        results = []
        async for result in model.structured_output(Result, messages, system_prompt="Be helpful"):
            results.append(result)

        mock_xai_sdk_fixture["xai_system"].assert_called_once_with("Be helpful")


class TestServerSideToolCalls:
    """Unit tests for server-side tool call handling."""

    def test_format_metadata_with_server_tool_calls(self, model: xAIModel) -> None:
        """Test formatting metadata chunk with server tool calls."""
        mock_usage = unittest.mock.Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        mock_usage.reasoning_tokens = None

        server_tool_calls = [
            {"id": "tool-1", "name": "x_search", "arguments": '{"query": "test"}'},
            {"id": "tool-2", "name": "web_search", "arguments": '{"query": "hello"}'},
        ]

        result = model._format_chunk(
            {
                "chunk_type": "metadata",
                "data": mock_usage,
                "server_tool_calls": server_tool_calls,
            }
        )

        assert "serverToolCalls" in result["metadata"]
        assert len(result["metadata"]["serverToolCalls"]) == 2
        assert result["metadata"]["serverToolCalls"][0]["name"] == "x_search"
        assert result["metadata"]["serverToolCalls"][1]["name"] == "web_search"

    def test_format_metadata_without_server_tool_calls(self, model: xAIModel) -> None:
        """Test formatting metadata chunk without server tool calls."""
        mock_usage = unittest.mock.Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        mock_usage.reasoning_tokens = None

        result = model._format_chunk(
            {
                "chunk_type": "metadata",
                "data": mock_usage,
            }
        )

        assert "serverToolCalls" not in result["metadata"]

    @pytest.mark.asyncio
    async def test_stream_with_server_side_tool_calls(
        self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str
    ) -> None:
        """Test streaming a response with server-side tool calls (not executed by Strands)."""
        # Override get_tool_call_type to return server_side_tool for this test
        mock_xai_sdk_fixture["get_tool_call_type"].return_value = "server_side_tool"

        model = xAIModel(model_id=model_id)
        mock_client = mock_xai_sdk_fixture["client"]
        mock_chat = unittest.mock.Mock()
        mock_client.chat.create.return_value = mock_chat

        # Mock chat.messages for state capture
        mock_msg = unittest.mock.Mock()
        mock_msg.role = 1  # ROLE_USER
        mock_msg.SerializeToString.return_value = b"mock_serialized"
        mock_chat.messages = [mock_msg]

        mock_response = unittest.mock.Mock()
        mock_response.usage = unittest.mock.Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.reasoning_tokens = None
        mock_response.citations = None
        mock_response.encrypted_content = None

        # Server-side tool call (e.g., x_search)
        mock_tool_call = unittest.mock.Mock()
        mock_tool_call.id = "server-tool-123"
        mock_tool_call.function = unittest.mock.Mock()
        mock_tool_call.function.name = "x_search"
        mock_tool_call.function.arguments = '{"query": "test"}'

        mock_chunk = unittest.mock.Mock()
        mock_chunk.content = "Here are the search results..."
        mock_chunk.reasoning_content = None
        mock_chunk.tool_calls = [mock_tool_call]

        async def mock_stream():
            yield mock_response, mock_chunk

        mock_chat.stream.return_value = mock_stream()

        events = []
        async for event in model.stream(messages=[]):
            events.append(event)

        # Server-side tools should NOT trigger tool_use stop reason
        stop_events = [e for e in events if "messageStop" in e]
        assert len(stop_events) == 1
        assert stop_events[0]["messageStop"]["stopReason"] == "end_turn"

        # Server-side tools should be in metadata
        metadata_events = [e for e in events if "metadata" in e]
        assert len(metadata_events) == 1
        assert "serverToolCalls" in metadata_events[0]["metadata"]
        assert len(metadata_events[0]["metadata"]["serverToolCalls"]) == 1
        assert metadata_events[0]["metadata"]["serverToolCalls"][0]["name"] == "x_search"

    @pytest.mark.asyncio
    async def test_stream_with_mixed_tool_calls(
        self, mock_xai_sdk_fixture: dict[str, unittest.mock.Mock], model_id: str
    ) -> None:
        """Test streaming with both client-side and server-side tool calls."""
        model = xAIModel(model_id=model_id)
        mock_client = mock_xai_sdk_fixture["client"]
        mock_chat = unittest.mock.Mock()
        mock_client.chat.create.return_value = mock_chat

        # Mock chat.messages for state capture
        mock_msg = unittest.mock.Mock()
        mock_msg.role = 1  # ROLE_USER
        mock_msg.SerializeToString.return_value = b"mock_serialized"
        mock_chat.messages = [mock_msg]

        mock_response = unittest.mock.Mock()
        mock_response.usage = unittest.mock.Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.usage.reasoning_tokens = None
        mock_response.citations = None
        mock_response.encrypted_content = None

        # Client-side tool call
        mock_client_tool = unittest.mock.Mock()
        mock_client_tool.id = "client-tool-123"
        mock_client_tool.function = unittest.mock.Mock()
        mock_client_tool.function.name = "get_weather"
        mock_client_tool.function.arguments = '{"city": "Paris"}'

        # Server-side tool call
        mock_server_tool = unittest.mock.Mock()
        mock_server_tool.id = "server-tool-456"
        mock_server_tool.function = unittest.mock.Mock()
        mock_server_tool.function.name = "x_search"
        mock_server_tool.function.arguments = '{"query": "weather"}'

        mock_chunk = unittest.mock.Mock()
        mock_chunk.content = None
        mock_chunk.reasoning_content = None
        mock_chunk.tool_calls = [mock_client_tool, mock_server_tool]

        # Mock get_tool_call_type to return different types based on tool
        def mock_get_type(tool_call):
            if tool_call.function.name == "get_weather":
                return "client_side_tool"
            return "server_side_tool"

        mock_xai_sdk_fixture["get_tool_call_type"].side_effect = mock_get_type

        async def mock_stream():
            yield mock_response, mock_chunk

        mock_chat.stream.return_value = mock_stream()

        events = []
        async for event in model.stream(messages=[]):
            events.append(event)

        # Should have tool_use stop reason (client-side tool present)
        stop_events = [e for e in events if "messageStop" in e]
        assert len(stop_events) == 1
        assert stop_events[0]["messageStop"]["stopReason"] == "tool_use"

        # Should have client-side tool in content blocks
        tool_start_events = [
            e
            for e in events
            if "contentBlockStart" in e and e.get("contentBlockStart", {}).get("start", {}).get("toolUse")
        ]
        assert len(tool_start_events) == 1
        assert tool_start_events[0]["contentBlockStart"]["start"]["toolUse"]["name"] == "get_weather"

        # Server-side tools should be in metadata
        metadata_events = [e for e in events if "metadata" in e]
        assert len(metadata_events) == 1
        assert "serverToolCalls" in metadata_events[0]["metadata"]
        assert len(metadata_events[0]["metadata"]["serverToolCalls"]) == 1
        assert metadata_events[0]["metadata"]["serverToolCalls"][0]["name"] == "x_search"

    def test_format_content_delta_server_tool(self, model: xAIModel) -> None:
        """Test formatting content_delta chunk for server-side tool (inline text)."""
        tool_data = {"id": "tool-123", "name": "x_search", "arguments": '{"query": "test"}'}
        result = model._format_chunk(
            {
                "chunk_type": "content_delta",
                "data_type": "server_tool",
                "data": tool_data,
            }
        )
        assert "contentBlockDelta" in result
        assert "text" in result["contentBlockDelta"]["delta"]
        assert "x_search" in result["contentBlockDelta"]["delta"]["text"]
        assert '{"query": "test"}' in result["contentBlockDelta"]["delta"]["text"]
