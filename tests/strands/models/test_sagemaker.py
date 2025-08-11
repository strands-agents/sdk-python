"""Tests for the Amazon SageMaker model provider."""

import json
import unittest.mock
from typing import Any, Dict, List

import boto3
import pytest
from botocore.config import Config as BotocoreConfig

from strands.models.sagemaker import (
    FunctionCall,
    ModelProvider,
    SageMakerAIModel,
    ToolCall,
    UsageMetadata,
)
from strands.types.content import Messages
from strands.types.tools import ToolSpec


@pytest.fixture
def boto_session():
    """Mock boto3 session."""
    with unittest.mock.patch.object(boto3, "Session") as mock_session:
        yield mock_session.return_value


@pytest.fixture
def sagemaker_client(boto_session):
    """Mock SageMaker runtime client."""
    return boto_session.client.return_value


@pytest.fixture
def endpoint_config() -> Dict[str, Any]:
    """Default endpoint configuration for tests."""
    return {
        "endpoint_name": "test-endpoint",
        "inference_component_name": "test-component",
        "region_name": "us-east-1",
    }


@pytest.fixture
def payload_config() -> Dict[str, Any]:
    """Default payload configuration for tests."""
    return {
        "max_tokens": 1024,
        "temperature": 0.7,
        "stream": True,
    }


@pytest.fixture
def model(boto_session, endpoint_config, payload_config):
    """SageMaker model instance with mocked boto session."""
    return SageMakerAIModel(endpoint_config=endpoint_config, payload_config=payload_config, boto_session=boto_session)


@pytest.fixture
def messages() -> Messages:
    """Sample messages for testing."""
    return [{"role": "user", "content": [{"text": "What is the capital of France?"}]}]


@pytest.fixture
def tool_specs() -> List[ToolSpec]:
    """Sample tool specifications for testing."""
    return [
        {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                }
            },
        }
    ]


@pytest.fixture
def system_prompt() -> str:
    """Sample system prompt for testing."""
    return "You are a helpful assistant."


class TestSageMakerAIModel:
    """Test suite for SageMakerAIModel."""

    def test_init_default(self, boto_session):
        """Test initialization with default parameters."""
        endpoint_config = {"endpoint_name": "test-endpoint", "region_name": "us-east-1"}
        payload_config = {"max_tokens": 1024}
        model = SageMakerAIModel(
            endpoint_config=endpoint_config, payload_config=payload_config, boto_session=boto_session
        )

        assert model.endpoint_config["endpoint_name"] == "test-endpoint"
        assert model.payload_config.get("stream", True) is True

        boto_session.client.assert_called_once_with(
            service_name="sagemaker-runtime",
            config=unittest.mock.ANY,
        )

    def test_init_with_all_params(self, boto_session):
        """Test initialization with all parameters."""
        endpoint_config = {
            "endpoint_name": "test-endpoint",
            "inference_component_name": "test-component",
            "region_name": "us-west-2",
        }
        payload_config = {
            "stream": False,
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        client_config = BotocoreConfig(user_agent_extra="test-agent")

        model = SageMakerAIModel(
            endpoint_config=endpoint_config,
            payload_config=payload_config,
            boto_session=boto_session,
            boto_client_config=client_config,
        )

        assert model.endpoint_config["endpoint_name"] == "test-endpoint"
        assert model.endpoint_config["inference_component_name"] == "test-component"
        assert model.payload_config["stream"] is False
        assert model.payload_config["max_tokens"] == 1024
        assert model.payload_config["temperature"] == 0.7

        boto_session.client.assert_called_once_with(
            service_name="sagemaker-runtime",
            config=unittest.mock.ANY,
        )

    def test_init_with_client_config(self, boto_session):
        """Test initialization with client configuration."""
        endpoint_config = {"endpoint_name": "test-endpoint", "region_name": "us-east-1"}
        payload_config = {"max_tokens": 1024}
        client_config = BotocoreConfig(user_agent_extra="test-agent")

        SageMakerAIModel(
            endpoint_config=endpoint_config,
            payload_config=payload_config,
            boto_session=boto_session,
            boto_client_config=client_config,
        )

        # Verify client was created with a config that includes our user agent
        boto_session.client.assert_called_once_with(
            service_name="sagemaker-runtime",
            config=unittest.mock.ANY,
        )

        # Get the actual config passed to client
        actual_config = boto_session.client.call_args[1]["config"]
        assert "strands-agents" in actual_config.user_agent_extra
        assert "test-agent" in actual_config.user_agent_extra

    def test_update_config(self, model):
        """Test updating model configuration."""
        new_config = {"target_model": "new-model", "target_variant": "new-variant"}
        model.update_config(**new_config)

        assert model.endpoint_config["target_model"] == "new-model"
        assert model.endpoint_config["target_variant"] == "new-variant"
        # Original values should be preserved
        assert model.endpoint_config["endpoint_name"] == "test-endpoint"
        assert model.endpoint_config["inference_component_name"] == "test-component"

    def test_get_config(self, model, endpoint_config):
        """Test getting model configuration."""
        config = model.get_config()
        assert config == model.endpoint_config
        assert isinstance(config, dict)

    # def test_format_request_messages_with_system_prompt(self, model):
    #     """Test formatting request messages with system prompt."""
    #     messages = [{"role": "user", "content": "Hello"}]
    #     system_prompt = "You are a helpful assistant."

    #     formatted_messages = model.format_request_messages(messages, system_prompt)

    #     assert len(formatted_messages) == 2
    #     assert formatted_messages[0]["role"] == "system"
    #     assert formatted_messages[0]["content"] == system_prompt
    #     assert formatted_messages[1]["role"] == "user"
    #     assert formatted_messages[1]["content"] == "Hello"

    # def test_format_request_messages_with_tool_calls(self, model):
    #     """Test formatting request messages with tool calls."""
    #     messages = [
    #         {"role": "user", "content": "Hello"},
    #         {
    #             "role": "assistant",
    #             "content": None,
    #             "tool_calls": [{"id": "123", "type": "function", "function": {"name": "test", "arguments": "{}"}}],
    #         },
    #     ]

    #     formatted_messages = model.format_request_messages(messages, None)

    #     assert len(formatted_messages) == 2
    #     assert formatted_messages[0]["role"] == "user"
    #     assert formatted_messages[1]["role"] == "assistant"
    #     assert "content" not in formatted_messages[1]
    #     assert "tool_calls" in formatted_messages[1]

    # def test_format_request(self, model, messages, tool_specs, system_prompt):
    #     """Test formatting a request with all parameters."""
    #     request = model.format_request(messages, tool_specs, system_prompt)

    #     assert request["EndpointName"] == "test-endpoint"
    #     assert request["InferenceComponentName"] == "test-component"
    #     assert request["ContentType"] == "application/json"
    #     assert request["Accept"] == "application/json"

    #     payload = json.loads(request["Body"])
    #     assert "messages" in payload
    #     assert len(payload["messages"]) > 0
    #     assert "tools" in payload
    #     assert len(payload["tools"]) == 1
    #     assert payload["tools"][0]["type"] == "function"
    #     assert payload["tools"][0]["function"]["name"] == "get_weather"
    #     assert payload["max_tokens"] == 1024
    #     assert payload["temperature"] == 0.7
    #     assert payload["stream"] is True

    # def test_format_request_without_tools(self, model, messages, system_prompt):
    #     """Test formatting a request without tools."""
    #     request = model.format_request(messages, None, system_prompt)

    #     payload = json.loads(request["Body"])
    #     assert "tools" in payload
    #     assert payload["tools"] == []

    @pytest.mark.asyncio
    async def test_stream_with_streaming_enabled(self, sagemaker_client, model, messages):
        """Test streaming response with streaming enabled."""
        # Mock the response from SageMaker
        mock_response = {
            "Body": [
                {
                    "PayloadPart": {
                        "Bytes": json.dumps(
                            {
                                "choices": [
                                    {
                                        "delta": {"content": "Paris is the capital of France."},
                                        "finish_reason": None,
                                    }
                                ]
                            }
                        ).encode("utf-8")
                    }
                },
                {
                    "PayloadPart": {
                        "Bytes": json.dumps(
                            {
                                "choices": [
                                    {
                                        "delta": {"content": " It is known for the Eiffel Tower."},
                                        "finish_reason": "stop",
                                    }
                                ]
                            }
                        ).encode("utf-8")
                    }
                },
            ]
        }
        sagemaker_client.invoke_endpoint_with_response_stream.return_value = mock_response

        response = [chunk async for chunk in model.stream(messages)]

        assert len(response) >= 5
        assert response[0] == {"messageStart": {"role": "assistant"}}

        # Find content events
        content_start = next((e for e in response if "contentBlockStart" in e), None)
        content_delta = next((e for e in response if "contentBlockDelta" in e), None)
        content_stop = next((e for e in response if "contentBlockStop" in e), None)
        message_stop = next((e for e in response if "messageStop" in e), None)

        assert content_start is not None
        assert content_delta is not None
        assert content_stop is not None
        assert message_stop is not None
        assert message_stop["messageStop"]["stopReason"] == "end_turn"

        sagemaker_client.invoke_endpoint_with_response_stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(self, sagemaker_client, model, messages):
        """Test streaming response with tool calls."""
        # Mock the response from SageMaker with tool calls
        mock_response = {
            "Body": [
                {
                    "PayloadPart": {
                        "Bytes": json.dumps(
                            {
                                "choices": [
                                    {
                                        "delta": {
                                            "content": None,
                                            "tool_calls": [
                                                {
                                                    "index": 0,
                                                    "id": "tool123",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "get_weather",
                                                        "arguments": '{"location": "Paris"}',
                                                    },
                                                }
                                            ],
                                        },
                                        "finish_reason": "tool_calls",
                                    }
                                ]
                            }
                        ).encode("utf-8")
                    }
                }
            ]
        }
        sagemaker_client.invoke_endpoint_with_response_stream.return_value = mock_response

        response = [chunk async for chunk in model.stream(messages)]

        # Verify the response contains tool call events
        assert len(response) >= 4
        assert response[0] == {"messageStart": {"role": "assistant"}}

        message_stop = next((e for e in response if "messageStop" in e), None)
        assert message_stop is not None
        assert message_stop["messageStop"]["stopReason"] == "tool_use"

        # Find tool call events
        tool_start = next(
            (
                e
                for e in response
                if "contentBlockStart" in e and e.get("contentBlockStart", {}).get("start", {}).get("toolUse")
            ),
            None,
        )
        tool_delta = next(
            (
                e
                for e in response
                if "contentBlockDelta" in e and e.get("contentBlockDelta", {}).get("delta", {}).get("toolUse")
            ),
            None,
        )
        tool_stop = next((e for e in response if "contentBlockStop" in e), None)

        assert tool_start is not None
        assert tool_delta is not None
        assert tool_stop is not None

        # Verify tool call data
        tool_use_data = tool_start["contentBlockStart"]["start"]["toolUse"]
        assert tool_use_data["toolUseId"] == "tool123"
        assert tool_use_data["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_stream_with_partial_json(self, sagemaker_client, model, messages):
        """Test streaming response with partial JSON chunks."""
        # Mock the response from SageMaker with split JSON
        mock_response = {
            "Body": [
                {"PayloadPart": {"Bytes": '{"choices": [{"delta": {"content": "Paris is'.encode("utf-8")}},
                {"PayloadPart": {"Bytes": ' the capital of France."}, "finish_reason": "stop"}]}'.encode("utf-8")}},
            ]
        }
        sagemaker_client.invoke_endpoint_with_response_stream.return_value = mock_response

        response = [chunk async for chunk in model.stream(messages)]

        assert len(response) == 5
        assert response[0] == {"messageStart": {"role": "assistant"}}

        # Find content events
        content_start = next((e for e in response if "contentBlockStart" in e), None)
        content_delta = next((e for e in response if "contentBlockDelta" in e), None)
        content_stop = next((e for e in response if "contentBlockStop" in e), None)
        message_stop = next((e for e in response if "messageStop" in e), None)

        assert content_start is not None
        assert content_delta is not None
        assert content_stop is not None
        assert message_stop is not None
        assert message_stop["messageStop"]["stopReason"] == "end_turn"

        # Verify content
        text_delta = content_delta["contentBlockDelta"]["delta"]["text"]
        assert text_delta == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_stream_non_streaming(self, sagemaker_client, model, messages):
        """Test non-streaming response."""
        # Configure model for non-streaming
        model.payload_config["stream"] = False

        # Mock the response from SageMaker
        mock_response = {"Body": unittest.mock.MagicMock()}
        mock_response["Body"].read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "message": {"content": "Paris is the capital of France.", "tool_calls": None},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "prompt_tokens_details": 0},
            }
        ).encode("utf-8")

        sagemaker_client.invoke_endpoint.return_value = mock_response

        response = [chunk async for chunk in model.stream(messages)]

        assert len(response) >= 6
        assert response[0] == {"messageStart": {"role": "assistant"}}

        # Find content events
        content_start = next((e for e in response if "contentBlockStart" in e), None)
        content_delta = next((e for e in response if "contentBlockDelta" in e), None)
        content_stop = next((e for e in response if "contentBlockStop" in e), None)
        message_stop = next((e for e in response if "messageStop" in e), None)

        assert content_start is not None
        assert content_delta is not None
        assert content_stop is not None
        assert message_stop is not None

        # Verify content
        text_delta = content_delta["contentBlockDelta"]["delta"]["text"]
        assert text_delta == "Paris is the capital of France."

        sagemaker_client.invoke_endpoint.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_non_streaming_with_tool_calls(self, sagemaker_client, model, messages):
        """Test non-streaming response with tool calls."""
        # Configure model for non-streaming
        model.payload_config["stream"] = False

        # Mock the response from SageMaker with tool calls
        mock_response = {"Body": unittest.mock.MagicMock()}
        mock_response["Body"].read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "tool123",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "prompt_tokens_details": 0},
            }
        ).encode("utf-8")

        sagemaker_client.invoke_endpoint.return_value = mock_response

        response = [chunk async for chunk in model.stream(messages)]

        # Verify basic structure
        assert len(response) >= 6
        assert response[0] == {"messageStart": {"role": "assistant"}}

        # Find tool call events
        tool_start = next(
            (
                e
                for e in response
                if "contentBlockStart" in e and e.get("contentBlockStart", {}).get("start", {}).get("toolUse")
            ),
            None,
        )
        tool_delta = next(
            (
                e
                for e in response
                if "contentBlockDelta" in e and e.get("contentBlockDelta", {}).get("delta", {}).get("toolUse")
            ),
            None,
        )
        tool_stop = next((e for e in response if "contentBlockStop" in e), None)
        message_stop = next((e for e in response if "messageStop" in e), None)

        assert tool_start is not None
        assert tool_delta is not None
        assert tool_stop is not None
        assert message_stop is not None

        # Verify tool call data
        tool_use_data = tool_start["contentBlockStart"]["start"]["toolUse"]
        assert tool_use_data["toolUseId"] == "tool123"
        assert tool_use_data["name"] == "get_weather"

        # Verify metadata
        metadata = next((e for e in response if "metadata" in e), None)
        assert metadata is not None
        usage_data = metadata["metadata"]["usage"]
        assert usage_data["totalTokens"] == 30


class TestModelProvider:
    """Test suite for model provider functionality."""

    def test_default_model_provider(self, boto_session):
        """Test that default model provider is set to OpenAI."""
        endpoint_config = {"endpoint_name": "test-endpoint", "region_name": "us-east-1"}
        payload_config = {"max_tokens": 1024}

        model = SageMakerAIModel(
            endpoint_config=endpoint_config, payload_config=payload_config, boto_session=boto_session
        )

        assert model.endpoint_config["model_provider"] == ModelProvider.OPENAI

    def test_model_provider_enum(self, boto_session):
        """Test setting model provider using enum."""
        endpoint_config = {
            "endpoint_name": "test-endpoint",
            "region_name": "us-east-1",
            "model_provider": ModelProvider.LLAMA,
        }
        payload_config = {"max_tokens": 1024}

        model = SageMakerAIModel(
            endpoint_config=endpoint_config, payload_config=payload_config, boto_session=boto_session
        )

        assert model.endpoint_config["model_provider"] == ModelProvider.LLAMA

    def test_model_provider_string(self, boto_session):
        """Test setting model provider using string."""
        endpoint_config = {"endpoint_name": "test-endpoint", "region_name": "us-east-1", "model_provider": "mistral"}
        payload_config = {"max_tokens": 1024}

        model = SageMakerAIModel(
            endpoint_config=endpoint_config, payload_config=payload_config, boto_session=boto_session
        )

        assert model.endpoint_config["model_provider"] == ModelProvider.MISTRAL

    def test_custom_formatter_required(self, boto_session):
        """Test that custom formatter is required when using CUSTOM provider."""
        endpoint_config = {
            "endpoint_name": "test-endpoint",
            "region_name": "us-east-1",
            "model_provider": ModelProvider.CUSTOM,
        }
        payload_config = {"max_tokens": 1024}

        with pytest.raises(ValueError, match="custom_formatter is required when model_provider is CUSTOM"):
            SageMakerAIModel(endpoint_config=endpoint_config, payload_config=payload_config, boto_session=boto_session)

    def test_custom_formatter_with_function(self, boto_session):
        """Test using custom formatter with a function."""

        def custom_formatter(messages, system_prompt=None):
            formatted = []
            if system_prompt:
                formatted.append({"role": "system", "text": system_prompt})
            for msg in messages:
                formatted.append({"role": msg["role"], "text": msg["content"][0]["text"]})
            return formatted

        endpoint_config = {
            "endpoint_name": "test-endpoint",
            "region_name": "us-east-1",
            "model_provider": ModelProvider.CUSTOM,
            "custom_formatter": custom_formatter,
        }
        payload_config = {"max_tokens": 1024}

        model = SageMakerAIModel(
            endpoint_config=endpoint_config, payload_config=payload_config, boto_session=boto_session
        )

        assert model.endpoint_config["model_provider"] == ModelProvider.CUSTOM
        assert model.endpoint_config["custom_formatter"] == custom_formatter

    def test_format_messages_openai_provider(self, boto_session, messages, system_prompt):
        """Test message formatting with OpenAI provider."""
        endpoint_config = {
            "endpoint_name": "test-endpoint",
            "region_name": "us-east-1",
            "model_provider": ModelProvider.OPENAI,
        }
        payload_config = {"max_tokens": 1024}

        model = SageMakerAIModel(
            endpoint_config=endpoint_config, payload_config=payload_config, boto_session=boto_session
        )

        formatted = model._format_messages_for_provider(messages, system_prompt)

        assert len(formatted) == 2  # system + user message
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"] == system_prompt
        assert formatted[1]["role"] == "user"
        assert len(formatted[1]["content"]) == 1
        assert formatted[1]["content"][0]["text"] == "What is the capital of France?"

    def test_format_messages_llama_provider(self, boto_session, messages, system_prompt):
        """Test message formatting with Llama provider."""
        endpoint_config = {
            "endpoint_name": "test-endpoint",
            "region_name": "us-east-1",
            "model_provider": ModelProvider.LLAMA,
        }
        payload_config = {"max_tokens": 1024}

        model = SageMakerAIModel(
            endpoint_config=endpoint_config, payload_config=payload_config, boto_session=boto_session
        )

        formatted = model._format_messages_for_provider(messages, system_prompt)

        assert len(formatted) == 2  # system + user message
        assert formatted[0]["role"] == "system"
        assert formatted[0]["content"] == system_prompt
        assert formatted[1]["role"] == "user"
        assert isinstance(formatted[1]["content"], str)  # Llama uses string content
        assert "What is the capital of France?" in formatted[1]["content"]

    def test_format_messages_custom_provider(self, boto_session, messages, system_prompt):
        """Test message formatting with custom provider."""

        def custom_formatter(msgs, sys_prompt=None):
            result = [{"custom_format": True}]
            if sys_prompt:
                result.append({"system": sys_prompt})
            for msg in msgs:
                result.append({"speaker": msg["role"], "text": msg["content"][0]["text"]})
            return result

        endpoint_config = {
            "endpoint_name": "test-endpoint",
            "region_name": "us-east-1",
            "model_provider": ModelProvider.CUSTOM,
            "custom_formatter": custom_formatter,
        }
        payload_config = {"max_tokens": 1024}

        model = SageMakerAIModel(
            endpoint_config=endpoint_config, payload_config=payload_config, boto_session=boto_session
        )

        formatted = model._format_messages_for_provider(messages, system_prompt)

        assert len(formatted) == 3  # custom marker + system + user
        assert formatted[0]["custom_format"] is True
        assert formatted[1]["system"] == system_prompt
        assert formatted[2]["speaker"] == "user"
        assert formatted[2]["text"] == "What is the capital of France?"

    def test_format_messages_with_tool_use_llama(self, boto_session, system_prompt):
        """Test Llama formatting with tool use messages."""
        messages_with_tools = [
            {"role": "user", "content": [{"text": "What's the weather?"}]},
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "tool123", "name": "get_weather", "input": {"location": "Paris"}}}
                ],
            },
            {
                "role": "user",
                "content": [{"toolResult": {"toolUseId": "tool123", "content": [{"text": "Sunny, 25°C"}]}}],
            },
        ]

        endpoint_config = {
            "endpoint_name": "test-endpoint",
            "region_name": "us-east-1",
            "model_provider": ModelProvider.LLAMA,
        }
        payload_config = {"max_tokens": 1024}

        model = SageMakerAIModel(
            endpoint_config=endpoint_config, payload_config=payload_config, boto_session=boto_session
        )

        formatted = model._format_messages_for_provider(messages_with_tools, system_prompt)

        assert len(formatted) == 4  # system + 3 messages
        # Check that tool calls are formatted as strings for Llama
        assistant_msg = formatted[2]
        assert isinstance(assistant_msg["content"], str)
        assert "TOOL_CALL" in assistant_msg["content"]
        assert "get_weather" in assistant_msg["content"]

        # Check that tool results are formatted as strings
        tool_result_msg = formatted[3]
        assert isinstance(tool_result_msg["content"], str)
        assert "TOOL_RESULT" in tool_result_msg["content"]
        assert "Sunny, 25°C" in tool_result_msg["content"]

    def test_format_messages_anthropic_with_images(self, boto_session):
        """Test Anthropic formatting with image content."""
        import base64

        # Create mock image data
        image_bytes = b"fake_image_data"
        messages_with_image = [
            {
                "role": "user",
                "content": [
                    {"text": "What's in this image?"},
                    {"image": {"format": "png", "source": {"bytes": image_bytes}}},
                ],
            }
        ]

        endpoint_config = {
            "endpoint_name": "test-endpoint",
            "region_name": "us-east-1",
            "model_provider": ModelProvider.ANTHROPIC,
        }
        payload_config = {"max_tokens": 1024}

        model = SageMakerAIModel(
            endpoint_config=endpoint_config, payload_config=payload_config, boto_session=boto_session
        )

        formatted = model._format_messages_for_provider(messages_with_image)

        assert len(formatted) == 1
        user_msg = formatted[0]
        assert user_msg["role"] == "user"
        assert len(user_msg["content"]) == 2

        # Check text content
        text_content = user_msg["content"][0]
        assert text_content["type"] == "text"
        assert text_content["text"] == "What's in this image?"

        # Check image content
        image_content = user_msg["content"][1]
        assert image_content["type"] == "image"
        assert image_content["source"]["type"] == "base64"
        assert image_content["source"]["media_type"] == "image/png"
        assert image_content["source"]["data"] == base64.b64encode(image_bytes).decode("utf-8")


class TestDataClasses:
    """Test suite for data classes."""

    def test_usage_metadata(self):
        """Test UsageMetadata dataclass."""
        usage = UsageMetadata(total_tokens=100, completion_tokens=30, prompt_tokens=70, prompt_tokens_details=5)

        assert usage.total_tokens == 100
        assert usage.completion_tokens == 30
        assert usage.prompt_tokens == 70
        assert usage.prompt_tokens_details == 5

    def test_function_call(self):
        """Test FunctionCall dataclass."""
        func = FunctionCall(name="get_weather", arguments='{"location": "Paris"}')

        assert func.name == "get_weather"
        assert func.arguments == '{"location": "Paris"}'

        # Test initialization with kwargs
        func2 = FunctionCall(**{"name": "get_time", "arguments": '{"timezone": "UTC"}'})

        assert func2.name == "get_time"
        assert func2.arguments == '{"timezone": "UTC"}'

    def test_tool_call(self):
        """Test ToolCall dataclass."""
        # Create a tool call using kwargs directly
        tool = ToolCall(
            id="tool123", type="function", function={"name": "get_weather", "arguments": '{"location": "Paris"}'}
        )

        assert tool.id == "tool123"
        assert tool.type == "function"
        assert tool.function.name == "get_weather"
        assert tool.function.arguments == '{"location": "Paris"}'

        # Test initialization with kwargs
        tool2 = ToolCall(
            **{
                "id": "tool456",
                "type": "function",
                "function": {"name": "get_time", "arguments": '{"timezone": "UTC"}'},
            }
        )

        assert tool2.id == "tool456"
        assert tool2.type == "function"
        assert tool2.function.name == "get_time"
        assert tool2.function.arguments == '{"timezone": "UTC"}'


class TestModelProviderEnum:
    """Test suite for ModelProvider enum."""

    def test_model_provider_values(self):
        """Test ModelProvider enum values."""
        assert ModelProvider.OPENAI.value == "openai"
        assert ModelProvider.MISTRAL.value == "mistral"
        assert ModelProvider.LLAMA.value == "llama"
        assert ModelProvider.ANTHROPIC.value == "anthropic"
        assert ModelProvider.CUSTOM.value == "custom"

    def test_model_provider_from_string(self):
        """Test creating ModelProvider from string."""
        assert ModelProvider("openai") == ModelProvider.OPENAI
        assert ModelProvider("mistral") == ModelProvider.MISTRAL
        assert ModelProvider("llama") == ModelProvider.LLAMA
        assert ModelProvider("anthropic") == ModelProvider.ANTHROPIC
        assert ModelProvider("custom") == ModelProvider.CUSTOM

    def test_invalid_model_provider(self):
        """Test invalid model provider string raises ValueError."""
        with pytest.raises(ValueError):
            ModelProvider("invalid_provider")
