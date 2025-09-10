"""Tests for the Amazon SageMaker model provider."""

import json
import unittest.mock
from typing import Any, Dict, List

import boto3
import pytest
from botocore.config import Config as BotocoreConfig

from strands.models.sagemaker import (
    FunctionCall,
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
    async def test_stream_with_partial_json(self, sagemaker_client, model, messages, captured_warnings):
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

        # Ensure no warnings emitted
        assert len(captured_warnings) == 0

    @pytest.mark.asyncio
    async def test_tool_choice_not_supported_warns(self, sagemaker_client, model, messages, captured_warnings, alist):
        """Test that non-None toolChoice emits warning for unsupported providers."""
        tool_choice = {"auto": {}}

        """Test streaming response with partial JSON chunks."""
        # Mock the response from SageMaker with split JSON
        mock_response = {
            "Body": [
                {"PayloadPart": {"Bytes": '{"choices": [{"delta": {"content": "Paris is'.encode("utf-8")}},
                {"PayloadPart": {"Bytes": ' the capital of France."}, "finish_reason": "stop"}]}'.encode("utf-8")}},
            ]
        }
        sagemaker_client.invoke_endpoint_with_response_stream.return_value = mock_response

        await alist(model.stream(messages, tool_choice=tool_choice))

        # Ensure toolChoice parameter warning
        assert len(captured_warnings) == 1
        assert "ToolChoice was provided to this provider but is not supported" in str(captured_warnings[0].message)

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


def test_config_validation_warns_on_unknown_keys_in_endpoint(boto_session, captured_warnings):
    """Test that unknown config keys emit a warning."""
    endpoint_config = {"endpoint_name": "test-endpoint", "region_name": "us-east-1", "invalid_param": "test"}
    payload_config = {"max_tokens": 1024}

    SageMakerAIModel(
        endpoint_config=endpoint_config,
        payload_config=payload_config,
        boto_session=boto_session,
    )

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "invalid_param" in str(captured_warnings[0].message)


def test_config_validation_warns_on_unknown_keys_in_payload(boto_session, captured_warnings):
    """Test that unknown config keys emit a warning."""
    endpoint_config = {"endpoint_name": "test-endpoint", "region_name": "us-east-1"}
    payload_config = {"max_tokens": 1024, "invalid_param": "test"}

    SageMakerAIModel(
        endpoint_config=endpoint_config,
        payload_config=payload_config,
        boto_session=boto_session,
    )

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "invalid_param" in str(captured_warnings[0].message)


def test_update_config_validation_warns_on_unknown_keys(model, captured_warnings):
    """Test that update_config warns on unknown keys."""
    model.update_config(wrong_param="test")

    assert len(captured_warnings) == 1
    assert "Invalid configuration parameters" in str(captured_warnings[0].message)
    assert "wrong_param" in str(captured_warnings[0].message)
