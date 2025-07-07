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
def model_config() -> Dict[str, Any]:
    """Default model configuration for tests."""
    return {
        "endpoint_name": "test-endpoint",
        "inference_component_name": "test-component",
        "stream": True,
        "max_tokens": 1024,
        "temperature": 0.7,
        "additional_args": {"top_p": 0.9},
    }


@pytest.fixture
def model(boto_session, model_config):
    """SageMaker model instance with mocked boto session."""
    return SageMakerAIModel(model_config=model_config, boto_session=boto_session)


@pytest.fixture
def messages() -> Messages:
    """Sample messages for testing."""
    return [{"role": "user", "content": "What is the capital of France?"}]


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
        model_config = {"endpoint_name": "test-endpoint"}
        model = SageMakerAIModel(model_config=model_config, boto_session=boto_session)

        assert model.config["endpoint_name"] == "test-endpoint"
        assert model.config.get("stream", True) is True

        boto_session.client.assert_called_once_with(
            service_name="sagemaker-runtime",
            config=unittest.mock.ANY,
        )

    def test_init_with_all_params(self, boto_session):
        """Test initialization with all parameters."""
        model_config = {
            "endpoint_name": "test-endpoint",
            "inference_component_name": "test-component",
            "stream": False,
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        region_name = "us-west-2"
        client_config = BotocoreConfig(user_agent_extra="test-agent")

        model = SageMakerAIModel(
            model_config=model_config,
            boto_session=boto_session,
            boto_client_config=client_config,
            region_name=region_name,
        )

        assert model.config["endpoint_name"] == "test-endpoint"
        assert model.config["inference_component_name"] == "test-component"
        assert model.config["stream"] is False
        assert model.config["max_tokens"] == 1024
        assert model.config["temperature"] == 0.7

        boto_session.client.assert_called_once_with(
            service_name="sagemaker-runtime",
            config=unittest.mock.ANY,
        )

    def test_init_with_client_config(self, boto_session):
        """Test initialization with client configuration."""
        model_config = {"endpoint_name": "test-endpoint"}
        client_config = BotocoreConfig(user_agent_extra="test-agent")

        SageMakerAIModel(
            model_config=model_config,
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
        new_config = {"temperature": 0.5, "top_p": 0.9}
        model.update_config(**new_config)

        assert model.config["temperature"] == 0.5
        assert model.config["top_p"] == 0.9
        # Original values should be preserved
        assert model.config["endpoint_name"] == "test-endpoint"
        assert model.config["inference_component_name"] == "test-component"

    def test_get_config(self, model, model_config):
        """Test getting model configuration."""
        config = model.get_config()
        assert config == model.config
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
    async def test_stream_with_streaming_enabled(self, sagemaker_client, model):
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

        request = {
            "EndpointName": "test-endpoint",
            "Body": "{}",
            "ContentType": "application/json",
            "Accept": "application/json",
        }

        response = [chunk async for chunk in model.stream(request)]

        assert len(response) >= 5
        assert response[0] == {"chunk_type": "message_start"}
        assert response[1] == {"chunk_type": "content_start", "data_type": "text"}
        assert response[-2] == {"chunk_type": "content_stop", "data_type": "text"}
        assert response[-1] == {"chunk_type": "message_stop", "data": "stop"}

        sagemaker_client.invoke_endpoint_with_response_stream.assert_called_once_with(**request)

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(self, sagemaker_client, model):
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

        request = {
            "EndpointName": "test-endpoint",
            "Body": "{}",
            "ContentType": "application/json",
            "Accept": "application/json",
        }

        response = [chunk async for chunk in model.stream(request)]

        # Verify the response contains tool call events
        assert len(response) >= 4
        assert response[0] == {"chunk_type": "message_start"}
        assert response[-1] == {"chunk_type": "message_stop", "data": "tool_calls"}

        # Find tool call events
        tool_start = next(
            (e for e in response if e.get("chunk_type") == "content_start" and e.get("data_type") == "tool"), None
        )
        tool_delta = next(
            (e for e in response if e.get("chunk_type") == "content_delta" and e.get("data_type") == "tool"), None
        )
        tool_stop = next(
            (e for e in response if e.get("chunk_type") == "content_stop" and e.get("data_type") == "tool"), None
        )

        assert tool_start is not None
        assert tool_delta is not None
        assert tool_stop is not None
        assert tool_delta["data"].id == "tool123"
        assert tool_delta["data"].function.name == "get_weather"
        assert tool_delta["data"].function.arguments == '{"location": "Paris"}'

    @pytest.mark.asyncio
    async def test_stream_with_partial_json(self, sagemaker_client, model):
        """Test streaming response with partial JSON chunks."""
        # Mock the response from SageMaker with split JSON
        mock_response = {
            "Body": [
                {"PayloadPart": {"Bytes": '{"choices": [{"delta": {"content": "Paris is'.encode("utf-8")}},
                {"PayloadPart": {"Bytes": ' the capital of France."}, "finish_reason": "stop"}]}'.encode("utf-8")}},
            ]
        }
        sagemaker_client.invoke_endpoint_with_response_stream.return_value = mock_response

        request = {
            "EndpointName": "test-endpoint",
            "Body": "{}",
            "ContentType": "application/json",
            "Accept": "application/json",
        }

        response = [chunk async for chunk in model.stream(request)]

        assert len(response) == 5
        assert response[0] == {"chunk_type": "message_start"}
        assert response[1] == {"chunk_type": "content_start", "data_type": "text"}
        assert response[2] == {
            "chunk_type": "content_delta",
            "data_type": "text",
            "data": "Paris is the capital of France.",
        }
        assert response[3] == {"chunk_type": "content_stop", "data_type": "text"}
        assert response[4] == {"chunk_type": "message_stop", "data": "stop"}

    @pytest.mark.asyncio
    async def test_stream_non_streaming(self, sagemaker_client, model):
        """Test non-streaming response."""
        # Configure model for non-streaming
        model.config["stream"] = False

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

        request = {
            "EndpointName": "test-endpoint",
            "Body": "{}",
            "ContentType": "application/json",
            "Accept": "application/json",
        }

        response = [chunk async for chunk in model.stream(request)]

        assert len(response) >= 6
        assert response[0] == {"chunk_type": "message_start"}
        assert response[1] == {"chunk_type": "content_start", "data_type": "text"}
        assert response[2] == {
            "chunk_type": "content_delta",
            "data_type": "text",
            "data": "Paris is the capital of France.",
        }

        sagemaker_client.invoke_endpoint.assert_called_once_with(**request)

    @pytest.mark.asyncio
    async def test_stream_non_streaming_with_tool_calls(self, sagemaker_client, model):
        """Test non-streaming response with tool calls."""
        # Configure model for non-streaming
        model.config["stream"] = False

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

        request = {
            "EndpointName": "test-endpoint",
            "Body": "{}",
            "ContentType": "application/json",
            "Accept": "application/json",
        }

        response = [chunk async for chunk in model.stream(request)]

        # Verify basic structure
        assert len(response) >= 7
        assert response[0] == {"chunk_type": "message_start"}

        # Find tool call events
        tool_events = [e for e in response if e.get("data_type") == "tool"]
        assert len(tool_events) >= 3  # start, delta, stop

        # Verify tool call data
        tool_data = next((e["data"] for e in tool_events if e.get("chunk_type") == "content_delta"), None)
        assert tool_data is not None
        assert isinstance(tool_data, ToolCall)
        assert tool_data.id == "tool123"
        assert tool_data.type == "function"
        assert tool_data.function.name == "get_weather"
        assert tool_data.function.arguments == '{"location": "Paris"}'

        # Verify metadata
        metadata = next((e for e in response if e.get("chunk_type") == "metadata"), None)
        assert metadata is not None
        assert isinstance(metadata["data"], UsageMetadata)
        assert metadata["data"].total_tokens == 30


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
