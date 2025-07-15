"""Integration tests for Baseten model provider."""

import os
import pytest
from typing import AsyncGenerator

from strands.models.baseten import BasetenModel
from strands.types.content import Messages
from strands.types.streaming import StreamEvent


@pytest.fixture
def baseten_model_apis():
    """Create a BasetenModel instance for Model APIs testing."""
    return BasetenModel(
        model_id="deepseek-ai/DeepSeek-R1-0528",
        client_args={
            "api_key": os.getenv("BASETEN_API_KEY"),
        },
    )


@pytest.fixture
def baseten_dedicated_deployment():
    """Create a BasetenModel instance for dedicated deployment testing."""
    # This would need a real base URL
    base_url = os.getenv("BASETEN_BASE_URL", "https://model-test-deployment.api.baseten.co/environments/production/sync/v1")
    
    return BasetenModel(
        model_id="test-deployment",
        base_url=base_url,
        client_args={
            "api_key": os.getenv("BASETEN_API_KEY"),
        },
    )


@pytest.mark.asyncio
async def test_baseten_model_apis_streaming(baseten_model_apis):
    """Test streaming with Baseten Model APIs."""
    if not os.getenv("BASETEN_API_KEY"):
        pytest.skip("BASETEN_API_KEY not set")

    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello, how are you?"}]}
    ]

    events = []
    async for event in baseten_model_apis.stream(messages):
        events.append(event)

    # Verify we get the expected event types
    assert len(events) > 0
    assert events[0]["messageStart"]["role"] == "assistant"
    
    # Check for content events
    content_events = [e for e in events if "contentBlockDelta" in e]
    assert len(content_events) > 0
    
    # Check for message stop
    stop_events = [e for e in events if "messageStop" in e]
    assert len(stop_events) > 0


@pytest.mark.asyncio
async def test_baseten_model_apis_with_system_prompt(baseten_model_apis):
    """Test streaming with Baseten Model APIs and system prompt."""
    if not os.getenv("BASETEN_API_KEY"):
        pytest.skip("BASETEN_API_KEY not set")

    messages: Messages = [
        {"role": "user", "content": [{"text": "What is 2+2?"}]}
    ]
    system_prompt = "You are a helpful math assistant. Always provide clear explanations."

    events = []
    async for event in baseten_model_apis.stream(messages, system_prompt=system_prompt):
        events.append(event)

    assert len(events) > 0
    assert events[0]["messageStart"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_baseten_model_apis_structured_output(baseten_model_apis):
    """Test structured output with Baseten Model APIs."""
    if not os.getenv("BASETEN_API_KEY"):
        pytest.skip("BASETEN_API_KEY not set")

    from pydantic import BaseModel

    class MathResult(BaseModel):
        answer: int
        explanation: str

    messages: Messages = [
        {"role": "user", "content": [{"text": "What is 5 + 3? Provide the answer as a number and explain your reasoning."}]}
    ]

    results = []
    async for result in baseten_model_apis.structured_output(MathResult, messages):
        results.append(result)

    assert len(results) == 1
    assert "output" in results[0]
    assert isinstance(results[0]["output"], MathResult)
    assert results[0]["output"].answer == 8


@pytest.mark.asyncio
async def test_baseten_model_apis_with_tools(baseten_model_apis):
    """Test streaming with Baseten Model APIs and tools."""
    if not os.getenv("BASETEN_API_KEY"):
        pytest.skip("BASETEN_API_KEY not set")

    messages: Messages = [
        {"role": "user", "content": [{"text": "Calculate 10 + 20"}]}
    ]
    
    tool_specs = [
        {
            "name": "calculator",
            "description": "A simple calculator that can perform basic arithmetic",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]

    events = []
    async for event in baseten_model_apis.stream(messages, tool_specs=tool_specs):
        events.append(event)

    assert len(events) > 0
    assert events[0]["messageStart"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_baseten_model_apis_complex_messages(baseten_model_apis):
    """Test streaming with complex message structures."""
    if not os.getenv("BASETEN_API_KEY"):
        pytest.skip("BASETEN_API_KEY not set")

    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there! How can I help you today?"}]},
        {"role": "user", "content": [{"text": "What's the weather like?"}]}
    ]

    events = []
    async for event in baseten_model_apis.stream(messages):
        events.append(event)

    assert len(events) > 0
    assert events[0]["messageStart"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_baseten_dedicated_deployment_streaming(baseten_dedicated_deployment):
    """Test streaming with Baseten dedicated deployment."""
    if not os.getenv("BASETEN_API_KEY") or not os.getenv("BASETEN_BASE_URL"):
        pytest.skip("BASETEN_API_KEY or BASETEN_BASE_URL not set")

    messages: Messages = [
        {"role": "user", "content": [{"text": "Hello from dedicated deployment!"}]}
    ]

    events = []
    async for event in baseten_dedicated_deployment.stream(messages):
        events.append(event)

    assert len(events) > 0
    assert events[0]["messageStart"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_baseten_dedicated_deployment_structured_output(baseten_dedicated_deployment):
    """Test structured output with Baseten dedicated deployment."""
    if not os.getenv("BASETEN_API_KEY") or not os.getenv("BASETEN_BASE_URL"):
        pytest.skip("BASETEN_API_KEY or BASETEN_BASE_URL not set")

    from pydantic import BaseModel

    class SimpleResponse(BaseModel):
        message: str

    messages: Messages = [
        {"role": "user", "content": [{"text": "Say hello"}]}
    ]

    results = []
    async for result in baseten_dedicated_deployment.structured_output(SimpleResponse, messages):
        results.append(result)

    assert len(results) == 1
    assert "output" in results[0]
    assert isinstance(results[0]["output"], SimpleResponse)


def test_baseten_config_management():
    """Test configuration management for BasetenModel."""
    model = BasetenModel(
        model_id="test-model",
        params={"max_tokens": 100, "temperature": 0.7}
    )

    # Test initial config
    config = model.get_config()
    assert config["model_id"] == "test-model"
    assert config["params"]["max_tokens"] == 100
    assert config["params"]["temperature"] == 0.7

    # Test config update
    model.update_config(params={"max_tokens": 200, "temperature": 0.5})
    updated_config = model.get_config()
    assert updated_config["params"]["max_tokens"] == 200
    assert updated_config["params"]["temperature"] == 0.5


def test_baseten_model_apis_configuration():
    """Test Model APIs configuration."""
    model = BasetenModel(
        model_id="deepseek-ai/DeepSeek-R1-0528",
        client_args={"api_key": "test-key"}
    )

    config = model.get_config()
    assert config["model_id"] == "deepseek-ai/DeepSeek-R1-0528"
    # Should use default base URL for Model APIs
    assert "base_url" not in config


def test_baseten_dedicated_deployment_configuration():
    """Test dedicated deployment configuration."""
    base_url = "https://model-test-deployment.api.baseten.co/environments/production/sync/v1"
    
    model = BasetenModel(
        model_id="test-deployment",
        base_url=base_url,
        client_args={"api_key": "test-key"}
    )

    config = model.get_config()
    assert config["model_id"] == "test-deployment"
    assert config["base_url"] == base_url


def test_baseten_message_formatting():
    """Test message formatting methods."""
    # Test text content formatting
    text_content = {"text": "Hello, world!"}
    formatted = BasetenModel.format_request_message_content(text_content)
    assert formatted == {"text": "Hello, world!", "type": "text"}

    # Test document content formatting
    doc_content = {
        "document": {
            "name": "test.pdf",
            "format": "pdf",
            "source": {"bytes": b"test content"}
        }
    }
    formatted = BasetenModel.format_request_message_content(doc_content)
    assert formatted["type"] == "file"
    assert formatted["file"]["filename"] == "test.pdf"

    # Test image content formatting
    img_content = {
        "image": {
            "format": "png",
            "source": {"bytes": b"test image"}
        }
    }
    formatted = BasetenModel.format_request_message_content(img_content)
    assert formatted["type"] == "image_url"


def test_baseten_message_formatting_with_tools():
    """Test message formatting with tool use and tool results."""
    # Test tool use formatting
    tool_use = {
        "name": "calculator",
        "input": {"expression": "2+2"},
        "toolUseId": "call_1"
    }
    formatted = BasetenModel.format_request_message_tool_call(tool_use)
    assert formatted["function"]["name"] == "calculator"
    assert formatted["id"] == "call_1"

    # Test tool result formatting
    tool_result = {
        "toolUseId": "call_1",
        "content": [{"json": {"result": 4}}]
    }
    formatted = BasetenModel.format_request_tool_message(tool_result)
    assert formatted["role"] == "tool"
    assert formatted["tool_call_id"] == "call_1"


def test_baseten_messages_formatting():
    """Test complete message formatting."""
    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there!"}]}
    ]
    
    formatted = BasetenModel.format_request_messages(messages)
    assert len(formatted) == 2
    assert formatted[0]["role"] == "user"
    assert formatted[1]["role"] == "assistant"


def test_baseten_messages_formatting_with_system_prompt():
    """Test message formatting with system prompt."""
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    system_prompt = "You are a helpful assistant."
    
    formatted = BasetenModel.format_request_messages(messages, system_prompt)
    assert len(formatted) == 2
    assert formatted[0]["role"] == "system"
    assert formatted[0]["content"] == system_prompt
    assert formatted[1]["role"] == "user"


def test_baseten_request_formatting():
    """Test complete request formatting."""
    model = BasetenModel(model_id="test-model")
    messages = [{"role": "user", "content": [{"text": "Hello"}]}]
    tool_specs = [
        {
            "name": "test_tool",
            "description": "A test tool",
            "inputSchema": {"json": {"type": "object"}}
        }
    ]
    
    request = model.format_request(messages, tool_specs, "You are helpful")
    
    assert request["model"] == "test-model"
    assert request["stream"] is True
    assert "tools" in request
    assert len(request["tools"]) == 1
    assert request["tools"][0]["function"]["name"] == "test_tool"


def test_baseten_chunk_formatting():
    """Test response chunk formatting."""
    model = BasetenModel(model_id="test-model")
    
    # Test message start
    chunk = model.format_chunk({"chunk_type": "message_start"})
    assert chunk == {"messageStart": {"role": "assistant"}}
    
    # Test content start
    chunk = model.format_chunk({"chunk_type": "content_start", "data_type": "text"})
    assert chunk == {"contentBlockStart": {"start": {}}}
    
    # Test content delta
    chunk = model.format_chunk({"chunk_type": "content_delta", "data_type": "text", "data": "Hello"})
    assert chunk == {"contentBlockDelta": {"delta": {"text": "Hello"}}}
    
    # Test content stop
    chunk = model.format_chunk({"chunk_type": "content_stop", "data_type": "text"})
    assert chunk == {"contentBlockStop": {}}
    
    # Test message stop
    chunk = model.format_chunk({"chunk_type": "message_stop", "data": "stop"})
    assert chunk == {"messageStop": {"stopReason": "end_turn"}}


def test_baseten_unsupported_content_type():
    """Test handling of unsupported content types."""
    unsupported_content = {"unsupported": "data"}
    
    with pytest.raises(TypeError, match="unsupported type"):
        BasetenModel.format_request_message_content(unsupported_content) 