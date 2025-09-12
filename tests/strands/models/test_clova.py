"""Unit tests for CLOVA model provider."""

from unittest.mock import AsyncMock, patch

import pytest

from strands.models.clova import ClovaModel, ClovaModelException


@pytest.fixture
def clova_model():
    """Create a ClovaModel instance for testing."""
    return ClovaModel(api_key="test-api-key", model="HCX-005")


def test_initialization():
    """Test ClovaModel initialization."""
    model = ClovaModel(api_key="test-key", model="HCX-005")
    assert model.api_key == "test-key"
    assert model.model == "HCX-005"
    assert model.temperature == 0.7
    assert model.max_tokens == 4096


def test_initialization_with_params():
    """Test ClovaModel initialization with custom parameters."""
    model = ClovaModel(
        api_key="test-key",
        model="HCX-005",
        temperature=0.5,
        max_tokens=2048,
        top_p=0.9,
    )
    assert model.temperature == 0.5
    assert model.max_tokens == 2048
    assert model.top_p == 0.9


def test_initialization_without_api_key():
    """Test ClovaModel initialization without API key."""
    with pytest.raises(ValueError, match="CLOVA API key is required"):
        ClovaModel(model="HCX-005")


@pytest.mark.asyncio
async def test_stream_with_successful_response(clova_model):
    """Test streaming with successful response."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream"}

    # Mock SSE stream - CLOVA format with message.content
    async def mock_aiter():
        yield b'data: {"message":{"content":"Hello"}}\n\n'
        yield b'data: {"message":{"content":" world"}}\n\n'
        yield b'data: {"finishReason":"stop"}\n\n'

    mock_response.aiter_bytes = mock_aiter

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_instance

        result = ""
        async for event in clova_model.stream("Test prompt"):
            if hasattr(event, "type") and event.type == "text":
                result += event.text
            elif isinstance(event, dict) and event.get("type") == "text":
                result += event["text"]

        assert result == "Hello world"


@pytest.mark.asyncio
async def test_stream_with_error_response(clova_model):
    """Test streaming with error response."""
    mock_response = AsyncMock()
    mock_response.status_code = 401
    mock_response.text = AsyncMock(return_value="Unauthorized")

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_instance

        with pytest.raises(ClovaModelException, match="CLOVA API request failed"):
            async for _ in clova_model.stream("Test prompt"):
                pass


@pytest.mark.asyncio
async def test_stream_with_system_message(clova_model):
    """Test streaming with system message."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/event-stream"}

    async def mock_aiter():
        yield b'data: {"message":{"content":"Response"}}\n\n'
        yield b'data: {"finishReason":"stop"}\n\n'

    mock_response.aiter_bytes = mock_aiter

    with patch("httpx.AsyncClient") as mock_client:
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_instance

        result = ""
        async for event in clova_model.stream("Test prompt", system_prompt="You are a helpful assistant"):
            if hasattr(event, "type") and event.type == "text":
                result += event.text
            elif isinstance(event, dict) and event.get("type") == "text":
                result += event["text"]

        # Verify the system message was included in the request
        call_args = mock_instance.post.call_args
        json_data = call_args.kwargs["json"]
        assert any(msg["role"] == "system" for msg in json_data["messages"])


@pytest.mark.asyncio
async def test_structured_output_not_implemented(clova_model):
    """Test that structured_output raises NotImplementedError."""
    from pydantic import BaseModel

    class TestModel(BaseModel):
        result: str

    with pytest.raises(NotImplementedError, match="Structured output is not yet supported for CLOVA models"):
        # structured_output is an async generator, need to call it properly
        async for _ in clova_model.structured_output(TestModel, "Test prompt"):
            pass


def test_model_str_representation(clova_model):
    """Test string representation of ClovaModel."""
    str_repr = str(clova_model)
    assert "ClovaModel" in str_repr
    assert "HCX-005" in str_repr


def test_update_config(clova_model):
    """Test updating model configuration."""
    clova_model.update_config(temperature=0.5, max_tokens=2048)
    assert clova_model.temperature == 0.5
    assert clova_model.max_tokens == 2048


def test_get_config(clova_model):
    """Test getting model configuration."""
    config = clova_model.get_config()
    assert config["model"] == "HCX-005"
    assert config["temperature"] == 0.7
    assert config["max_tokens"] == 4096
