"""Unit tests for Nova Sonic bidirectional model implementation.

Tests the unified BidirectionalModel interface implementation for Amazon Nova Sonic,
covering connection lifecycle, event conversion, audio streaming, and tool execution.
"""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from strands.experimental.bidirectional_streaming.models.novasonic import (
    NovaSonicModel,
)
from strands.types.tools import ToolResult


# Test fixtures
@pytest.fixture
def model_id():
    """Nova Sonic model identifier."""
    return "amazon.nova-sonic-v1:0"


@pytest.fixture
def region():
    """AWS region."""
    return "us-east-1"


@pytest.fixture
def mock_stream():
    """Mock Nova Sonic bidirectional stream."""
    stream = AsyncMock()
    stream.input_stream = AsyncMock()
    stream.input_stream.send = AsyncMock()
    stream.input_stream.close = AsyncMock()
    stream.await_output = AsyncMock()
    return stream


@pytest.fixture
def mock_client(mock_stream):
    """Mock Bedrock Runtime client."""
    client = AsyncMock()
    client.invoke_model_with_bidirectional_stream = AsyncMock(return_value=mock_stream)
    return client


@pytest_asyncio.fixture
async def nova_model(model_id, region):
    """Create Nova Sonic model instance."""
    model = NovaSonicModel(model_id=model_id, region=region)
    yield model
    # Cleanup
    if model._active:
        await model.close()


# Initialization and Connection Tests


@pytest.mark.asyncio
async def test_model_initialization(model_id, region):
    """Test model initialization with configuration."""
    model = NovaSonicModel(model_id=model_id, region=region)

    assert model.model_id == model_id
    assert model.region == region
    assert model.stream is None
    assert not model._active
    assert model.prompt_name is None


@pytest.mark.asyncio
async def test_connection_lifecycle(nova_model, mock_client, mock_stream):
    """Test complete connection lifecycle with various configurations."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client

        # Test basic connection
        await nova_model.connect(system_prompt="Test system prompt")
        assert nova_model._active
        assert nova_model.stream == mock_stream
        assert nova_model.prompt_name is not None
        assert mock_client.invoke_model_with_bidirectional_stream.called

        # Test close
        await nova_model.close()
        assert not nova_model._active
        assert mock_stream.input_stream.close.called

        # Test connection with tools
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather information",
                "inputSchema": {"json": json.dumps({"type": "object", "properties": {}})}
            }
        ]
        await nova_model.connect(system_prompt="You are helpful", tools=tools)
        # Verify initialization events were sent (connectionStart, promptStart, system prompt)
        assert mock_stream.input_stream.send.call_count >= 3
        await nova_model.close()


@pytest.mark.asyncio
async def test_connection_edge_cases(nova_model, mock_client, mock_stream, model_id, region):
    """Test connection error handling and edge cases."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client

        # Test double connection
        await nova_model.connect()
        with pytest.raises(RuntimeError, match="Connection already active"):
            await nova_model.connect()
        await nova_model.close()

    # Test close when already closed
    model2 = NovaSonicModel(model_id=model_id, region=region)
    await model2.close()  # Should not raise
    await model2.close()  # Second call should also be safe


# Send Method Tests


@pytest.mark.asyncio
async def test_send_all_content_types(nova_model, mock_client, mock_stream):
    """Test sending all content types through unified send() method."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client

        await nova_model.connect()

        # Test text content
        text_event = {"text": "Hello, Nova!", "role": "user"}
        await nova_model.send(text_event)
        # Should send contentStart, textInput, and contentEnd
        assert mock_stream.input_stream.send.call_count >= 3

        # Test audio content
        audio_event = {
            "audioData": b"audio data",
            "format": "pcm",
            "sampleRate": 16000,
            "channels": 1
        }
        await nova_model.send(audio_event)
        # Should start audio connection and send audio
        assert nova_model.audio_connection_active
        assert mock_stream.input_stream.send.called

        # Test tool result
        tool_result = {
            "toolUseId": "tool-123",
            "status": "success",
            "content": [{"text": "Weather is sunny"}]
        }
        await nova_model.send(tool_result)
        # Should send contentStart, toolResult, and contentEnd
        assert mock_stream.input_stream.send.called

        await nova_model.close()


@pytest.mark.asyncio
async def test_send_edge_cases(nova_model, mock_client, mock_stream, caplog):
    """Test send() edge cases and error handling."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client

        # Test send when inactive
        text_event = {"text": "Hello", "role": "user"}
        await nova_model.send(text_event)  # Should not raise

        # Test image content (not supported)
        await nova_model.connect()
        image_event = {
            "imageData": b"image data",
            "mimeType": "image/jpeg"
        }
        await nova_model.send(image_event)
        # Should log warning about unsupported image input
        assert any("not supported" in record.message.lower() for record in caplog.records)

        await nova_model.close()


# Receive and Event Conversion Tests


@pytest.mark.asyncio
async def test_receive_lifecycle_events(nova_model, mock_client, mock_stream):
    """Test that receive() emits connection start and end events."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client

        # Setup mock to return no events and then stop
        async def mock_wait_for(*args, **kwargs):
            await asyncio.sleep(0.1)
            nova_model._active = False
            raise asyncio.TimeoutError()

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            await nova_model.connect()

            events = []
            async for event in nova_model.receive():
                events.append(event)

            # Should have connection start and end
            assert len(events) >= 2
            assert "BidirectionalConnectionStart" in events[0]
            assert events[0]["BidirectionalConnectionStart"]["connectionId"] == nova_model.prompt_name
            assert "BidirectionalConnectionEnd" in events[-1]


@pytest.mark.asyncio
async def test_event_conversion(nova_model):
    """Test conversion of all Nova Sonic event types to standard format."""
    # Test audio output
    audio_bytes = b"test audio data"
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    nova_event = {"audioOutput": {"content": audio_base64}}
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert "audioOutput" in result
    assert result["audioOutput"]["audioData"] == audio_bytes
    assert result["audioOutput"]["format"] == "pcm"
    assert result["audioOutput"]["sampleRate"] == 24000

    # Test text output
    nova_event = {"textOutput": {"content": "Hello, world!", "role": "ASSISTANT"}}
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert "textOutput" in result
    assert result["textOutput"]["text"] == "Hello, world!"
    assert result["textOutput"]["role"] == "assistant"

    # Test tool use
    tool_input = {"location": "Seattle"}
    nova_event = {
        "toolUse": {
            "toolUseId": "tool-123",
            "toolName": "get_weather",
            "content": json.dumps(tool_input)
        }
    }
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert "toolUse" in result
    assert result["toolUse"]["toolUseId"] == "tool-123"
    assert result["toolUse"]["name"] == "get_weather"
    assert result["toolUse"]["input"] == tool_input

    # Test interruption
    nova_event = {"stopReason": "INTERRUPTED"}
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert "interruptionDetected" in result
    assert result["interruptionDetected"]["reason"] == "user_input"

    # Test usage metrics
    nova_event = {
        "usageEvent": {
            "totalTokens": 100,
            "totalInputTokens": 40,
            "totalOutputTokens": 60,
            "details": {
                "total": {
                    "output": {
                        "speechTokens": 30
                    }
                }
            }
        }
    }
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert "usageMetrics" in result
    assert result["usageMetrics"]["totalTokens"] == 100
    assert result["usageMetrics"]["inputTokens"] == 40
    assert result["usageMetrics"]["outputTokens"] == 60
    assert result["usageMetrics"]["audioTokens"] == 30

    # Test content start tracks role
    nova_event = {"contentStart": {"role": "USER"}}
    result = nova_model._convert_nova_event(nova_event)
    assert result is None  # contentStart doesn't emit an event
    assert nova_model._current_role == "USER"


# Audio Streaming Tests


@pytest.mark.asyncio
async def test_audio_connection_lifecycle(nova_model, mock_client, mock_stream):
    """Test audio connection start and end lifecycle."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client

        await nova_model.connect()

        # Start audio connection
        await nova_model._start_audio_connection()
        assert nova_model.audio_connection_active

        # End audio connection
        await nova_model._end_audio_input()
        assert not nova_model.audio_connection_active

        await nova_model.close()


@pytest.mark.asyncio
async def test_silence_detection(nova_model, mock_client, mock_stream):
    """Test that silence detection automatically ends audio input."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client
        nova_model.silence_threshold = 0.1  # Short threshold for testing

        await nova_model.connect()

        # Send audio to start connection
        audio_event = {
            "audioData": b"audio data",
            "format": "pcm",
            "sampleRate": 16000,
            "channels": 1
        }

        await nova_model.send(audio_event)
        assert nova_model.audio_connection_active

        # Wait for silence detection
        await asyncio.sleep(0.2)

        # Audio connection should be ended
        assert not nova_model.audio_connection_active

        await nova_model.close()


# Helper Method Tests


@pytest.mark.asyncio
async def test_tool_configuration(nova_model):
    """Test building tool configuration from tool specs."""
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "inputSchema": {
                "json": json.dumps({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                })
            }
        }
    ]

    tool_config = nova_model._build_tool_configuration(tools)

    assert len(tool_config) == 1
    assert tool_config[0]["toolSpec"]["name"] == "get_weather"
    assert tool_config[0]["toolSpec"]["description"] == "Get weather information"
    assert "inputSchema" in tool_config[0]["toolSpec"]


@pytest.mark.asyncio
async def test_event_templates(nova_model):
    """Test event template generation."""
    # Test connection start event
    event_json = nova_model._get_connection_start_event()
    event = json.loads(event_json)
    assert "event" in event
    assert "sessionStart" in event["event"]
    assert "inferenceConfiguration" in event["event"]["sessionStart"]

    # Test prompt start event
    nova_model.prompt_name = "test-prompt"
    event_json = nova_model._get_prompt_start_event([])
    event = json.loads(event_json)
    assert "event" in event
    assert "promptStart" in event["event"]
    assert event["event"]["promptStart"]["promptName"] == "test-prompt"

    # Test text input event
    content_name = "test-content"
    event_json = nova_model._get_text_input_event(content_name, "Hello")
    event = json.loads(event_json)
    assert "event" in event
    assert "textInput" in event["event"]
    assert event["event"]["textInput"]["content"] == "Hello"

    # Test tool result event
    result = {"result": "Success"}
    event_json = nova_model._get_tool_result_event(content_name, result)
    event = json.loads(event_json)
    assert "event" in event
    assert "toolResult" in event["event"]
    assert json.loads(event["event"]["toolResult"]["content"]) == result


# Error Handling Tests


@pytest.mark.asyncio
async def test_error_handling(nova_model, mock_client, mock_stream):
    """Test error handling in various scenarios."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client

        # Test response processor handles errors gracefully
        async def mock_error(*args, **kwargs):
            raise Exception("Test error")

        mock_stream.await_output.side_effect = mock_error

        await nova_model.connect()

        # Wait a bit for response processor to handle error
        await asyncio.sleep(0.1)

        # Should still be able to close cleanly
        await nova_model.close()
