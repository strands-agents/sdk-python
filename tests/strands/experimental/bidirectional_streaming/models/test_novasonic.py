"""Unit tests for Nova Sonic bidirectional model implementation.

Tests the unified BidirectionalModel interface implementation for Amazon Nova Sonic,
covering connection lifecycle, event conversion, audio streaming, and tool execution.
"""

import asyncio
import base64
import json
import uuid
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio

from strands.experimental.bidirectional_streaming.models.novasonic import (
    NovaSonicBidirectionalModel,
)
from strands.types.tools import ToolResult, ToolSpec


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
    model = NovaSonicBidirectionalModel(model_id=model_id, region=region)
    yield model
    # Cleanup
    if model._active:
        await model.close()


# Connection lifecycle tests
@pytest.mark.asyncio
async def test_model_initialization(model_id, region):
    """Test model initialization with configuration."""
    model = NovaSonicBidirectionalModel(model_id=model_id, region=region)
    
    assert model.model_id == model_id
    assert model.region == region
    assert model.stream is None
    assert not model._active
    assert model.prompt_name is None


@pytest.mark.asyncio
async def test_connect_establishes_connection(nova_model, mock_client, mock_stream):
    """Test that connect() establishes bidirectional connection."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client
        
        await nova_model.connect(system_prompt="Test system prompt")
        
        assert nova_model._active
        assert nova_model.stream == mock_stream
        assert nova_model.prompt_name is not None
        assert mock_client.invoke_model_with_bidirectional_stream.called


@pytest.mark.asyncio
async def test_connect_sends_initialization_events(nova_model, mock_client, mock_stream):
    """Test that connect() sends proper initialization sequence."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client
        
        system_prompt = "You are a helpful assistant"
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather information",
                "inputSchema": {"json": json.dumps({"type": "object", "properties": {}})}
            }
        ]
        
        await nova_model.connect(system_prompt=system_prompt, tools=tools)
        
        # Verify initialization events were sent
        assert mock_stream.input_stream.send.call_count >= 3  # connectionStart, promptStart, system prompt


@pytest.mark.asyncio
async def test_close_cleanup(nova_model, mock_client, mock_stream):
    """Test that close() properly cleans up resources."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client
        
        await nova_model.connect()
        await nova_model.close()
        
        assert not nova_model._active
        assert mock_stream.input_stream.close.called


# Event conversion tests
@pytest.mark.asyncio
async def test_receive_emits_connection_start(nova_model, mock_client, mock_stream):
    """Test that receive() emits connection start event."""
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


@pytest.mark.asyncio
async def test_convert_audio_output_event(nova_model):
    """Test conversion of Nova Sonic audio output to standard format."""
    audio_bytes = b"test audio data"
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    
    nova_event = {
        "audioOutput": {
            "content": audio_base64
        }
    }
    
    result = nova_model._convert_nova_event(nova_event)
    
    assert result is not None
    assert "audioOutput" in result
    assert result["audioOutput"]["audioData"] == audio_bytes
    assert result["audioOutput"]["format"] == "pcm"
    assert result["audioOutput"]["sampleRate"] == 24000


@pytest.mark.asyncio
async def test_convert_text_output_event(nova_model):
    """Test conversion of Nova Sonic text output to standard format."""
    nova_event = {
        "textOutput": {
            "content": "Hello, world!",
            "role": "ASSISTANT"
        }
    }
    
    result = nova_model._convert_nova_event(nova_event)
    
    assert result is not None
    assert "textOutput" in result
    assert result["textOutput"]["text"] == "Hello, world!"
    assert result["textOutput"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_convert_tool_use_event(nova_model):
    """Test conversion of Nova Sonic tool use to standard format."""
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


@pytest.mark.asyncio
async def test_convert_interruption_event(nova_model):
    """Test conversion of Nova Sonic interruption to standard format."""
    nova_event = {
        "stopReason": "INTERRUPTED"
    }
    
    result = nova_model._convert_nova_event(nova_event)
    
    assert result is not None
    assert "interruptionDetected" in result
    assert result["interruptionDetected"]["reason"] == "user_input"


@pytest.mark.asyncio
async def test_convert_usage_metrics_event(nova_model):
    """Test conversion of Nova Sonic usage event to standard format."""
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


@pytest.mark.asyncio
async def test_convert_content_start_tracks_role(nova_model):
    """Test that contentStart events track role for subsequent text output."""
    nova_event = {
        "contentStart": {
            "role": "USER"
        }
    }
    
    result = nova_model._convert_nova_event(nova_event)
    
    # contentStart doesn't emit an event but stores role
    assert result is None
    assert nova_model._current_role == "USER"


# Send method tests
@pytest.mark.asyncio
async def test_send_text_content(nova_model, mock_client, mock_stream):
    """Test sending text content through unified send() method."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client
        
        await nova_model.connect()
        
        text_event = {
            "text": "Hello, Nova!",
            "role": "user"
        }
        
        await nova_model.send(text_event)
        
        # Should send contentStart, textInput, and contentEnd
        assert mock_stream.input_stream.send.call_count >= 3


@pytest.mark.asyncio
async def test_send_audio_content(nova_model, mock_client, mock_stream):
    """Test sending audio content through unified send() method."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client
        
        await nova_model.connect()
        
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


@pytest.mark.asyncio
async def test_send_tool_result(nova_model, mock_client, mock_stream):
    """Test sending tool result through unified send() method."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client
        
        await nova_model.connect()
        
        tool_result = {
            "toolUseId": "tool-123",
            "status": "success",
            "content": [{"text": "Weather is sunny"}]
        }
        
        await nova_model.send(tool_result)
        
        # Should send contentStart, toolResult, and contentEnd
        assert mock_stream.input_stream.send.call_count >= 3


@pytest.mark.asyncio
async def test_send_image_content_not_supported(nova_model, mock_client, mock_stream, caplog):
    """Test that image content logs warning (not supported by Nova Sonic)."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client
        
        await nova_model.connect()
        
        image_event = {
            "imageData": b"image data",
            "mimeType": "image/jpeg"
        }
        
        await nova_model.send(image_event)
        
        # Should log warning about unsupported image input
        assert any("not supported" in record.message.lower() for record in caplog.records)


# Audio streaming tests
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


@pytest.mark.asyncio
async def test_silence_detection_ends_audio(nova_model, mock_client, mock_stream):
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


# Tool configuration tests
@pytest.mark.asyncio
async def test_build_tool_configuration(nova_model):
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


# Event template tests
@pytest.mark.asyncio
async def test_get_connection_start_event(nova_model):
    """Test connection start event generation."""
    event_json = nova_model._get_connection_start_event()
    event = json.loads(event_json)
    
    assert "event" in event
    assert "sessionStart" in event["event"]
    assert "inferenceConfiguration" in event["event"]["sessionStart"]


@pytest.mark.asyncio
async def test_get_prompt_start_event(nova_model):
    """Test prompt start event generation."""
    nova_model.prompt_name = "test-prompt"
    
    event_json = nova_model._get_prompt_start_event([])
    event = json.loads(event_json)
    
    assert "event" in event
    assert "promptStart" in event["event"]
    assert event["event"]["promptStart"]["promptName"] == "test-prompt"


@pytest.mark.asyncio
async def test_get_text_input_event(nova_model):
    """Test text input event generation."""
    nova_model.prompt_name = "test-prompt"
    content_name = "test-content"
    
    event_json = nova_model._get_text_input_event(content_name, "Hello")
    event = json.loads(event_json)
    
    assert "event" in event
    assert "textInput" in event["event"]
    assert event["event"]["textInput"]["content"] == "Hello"


@pytest.mark.asyncio
async def test_get_tool_result_event(nova_model):
    """Test tool result event generation."""
    nova_model.prompt_name = "test-prompt"
    content_name = "test-content"
    result = {"result": "Success"}
    
    event_json = nova_model._get_tool_result_event(content_name, result)
    event = json.loads(event_json)
    
    assert "event" in event
    assert "toolResult" in event["event"]
    assert json.loads(event["event"]["toolResult"]["content"]) == result


# Error handling tests
@pytest.mark.asyncio
async def test_send_when_inactive(nova_model):
    """Test that send() handles inactive connection gracefully."""
    text_event = {
        "text": "Hello",
        "role": "user"
    }
    
    # Should not raise error when inactive
    await nova_model.send(text_event)


@pytest.mark.asyncio
async def test_close_when_already_closed(nova_model):
    """Test that close() handles already closed connection."""
    # Should not raise error when already inactive
    await nova_model.close()
    await nova_model.close()  # Second call should be safe


@pytest.mark.asyncio
async def test_response_processor_handles_errors(nova_model, mock_client, mock_stream):
    """Test that response processor handles errors gracefully."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client
        
        # Setup mock to raise error
        async def mock_error(*args, **kwargs):
            raise Exception("Test error")
        
        mock_stream.await_output.side_effect = mock_error
        
        await nova_model.connect()
        
        # Wait a bit for response processor to handle error
        await asyncio.sleep(0.1)
        
        # Should still be able to close cleanly
        await nova_model.close()


# Integration-style tests
@pytest.mark.asyncio
async def test_full_conversation_flow(nova_model, mock_client, mock_stream):
    """Test a complete conversation flow with text and audio."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model._client = mock_client
        
        # Connect
        await nova_model.connect(system_prompt="You are helpful")
        
        # Send text
        await nova_model.send({"text": "Hello", "role": "user"})
        
        # Send audio
        await nova_model.send({
            "audioData": b"audio",
            "format": "pcm",
            "sampleRate": 16000,
            "channels": 1
        })
        
        # Send tool result
        await nova_model.send({
            "toolUseId": "tool-1",
            "status": "success",
            "content": [{"text": "Result"}]
        })
        
        # Close
        await nova_model.close()
        
        # Verify all operations completed
        assert not nova_model._active
