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
    BidiNovaSonicModel,
)
from strands.experimental.bidirectional_streaming.types.events import (
    BidiAudioInputEvent,
    BidiAudioStreamEvent,
    BidiImageInputEvent,
    BidiInterruptionEvent,
    BidiResponseStartEvent,
    BidiTextInputEvent,
    BidiTranscriptStreamEvent,
    BidiUsageEvent,
)
from strands.types._events import ToolResultEvent
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
    model = BidiNovaSonicModel(model_id=model_id, region=region)
    yield model
    # Cleanup
    if model._active:
        await model.stop()


# Initialization and Connection Tests


@pytest.mark.asyncio
async def test_model_initialization(model_id, region):
    """Test model initialization with configuration."""
    model = BidiNovaSonicModel(model_id=model_id, region=region)

    assert model.model_id == model_id
    assert model.region == region
    assert model.stream is None
    assert not model._active
    assert model.connection_id is None


@pytest.mark.asyncio
async def test_connection_lifecycle(nova_model, mock_client, mock_stream):
    """Test complete connection lifecycle with various configurations."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model.client = mock_client

        # Test basic connection
        await nova_model.start(system_prompt="Test system prompt")
        assert nova_model._active
        assert nova_model.stream == mock_stream
        assert nova_model.connection_id is not None
        assert mock_client.invoke_model_with_bidirectional_stream.called

        # Test close
        await nova_model.stop()
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
        await nova_model.start(system_prompt="You are helpful", tools=tools)
        # Verify initialization events were sent (connectionStart, promptStart, system prompt)
        assert mock_stream.input_stream.send.call_count >= 3
        await nova_model.stop()


@pytest.mark.asyncio
async def test_connection_edge_cases(nova_model, mock_client, mock_stream, model_id, region):
    """Test connection error handling and edge cases."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model.client = mock_client

        # Test double connection
        await nova_model.start()
        with pytest.raises(RuntimeError, match="Connection already active"):
            await nova_model.start()
        await nova_model.stop()

    # Test close when already closed
    model2 = BidiNovaSonicModel(model_id=model_id, region=region)
    await model2.stop()  # Should not raise
    await model2.stop()  # Second call should also be safe


# Send Method Tests


@pytest.mark.asyncio
async def test_send_all_content_types(nova_model, mock_client, mock_stream):
    """Test sending all content types through unified send() method."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model.client = mock_client

        await nova_model.start()

        # Test text content
        text_event = BidiTextInputEvent(text="Hello, Nova!", role="user")
        await nova_model.send(text_event)
        # Should send contentStart, textInput, and contentEnd
        assert mock_stream.input_stream.send.call_count >= 3

        # Test audio content (base64 encoded)
        audio_b64 = base64.b64encode(b"audio data").decode('utf-8')
        audio_event = BidiAudioInputEvent(
            audio=audio_b64,
            format="pcm",
            sample_rate=16000,
            channels=1
        )
        await nova_model.send(audio_event)
        # Should start audio connection and send audio
        assert nova_model.audio_connection_active
        assert mock_stream.input_stream.send.called

        # Test tool result
        tool_result: ToolResult = {
            "toolUseId": "tool-123",
            "status": "success",
            "content": [{"text": "Weather is sunny"}]
        }
        await nova_model.send(ToolResultEvent(tool_result))
        # Should send contentStart, toolResult, and contentEnd
        assert mock_stream.input_stream.send.called

        await nova_model.stop()


@pytest.mark.asyncio
async def test_send_edge_cases(nova_model, mock_client, mock_stream, caplog):
    """Test send() edge cases and error handling."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model.client = mock_client

        # Test send when inactive
        text_event = BidiTextInputEvent(text="Hello", role="user")
        await nova_model.send(text_event)  # Should not raise

        # Test image content (not supported, base64 encoded, no encoding parameter)
        await nova_model.start()
        image_b64 = base64.b64encode(b"image data").decode('utf-8')
        image_event = BidiImageInputEvent(
            image=image_b64,
            mime_type="image/jpeg",
        )
        await nova_model.send(image_event)
        # Should log warning about unsupported image input
        assert any("not supported" in record.message.lower() for record in caplog.records)

        await nova_model.stop()


# Receive and Event Conversion Tests


@pytest.mark.asyncio
async def test_receive_lifecycle_events(nova_model, mock_client, mock_stream):
    """Test that receive() emits connection start and end events."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model.client = mock_client

        # Setup mock to return no events and then stop
        async def mock_wait_for(*args, **kwargs):
            await asyncio.sleep(0.1)
            nova_model._active = False
            raise asyncio.TimeoutError()

        with patch("asyncio.wait_for", side_effect=mock_wait_for):
            await nova_model.start()

            events = []
            async for event in nova_model.receive():
                events.append(event)

            # Should have session start and end (new TypedEvent format)
            assert len(events) >= 2
            assert events[0].get("type") == "bidi_connection_start"
            assert events[0].get("connection_id") == nova_model.connection_id
            assert events[-1].get("type") == "bidi_connection_close"


@pytest.mark.asyncio
async def test_event_conversion(nova_model):
    """Test conversion of all Nova Sonic event types to standard format."""
    # Test audio output (now returns BidiAudioStreamEvent)
    audio_bytes = b"test audio data"
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    nova_event = {"audioOutput": {"content": audio_base64}}
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert isinstance(result, BidiAudioStreamEvent)
    assert result.get("type") == "bidi_audio_stream"
    # Audio is kept as base64 string
    assert result.get("audio") == audio_base64
    assert result.get("format") == "pcm"
    assert result.get("sample_rate") == 24000

    # Test text output (now returns BidiTranscriptStreamEvent)
    nova_event = {"textOutput": {"content": "Hello, world!", "role": "ASSISTANT"}}
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert isinstance(result, BidiTranscriptStreamEvent)
    assert result.get("type") == "bidi_transcript_stream"
    assert result.get("text") == "Hello, world!"
    assert result.get("role") == "assistant"
    assert result.delta == {"text": "Hello, world!"}
    assert result.current_transcript == "Hello, world!"

    # Test tool use (now returns ToolUseStreamEvent from core strands)
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
    # ToolUseStreamEvent has delta and current_tool_use, not a "type" field
    assert "delta" in result
    assert "toolUse" in result["delta"]
    tool_use = result["delta"]["toolUse"]
    assert tool_use["toolUseId"] == "tool-123"
    assert tool_use["name"] == "get_weather"
    assert tool_use["input"] == tool_input

    # Test interruption (now returns BidiInterruptionEvent)
    nova_event = {"stopReason": "INTERRUPTED"}
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert isinstance(result, BidiInterruptionEvent)
    assert result.get("type") == "bidi_interruption"
    assert result.get("reason") == "user_speech"

    # Test usage metrics (now returns BidiUsageEvent)
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
    assert isinstance(result, BidiUsageEvent)
    assert result.get("type") == "bidi_usage"
    assert result.get("totalTokens") == 100
    assert result.get("inputTokens") == 40
    assert result.get("outputTokens") == 60

    # Test content start tracks role and emits BidiResponseStartEvent
    nova_event = {"contentStart": {"role": "USER"}}
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert isinstance(result, BidiResponseStartEvent)
    assert result.get("type") == "bidi_response_start"
    assert nova_model._current_role == "USER"


# Audio Streaming Tests


@pytest.mark.asyncio
async def test_audio_connection_lifecycle(nova_model, mock_client, mock_stream):
    """Test audio connection start and end lifecycle."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model.client = mock_client

        await nova_model.start()

        # Start audio connection
        await nova_model._start_audio_connection()
        assert nova_model.audio_connection_active

        # End audio connection
        await nova_model._end_audio_input()
        assert not nova_model.audio_connection_active

        await nova_model.stop()


@pytest.mark.asyncio
async def test_silence_detection(nova_model, mock_client, mock_stream):
    """Test that silence detection automatically ends audio input."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model.client = mock_client
        nova_model.silence_threshold = 0.1  # Short threshold for testing

        await nova_model.start()

        # Send audio to start connection (base64 encoded)
        audio_b64 = base64.b64encode(b"audio data").decode('utf-8')
        audio_event = BidiAudioInputEvent(
            audio=audio_b64,
            format="pcm",
            sample_rate=16000,
            channels=1
        )

        await nova_model.send(audio_event)
        assert nova_model.audio_connection_active

        # Wait for silence detection
        await asyncio.sleep(0.2)

        # Audio connection should be ended
        assert not nova_model.audio_connection_active

        await nova_model.stop()


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
    nova_model.connection_id = "test-connection"
    event_json = nova_model._get_prompt_start_event([])
    event = json.loads(event_json)
    assert "event" in event
    assert "promptStart" in event["event"]
    assert event["event"]["promptStart"]["promptName"] == "test-connection"

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
        nova_model.client = mock_client

        # Test response processor handles errors gracefully
        async def mock_error(*args, **kwargs):
            raise Exception("Test error")

        mock_stream.await_output.side_effect = mock_error

        await nova_model.start()

        # Wait a bit for response processor to handle error
        await asyncio.sleep(0.1)

        # Should still be able to close cleanly
        await nova_model.stop()
