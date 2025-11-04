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
    assert model.session_id is None


@pytest.mark.asyncio
async def test_connection_lifecycle(nova_model, mock_client, mock_stream):
    """Test complete connection lifecycle with various configurations."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model.client = mock_client

        # Test basic connection
        await nova_model.connect(system_prompt="Test system prompt")
        assert nova_model._active
        assert nova_model.stream == mock_stream
        assert nova_model.session_id is not None
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
        nova_model.client = mock_client

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
    from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import (
        TextInputEvent,
        AudioInputEvent,
    )
    from strands.types._events import ToolResultEvent
    
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model.client = mock_client

        await nova_model.connect()

        # Test text content
        text_event = TextInputEvent(text="Hello, Nova!", role="user")
        await nova_model.send(text_event)
        # Should send contentStart, textInput, and contentEnd
        assert mock_stream.input_stream.send.call_count >= 3

        # Test audio content (base64 encoded)
        audio_b64 = base64.b64encode(b"audio data").decode('utf-8')
        audio_event = AudioInputEvent(
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

        await nova_model.close()


@pytest.mark.asyncio
async def test_send_edge_cases(nova_model, mock_client, mock_stream, caplog):
    """Test send() edge cases and error handling."""
    from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import (
        TextInputEvent,
        ImageInputEvent,
    )
    
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model.client = mock_client

        # Test send when inactive
        text_event = TextInputEvent(text="Hello", role="user")
        await nova_model.send(text_event)  # Should not raise

        # Test image content (not supported, base64 encoded, no encoding parameter)
        await nova_model.connect()
        import base64
        image_b64 = base64.b64encode(b"image data").decode('utf-8')
        image_event = ImageInputEvent(
            image=image_b64,
            mime_type="image/jpeg",
        )
        await nova_model.send(image_event)
        # Should log warning about unsupported image input
        assert any("not supported" in record.message.lower() for record in caplog.records)

        await nova_model.close()


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
            await nova_model.connect()

            events = []
            async for event in nova_model.receive():
                events.append(event)

            # Should have session start and end (new TypedEvent format)
            assert len(events) >= 2
            assert events[0].get("type") == "bidirectional_session_start"
            assert events[0].get("session_id") == nova_model.session_id
            assert events[-1].get("type") == "bidirectional_session_end"


@pytest.mark.asyncio
async def test_event_conversion(nova_model):
    """Test conversion of all Nova Sonic event types to standard format."""
    # Test audio output (now returns AudioStreamEvent)
    from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import AudioStreamEvent
    audio_bytes = b"test audio data"
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    nova_event = {"audioOutput": {"content": audio_base64}}
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert isinstance(result, AudioStreamEvent)
    assert result.get("type") == "bidirectional_audio_stream"
    # Audio is kept as base64 string
    assert result.get("audio") == audio_base64
    assert result.get("format") == "pcm"
    assert result.get("sample_rate") == 24000

    # Test text output (now returns TranscriptStreamEvent)
    from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import TranscriptStreamEvent
    nova_event = {"textOutput": {"content": "Hello, world!", "role": "ASSISTANT"}}
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert isinstance(result, TranscriptStreamEvent)
    assert result.get("type") == "bidirectional_transcript_stream"
    assert result.get("text") == "Hello, world!"
    assert result.get("source") == "assistant"

    # Test tool use (now returns dict with tool_use)
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
    assert result.get("type") == "tool_use"
    tool_use = result.get("tool_use")
    assert tool_use["toolUseId"] == "tool-123"
    assert tool_use["name"] == "get_weather"
    assert tool_use["input"] == tool_input

    # Test interruption (now returns InterruptionEvent)
    from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import InterruptionEvent
    nova_event = {"stopReason": "INTERRUPTED"}
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert isinstance(result, InterruptionEvent)
    assert result.get("type") == "bidirectional_interruption"
    assert result.get("reason") == "user_speech"

    # Test usage metrics (now returns MultimodalUsage)
    from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import MultimodalUsage
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
    assert isinstance(result, MultimodalUsage)
    assert result.get("type") == "multimodal_usage"
    assert result.get("totalTokens") == 100
    assert result.get("inputTokens") == 40
    assert result.get("outputTokens") == 60

    # Test content start tracks role and emits TurnStartEvent
    from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import TurnStartEvent
    nova_event = {"contentStart": {"role": "USER"}}
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert isinstance(result, TurnStartEvent)
    assert result.get("type") == "bidirectional_turn_start"
    assert nova_model._current_role == "USER"


# Audio Streaming Tests


@pytest.mark.asyncio
async def test_audio_connection_lifecycle(nova_model, mock_client, mock_stream):
    """Test audio connection start and end lifecycle."""
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model.client = mock_client

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
    from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import AudioInputEvent
    
    with patch.object(nova_model, "_initialize_client", new_callable=AsyncMock):
        nova_model.client = mock_client
        nova_model.silence_threshold = 0.1  # Short threshold for testing

        await nova_model.connect()

        # Send audio to start connection (base64 encoded)
        import base64
        audio_b64 = base64.b64encode(b"audio data").decode('utf-8')
        audio_event = AudioInputEvent(
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
    nova_model.session_id = "test-session"
    event_json = nova_model._get_prompt_start_event([])
    event = json.loads(event_json)
    assert "event" in event
    assert "promptStart" in event["event"]
    assert event["event"]["promptStart"]["promptName"] == "test-session"

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

        await nova_model.connect()

        # Wait a bit for response processor to handle error
        await asyncio.sleep(0.1)

        # Should still be able to close cleanly
        await nova_model.close()
