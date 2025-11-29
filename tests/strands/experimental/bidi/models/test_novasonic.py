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
from aws_sdk_bedrock_runtime.models import ModelTimeoutException, ValidationException

from strands.experimental.bidi.models.novasonic import (
    BidiNovaSonicModel,
)
from strands.experimental.bidi.models.bidi_model import BidiModelTimeoutError
from strands.experimental.bidi.types.events import (
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
    with patch("strands.experimental.bidi.models.novasonic.BedrockRuntimeClient") as mock_cls:
        mock_instance = AsyncMock()
        mock_instance.invoke_model_with_bidirectional_stream = AsyncMock(return_value=mock_stream)
        mock_cls.return_value = mock_instance

        yield mock_instance


@pytest_asyncio.fixture
def nova_model(model_id, region, mock_client):
    """Create Nova Sonic model instance."""
    _ = mock_client

    model = BidiNovaSonicModel(model_id=model_id, client_config={"region": region})
    yield model


# Initialization and Connection Tests


@pytest.mark.asyncio
async def test_model_initialization(model_id, region):
    """Test model initialization with configuration."""
    model = BidiNovaSonicModel(model_id=model_id, client_config={"region": region})

    assert model.model_id == model_id
    assert model.region == region
    assert model._connection_id is None


# Audio Configuration Tests


@pytest.mark.asyncio
async def test_audio_config_defaults(model_id, region):
    """Test default audio configuration."""
    model = BidiNovaSonicModel(model_id=model_id, client_config={"region": region})

    assert model.config["audio"]["input_rate"] == 16000
    assert model.config["audio"]["output_rate"] == 16000
    assert model.config["audio"]["channels"] == 1
    assert model.config["audio"]["format"] == "pcm"
    assert model.config["audio"]["voice"] == "matthew"


@pytest.mark.asyncio
async def test_audio_config_partial_override(model_id, region):
    """Test partial audio configuration override."""
    provider_config = {"audio": {"output_rate": 24000, "voice": "ruth"}}
    model = BidiNovaSonicModel(model_id=model_id, client_config={"region": region}, provider_config=provider_config)

    # Overridden values
    assert model.config["audio"]["output_rate"] == 24000
    assert model.config["audio"]["voice"] == "ruth"

    # Default values preserved
    assert model.config["audio"]["input_rate"] == 16000
    assert model.config["audio"]["channels"] == 1
    assert model.config["audio"]["format"] == "pcm"


@pytest.mark.asyncio
async def test_audio_config_full_override(model_id, region):
    """Test full audio configuration override."""
    provider_config = {
        "audio": {
            "input_rate": 48000,
            "output_rate": 48000,
            "channels": 2,
            "format": "pcm",
            "voice": "stephen",
        }
    }
    model = BidiNovaSonicModel(model_id=model_id, client_config={"region": region}, provider_config=provider_config)

    assert model.config["audio"]["input_rate"] == 48000
    assert model.config["audio"]["output_rate"] == 48000
    assert model.config["audio"]["channels"] == 2
    assert model.config["audio"]["format"] == "pcm"
    assert model.config["audio"]["voice"] == "stephen"


@pytest.mark.asyncio
async def test_connection_lifecycle(nova_model, mock_client, mock_stream):
    """Test complete connection lifecycle with various configurations."""

    # Test basic connection
    await nova_model.start(system_prompt="Test system prompt")
    assert nova_model._stream == mock_stream
    assert nova_model._connection_id is not None
    assert mock_client.invoke_model_with_bidirectional_stream.called

    # Test close
    await nova_model.stop()
    assert mock_stream.close.called

    # Test connection with tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "inputSchema": {"json": json.dumps({"type": "object", "properties": {}})},
        }
    ]
    await nova_model.start(system_prompt="You are helpful", tools=tools)
    # Verify initialization events were sent (connectionStart, promptStart, system prompt)
    assert mock_stream.input_stream.send.call_count >= 3
    await nova_model.stop()


@pytest.mark.asyncio
async def test_model_stop_alone(nova_model):
    await nova_model.stop()  # Should not raise


@pytest.mark.asyncio
async def test_connection_with_message_history(nova_model, mock_client, mock_stream):
    """Test connection initialization with conversation history."""
    nova_model.client = mock_client

    # Create message history
    messages = [
        {"role": "user", "content": [{"text": "What's the weather?"}]},
        {"role": "assistant", "content": [{"text": "I'll check the weather for you."}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "tool-123", "name": "get_weather", "input": {}}}],
        },
        {
            "role": "user",
            "content": [{"toolResult": {"toolUseId": "tool-123", "content": [{"text": "Sunny, 72°F"}]}}],
        },
        {"role": "assistant", "content": [{"text": "It's sunny and 72 degrees."}]},
    ]

    # Start connection with message history
    await nova_model.start(system_prompt="You are a helpful assistant", messages=messages)

    # Verify initialization events were sent
    # Should include: sessionStart, promptStart, system prompt (3 events),
    # and message history (only text messages: 3 messages * 3 events each = 9 events)
    # Tool use/result messages are now skipped in history
    # Total: 1 + 1 + 3 + 9 = 14 events minimum
    assert mock_stream.input_stream.send.call_count >= 14

    # Verify the events contain proper role information
    sent_events = [call.args[0].value.bytes_.decode("utf-8") for call in mock_stream.input_stream.send.call_args_list]

    # Check that USER and ASSISTANT roles are present in contentStart events
    user_events = [e for e in sent_events if '"role": "USER"' in e]
    assistant_events = [e for e in sent_events if '"role": "ASSISTANT"' in e]

    # Only text messages are sent, so we expect 1 user message and 2 assistant messages
    assert len(user_events) >= 1
    assert len(assistant_events) >= 2

    await nova_model.stop()


# Send Method Tests


@pytest.mark.asyncio
async def test_send_all_content_types(nova_model, mock_stream):
    """Test sending all content types through unified send() method."""
    await nova_model.start()

    # Test text content
    text_event = BidiTextInputEvent(text="Hello, Nova!", role="user")
    await nova_model.send(text_event)
    # Should send contentStart, textInput, and contentEnd
    assert mock_stream.input_stream.send.call_count >= 3

    # Test audio content (base64 encoded)
    audio_b64 = base64.b64encode(b"audio data").decode("utf-8")
    audio_event = BidiAudioInputEvent(audio=audio_b64, format="pcm", sample_rate=16000, channels=1)
    await nova_model.send(audio_event)
    # Should start audio connection and send audio
    assert nova_model._audio_content_name
    assert mock_stream.input_stream.send.called

    # Test tool result with single content item (should be unwrapped)
    tool_result_single: ToolResult = {
        "toolUseId": "tool-123",
        "status": "success",
        "content": [{"text": "Weather is sunny"}],
    }
    await nova_model.send(ToolResultEvent(tool_result_single))
    # Should send contentStart, toolResult, and contentEnd
    assert mock_stream.input_stream.send.called

    # Test tool result with multiple content items (should send as array)
    tool_result_multi: ToolResult = {
        "toolUseId": "tool-456",
        "status": "success",
        "content": [{"text": "Part 1"}, {"json": {"data": "value"}}],
    }
    await nova_model.send(ToolResultEvent(tool_result_multi))
    assert mock_stream.input_stream.send.called

    await nova_model.stop()


@pytest.mark.asyncio
async def test_send_edge_cases(nova_model):
    """Test send() edge cases and error handling."""

    # Test image content (not supported, base64 encoded, no encoding parameter)
    await nova_model.start()
    image_b64 = base64.b64encode(b"image data").decode("utf-8")
    image_event = BidiImageInputEvent(
        image=image_b64,
        mime_type="image/jpeg",
    )

    with pytest.raises(ValueError, match=r"content not supported"):
        await nova_model.send(image_event)

    await nova_model.stop()


# Receive and Event Conversion Tests


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
    assert result.get("sample_rate") == 16000

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
    nova_event = {"toolUse": {"toolUseId": "tool-123", "toolName": "get_weather", "content": json.dumps(tool_input)}}
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
            "details": {"total": {"output": {"speechTokens": 30}}},
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
    # TEXT type contentStart (matches API spec)
    nova_event = {
        "contentStart": {
            "role": "ASSISTANT",
            "type": "TEXT",
            "additionalModelFields": '{"generationStage":"FINAL"}',
            "contentId": "content-123",
        }
    }
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert isinstance(result, BidiResponseStartEvent)
    assert result.get("type") == "bidi_response_start"
    assert nova_model._generation_stage == "FINAL"

    # Test AUDIO type contentStart (no additionalModelFields)
    nova_event = {"contentStart": {"role": "ASSISTANT", "type": "AUDIO", "contentId": "content-456"}}
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert isinstance(result, BidiResponseStartEvent)

    # Test TOOL type contentStart
    nova_event = {"contentStart": {"role": "TOOL", "type": "TOOL", "contentId": "content-789"}}
    result = nova_model._convert_nova_event(nova_event)
    assert result is not None
    assert isinstance(result, BidiResponseStartEvent)


# Audio Streaming Tests


@pytest.mark.asyncio
async def test_audio_connection_lifecycle(nova_model):
    """Test audio connection start and end lifecycle."""

    await nova_model.start()

    # Start audio connection
    await nova_model._start_audio_connection()
    assert nova_model._audio_content_name

    # End audio connection
    await nova_model._end_audio_input()
    assert not nova_model._audio_content_name

    await nova_model.stop()


# Helper Method Tests


@pytest.mark.asyncio
async def test_tool_configuration(nova_model):
    """Test building tool configuration from tool specs."""
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "inputSchema": {"json": json.dumps({"type": "object", "properties": {"location": {"type": "string"}}})},
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
    nova_model._connection_id = "test-connection"
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


@pytest.mark.asyncio
async def test_message_history_conversion(nova_model):
    """Test conversion of agent messages to Nova Sonic history events."""
    nova_model.connection_id = "test-connection"

    # Test with various message types
    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there!"}]},
        {
            "role": "assistant",
            "content": [{"toolUse": {"toolUseId": "tool-1", "name": "calculator", "input": {"expr": "2+2"}}}],
        },
        {"role": "user", "content": [{"toolResult": {"toolUseId": "tool-1", "content": [{"text": "4"}]}}]},
        {"role": "assistant", "content": [{"text": "The answer is 4"}]},
    ]

    events = nova_model._get_message_history_events(messages)

    # Only text messages generate events (3 messages * 3 events each = 9 events)
    # Tool use/result messages are now skipped in history
    assert len(events) == 9

    # Parse and verify events
    parsed_events = [json.loads(e) for e in events]

    # Check first message (user)
    assert "contentStart" in parsed_events[0]["event"]
    assert parsed_events[0]["event"]["contentStart"]["role"] == "USER"
    assert "textInput" in parsed_events[1]["event"]
    assert parsed_events[1]["event"]["textInput"]["content"] == "Hello"
    assert "contentEnd" in parsed_events[2]["event"]

    # Check second message (assistant)
    assert "contentStart" in parsed_events[3]["event"]
    assert parsed_events[3]["event"]["contentStart"]["role"] == "ASSISTANT"
    assert "textInput" in parsed_events[4]["event"]
    assert parsed_events[4]["event"]["textInput"]["content"] == "Hi there!"

    # Check third message (assistant - last text message)
    assert "contentStart" in parsed_events[6]["event"]
    assert parsed_events[6]["event"]["contentStart"]["role"] == "ASSISTANT"
    assert "textInput" in parsed_events[7]["event"]
    assert parsed_events[7]["event"]["textInput"]["content"] == "The answer is 4"


@pytest.mark.asyncio
async def test_message_history_empty_and_edge_cases(nova_model):
    """Test message history conversion with empty and edge cases."""
    nova_model.connection_id = "test-connection"

    # Test with empty messages
    events = nova_model._get_message_history_events([])
    assert len(events) == 0

    # Test with message containing no text content
    messages = [{"role": "user", "content": []}]
    events = nova_model._get_message_history_events(messages)
    assert len(events) == 0  # No events generated for empty content

    # Test with multiple text blocks in one message
    messages = [{"role": "user", "content": [{"text": "First part"}, {"text": "Second part"}]}]
    events = nova_model._get_message_history_events(messages)
    assert len(events) == 3  # contentStart, textInput, contentEnd
    parsed = json.loads(events[1])
    content = parsed["event"]["textInput"]["content"]
    assert "First part" in content
    assert "Second part" in content


# Error Handling Tests


@pytest.mark.asyncio
async def test_custom_audio_rates_in_events(model_id, region):
    """Test that audio events use configured sample rates."""
    # Create model with custom audio configuration
    provider_config = {"audio": {"output_rate": 48000, "channels": 2}}
    model = BidiNovaSonicModel(model_id=model_id, client_config={"region": region}, provider_config=provider_config)

    # Test audio output event uses custom configuration
    audio_bytes = b"test audio data"
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    nova_event = {"audioOutput": {"content": audio_base64}}
    result = model._convert_nova_event(nova_event)
    
    assert result is not None
    assert isinstance(result, BidiAudioStreamEvent)
    # Should use configured rates, not constants
    assert result.sample_rate == 48000  # Custom config
    assert result.channels == 2         # Custom config
    assert result.format == "pcm"


@pytest.mark.asyncio
async def test_default_audio_rates_in_events(model_id, region):
    """Test that audio events use default sample rates when no custom config."""
    # Create model without custom audio configuration
    model = BidiNovaSonicModel(model_id=model_id, client_config={"region": region})

    # Test audio output event uses defaults
    audio_bytes = b"test audio data"
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    nova_event = {"audioOutput": {"content": audio_base64}}
    result = model._convert_nova_event(nova_event)
    
    assert result is not None
    assert isinstance(result, BidiAudioStreamEvent)
    # Should use default rates
    assert result.sample_rate == 16000  # Default output rate
    assert result.channels == 1         # Default channels
    assert result.format == "pcm"


# Error Handling Tests
@pytest.mark.asyncio
async def test_bidi_nova_sonic_model_receive_timeout(nova_model, mock_stream):
    mock_output = AsyncMock()
    mock_output.receive.side_effect = ModelTimeoutException("Connection timeout")
    mock_stream.await_output.return_value = (None, mock_output)
    
    await nova_model.start()
    
    with pytest.raises(BidiModelTimeoutError, match=r"Connection timeout"):
        async for _ in nova_model.receive():
            pass


@pytest.mark.asyncio
async def test_bidi_nova_sonic_model_receive_timeout_validation(nova_model, mock_stream):
    mock_output = AsyncMock()
    mock_output.receive.side_effect = ValidationException("InternalErrorCode=531: Request timeout")
    mock_stream.await_output.return_value = (None, mock_output)
    
    await nova_model.start()
    
    with pytest.raises(BidiModelTimeoutError, match=r"InternalErrorCode=531"):
        async for _ in nova_model.receive():
            pass


@pytest.mark.asyncio
async def test_error_handling(nova_model, mock_stream):
    """Test error handling in various scenarios."""

    # Test response processor handles errors gracefully
    async def mock_error(*args, **kwargs):
        raise Exception("Test error")

    mock_stream.await_output.side_effect = mock_error

    await nova_model.start()

    # Wait a bit for response processor to handle error
    await asyncio.sleep(0.1)

    # Should still be able to close cleanly
    await nova_model.stop()


# Tool Result Content Tests


@pytest.mark.asyncio
async def test_tool_result_single_content_unwrapped(nova_model, mock_stream):
    """Test that single content item is unwrapped (optimization)."""
    await nova_model.start()

    tool_result: ToolResult = {
        "toolUseId": "tool-123",
        "status": "success",
        "content": [{"text": "Single result"}],
    }

    await nova_model.send(ToolResultEvent(tool_result))

    # Verify events were sent
    assert mock_stream.input_stream.send.called
    calls = mock_stream.input_stream.send.call_args_list

    # Find the toolResult event
    tool_result_events = []
    for call in calls:
        event_json = call.args[0].value.bytes_.decode("utf-8")
        event = json.loads(event_json)
        if "toolResult" in event.get("event", {}):
            tool_result_events.append(event)

    assert len(tool_result_events) > 0
    tool_result_event = tool_result_events[0]["event"]["toolResult"]

    # Single content should be unwrapped (not in array)
    content = json.loads(tool_result_event["content"])
    assert content == {"text": "Single result"}

    await nova_model.stop()


@pytest.mark.asyncio
async def test_tool_result_multiple_content_as_array(nova_model, mock_stream):
    """Test that multiple content items are sent as array."""
    await nova_model.start()

    tool_result: ToolResult = {
        "toolUseId": "tool-456",
        "status": "success",
        "content": [{"text": "Part 1"}, {"json": {"data": "value"}}],
    }

    await nova_model.send(ToolResultEvent(tool_result))

    # Verify events were sent
    assert mock_stream.input_stream.send.called
    calls = mock_stream.input_stream.send.call_args_list

    # Find the toolResult event
    tool_result_events = []
    for call in calls:
        event_json = call.args[0].value.bytes_.decode("utf-8")
        event = json.loads(event_json)
        if "toolResult" in event.get("event", {}):
            tool_result_events.append(event)

    assert len(tool_result_events) > 0
    tool_result_event = tool_result_events[0]["event"]["toolResult"]

    # Multiple content should be in array format
    content = json.loads(tool_result_event["content"])
    assert "content" in content
    assert isinstance(content["content"], list)
    assert len(content["content"]) == 2
    assert content["content"][0] == {"text": "Part 1"}
    assert content["content"][1] == {"json": {"data": "value"}}

    await nova_model.stop()


@pytest.mark.asyncio
async def test_tool_result_empty_content(nova_model, mock_stream):
    """Test that empty content is handled gracefully."""
    await nova_model.start()

    tool_result: ToolResult = {
        "toolUseId": "tool-789",
        "status": "success",
        "content": [],
    }

    await nova_model.send(ToolResultEvent(tool_result))

    # Verify events were sent
    assert mock_stream.input_stream.send.called
    calls = mock_stream.input_stream.send.call_args_list

    # Find the toolResult event
    tool_result_events = []
    for call in calls:
        event_json = call.args[0].value.bytes_.decode("utf-8")
        event = json.loads(event_json)
        if "toolResult" in event.get("event", {}):
            tool_result_events.append(event)

    assert len(tool_result_events) > 0
    tool_result_event = tool_result_events[0]["event"]["toolResult"]

    # Empty content should result in empty array wrapped in content key
    content = json.loads(tool_result_event["content"])
    assert content == {"content": []}

    await nova_model.stop()


@pytest.mark.asyncio
async def test_tool_result_unsupported_content_type(nova_model):
    """Test that unsupported content types raise ValueError."""
    await nova_model.start()

    # Test with image content (unsupported)
    tool_result_image: ToolResult = {
        "toolUseId": "tool-999",
        "status": "success",
        "content": [{"image": {"format": "jpeg", "source": {"bytes": b"image_data"}}}],
    }

    with pytest.raises(ValueError, match=r"Content type not supported by Nova Sonic"):
        await nova_model.send(ToolResultEvent(tool_result_image))

    # Test with document content (unsupported)
    tool_result_doc: ToolResult = {
        "toolUseId": "tool-888",
        "status": "success",
        "content": [{"document": {"format": "pdf", "source": {"bytes": b"doc_data"}}}],
    }

    with pytest.raises(ValueError, match=r"Content type not supported by Nova Sonic"):
        await nova_model.send(ToolResultEvent(tool_result_doc))

    # Test with mixed content (one unsupported)
    tool_result_mixed: ToolResult = {
        "toolUseId": "tool-777",
        "status": "success",
        "content": [{"text": "Valid text"}, {"image": {"format": "jpeg", "source": {"bytes": b"image_data"}}}],
    }

    with pytest.raises(ValueError, match=r"Content type not supported by Nova Sonic"):
        await nova_model.send(ToolResultEvent(tool_result_mixed))

    await nova_model.stop()
