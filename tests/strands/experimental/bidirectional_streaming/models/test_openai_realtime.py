"""Unit tests for OpenAI Realtime bidirectional streaming model.

Tests the unified BidiOpenAIRealtimeModel interface including:
- Model initialization and configuration
- Connection establishment with WebSocket
- Unified send() method with different content types
- Event receiving and conversion
- Connection lifecycle management
"""

import asyncio
import base64
import json
import unittest.mock

import pytest

from strands.experimental.bidirectional_streaming.models.openai import BidiOpenAIRealtimeModel
from strands.experimental.bidirectional_streaming.types.events import (
    BidiAudioInputEvent,
    BidiImageInputEvent,
    BidiTextInputEvent,
)
from strands.types.tools import ToolResult


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection."""
    mock_ws = unittest.mock.AsyncMock()
    mock_ws.send = unittest.mock.AsyncMock()
    mock_ws.close = unittest.mock.AsyncMock()
    return mock_ws


@pytest.fixture
def mock_websockets_connect(mock_websocket):
    """Mock websockets.connect function."""
    async def async_connect(*args, **kwargs):
        return mock_websocket

    with unittest.mock.patch("strands.experimental.bidirectional_streaming.models.openai.websockets.connect") as mock_connect:
        mock_connect.side_effect = async_connect
        yield mock_connect, mock_websocket


@pytest.fixture
def model_name():
    return "gpt-realtime"


@pytest.fixture
def api_key():
    return "test-api-key"


@pytest.fixture
def model(api_key, model_name):
    """Create an BidiOpenAIRealtimeModel instance."""
    return BidiOpenAIRealtimeModel(model=model_name, api_key=api_key)


@pytest.fixture
def tool_spec():
    return {
        "description": "Calculate mathematical expressions",
        "name": "calculator",
        "inputSchema": {"json": {"type": "object", "properties": {"expression": {"type": "string"}}}},
    }


@pytest.fixture
def system_prompt():
    return "You are a helpful assistant"


@pytest.fixture
def messages():
    return [{"role": "user", "content": [{"text": "Hello"}]}]


# Initialization Tests


def test_model_initialization(api_key, model_name):
    """Test model initialization with various configurations."""
    # Test default config
    model_default = BidiOpenAIRealtimeModel(api_key="test-key")
    assert model_default.model == "gpt-realtime"
    assert model_default.api_key == "test-key"
    assert model_default._active is False
    assert model_default.websocket is None

    # Test with custom model
    model_custom = BidiOpenAIRealtimeModel(model=model_name, api_key=api_key)
    assert model_custom.model == model_name
    assert model_custom.api_key == api_key

    # Test with organization and project
    model_org = BidiOpenAIRealtimeModel(
        model=model_name,
        api_key=api_key,
        organization="org-123",
        project="proj-456"
    )
    assert model_org.organization == "org-123"
    assert model_org.project == "proj-456"

    # Test with env API key
    with unittest.mock.patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
        model_env = BidiOpenAIRealtimeModel()
        assert model_env.api_key == "env-key"


def test_init_without_api_key_raises():
    """Test that initialization without API key raises error."""
    with unittest.mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            BidiOpenAIRealtimeModel()


# Connection Tests


@pytest.mark.asyncio
async def test_connection_lifecycle(mock_websockets_connect, model, system_prompt, tool_spec, messages):
    """Test complete connection lifecycle with various configurations."""
    mock_connect, mock_ws = mock_websockets_connect

    # Test basic connection
    await model.connect()
    assert model._active is True
    assert model.connection_id is not None
    assert model.websocket == mock_ws
    assert model._event_queue is not None
    assert model._response_task is not None
    mock_connect.assert_called_once()

    # Test close
    await model.close()
    assert model._active is False
    mock_ws.close.assert_called_once()

    # Test connection with system prompt
    await model.connect(system_prompt=system_prompt)
    calls = mock_ws.send.call_args_list
    session_update = next(
        (json.loads(call[0][0]) for call in calls if json.loads(call[0][0]).get("type") == "session.update"),
        None
    )
    assert session_update is not None
    assert system_prompt in session_update["session"]["instructions"]
    await model.close()

    # Test connection with tools
    await model.connect(tools=[tool_spec])
    calls = mock_ws.send.call_args_list
    # Tools are sent in a separate session.update after initial connection
    session_updates = [json.loads(call[0][0]) for call in calls if json.loads(call[0][0]).get("type") == "session.update"]
    assert len(session_updates) > 0
    # Check if any session update has tools
    has_tools = any("tools" in update.get("session", {}) for update in session_updates)
    assert has_tools
    await model.close()

    # Test connection with messages
    await model.connect(messages=messages)
    calls = mock_ws.send.call_args_list
    item_creates = [json.loads(call[0][0]) for call in calls if json.loads(call[0][0]).get("type") == "conversation.item.create"]
    assert len(item_creates) > 0
    await model.close()

    # Test connection with organization header
    model_org = BidiOpenAIRealtimeModel(api_key="test-key", organization="org-123")
    await model_org.connect()
    call_kwargs = mock_connect.call_args.kwargs
    headers = call_kwargs.get("additional_headers", [])
    org_header = [h for h in headers if h[0] == "OpenAI-Organization"]
    assert len(org_header) == 1
    assert org_header[0][1] == "org-123"
    await model_org.close()


@pytest.mark.asyncio
async def test_connection_edge_cases(mock_websockets_connect, api_key, model_name):
    """Test connection error handling and edge cases."""
    mock_connect, mock_ws = mock_websockets_connect

    # Test connection error
    model1 = BidiOpenAIRealtimeModel(model=model_name, api_key=api_key)
    mock_connect.side_effect = Exception("Connection failed")
    with pytest.raises(Exception, match="Connection failed"):
        await model1.connect()

    # Reset mock
    async def async_connect(*args, **kwargs):
        return mock_ws
    mock_connect.side_effect = async_connect

    # Test double connection
    model2 = BidiOpenAIRealtimeModel(model=model_name, api_key=api_key)
    await model2.connect()
    with pytest.raises(RuntimeError, match="Connection already active"):
        await model2.connect()
    await model2.close()

    # Test close when not connected
    model3 = BidiOpenAIRealtimeModel(model=model_name, api_key=api_key)
    await model3.close()  # Should not raise

    # Test close error handling (should not raise, just log)
    model4 = BidiOpenAIRealtimeModel(model=model_name, api_key=api_key)
    await model4.connect()
    mock_ws.close.side_effect = Exception("Close failed")
    await model4.close()  # Should not raise
    assert model4._active is False


# Send Method Tests


@pytest.mark.asyncio
async def test_send_all_content_types(mock_websockets_connect, model):
    """Test sending all content types through unified send() method."""
    from strands.types._events import ToolResultEvent
    
    _, mock_ws = mock_websockets_connect
    await model.connect()

    # Test text input
    text_input = BidiTextInputEvent(text="Hello", role="user")
    await model.send(text_input)
    calls = mock_ws.send.call_args_list
    messages = [json.loads(call[0][0]) for call in calls]
    item_create = [m for m in messages if m.get("type") == "conversation.item.create"]
    response_create = [m for m in messages if m.get("type") == "response.create"]
    assert len(item_create) > 0
    assert len(response_create) > 0

    # Test audio input (base64 encoded)
    audio_b64 = base64.b64encode(b"audio_bytes").decode('utf-8')
    audio_input = BidiAudioInputEvent(
        audio=audio_b64,
        format="pcm",
        sample_rate=24000,
        channels=1,
    )
    await model.send(audio_input)
    calls = mock_ws.send.call_args_list
    messages = [json.loads(call[0][0]) for call in calls]
    audio_append = [m for m in messages if m.get("type") == "input_audio_buffer.append"]
    assert len(audio_append) > 0
    assert "audio" in audio_append[0]
    # Audio should be passed through as base64
    assert audio_append[0]["audio"] == audio_b64

    # Test tool result
    tool_result: ToolResult = {
        "toolUseId": "tool-123",
        "status": "success",
        "content": [{"text": "Result: 42"}],
    }
    await model.send(ToolResultEvent(tool_result))
    calls = mock_ws.send.call_args_list
    messages = [json.loads(call[0][0]) for call in calls]
    item_create = [m for m in messages if m.get("type") == "conversation.item.create"]
    assert len(item_create) > 0
    item = item_create[-1].get("item", {})
    assert item.get("type") == "function_call_output"
    assert item.get("call_id") == "tool-123"

    await model.close()


@pytest.mark.asyncio
async def test_send_edge_cases(mock_websockets_connect, model):
    """Test send() edge cases and error handling."""
    _, mock_ws = mock_websockets_connect

    # Test send when inactive
    text_input = BidiTextInputEvent(text="Hello", role="user")
    await model.send(text_input)
    mock_ws.send.assert_not_called()

    # Test image input (not supported, base64 encoded, no encoding parameter)
    await model.connect()
    image_b64 = base64.b64encode(b"image_bytes").decode('utf-8')
    image_input = BidiImageInputEvent(
        image=image_b64,
        mime_type="image/jpeg",
    )
    with unittest.mock.patch("strands.experimental.bidirectional_streaming.models.openai.logger") as mock_logger:
        await model.send(image_input)
        mock_logger.warning.assert_called_with("Image input not supported by OpenAI Realtime API")

    # Test unknown content type
    unknown_content = {"unknown_field": "value"}
    with unittest.mock.patch("strands.experimental.bidirectional_streaming.models.openai.logger") as mock_logger:
        await model.send(unknown_content)
        assert mock_logger.warning.called

    await model.close()


# Receive Method Tests


@pytest.mark.asyncio
async def test_receive_lifecycle_events(mock_websockets_connect, model):
    """Test that receive() emits connection start and end events."""
    _, _ = mock_websockets_connect

    await model.connect()

    # Get first event
    receive_gen = model.receive()
    first_event = await anext(receive_gen)

    # First event should be connection start (new TypedEvent format)
    assert first_event.get("type") == "bidirectional_connection_start"
    assert first_event.get("connection_id") == model.connection_id
    assert first_event.get("model") == model.model

    # Close to trigger session end
    await model.close()

    # Collect remaining events
    events = [first_event]
    try:
        async for event in receive_gen:
            events.append(event)
    except StopAsyncIteration:
        pass

    # Last event should be connection close (new TypedEvent format)
    assert events[-1].get("type") == "bidirectional_connection_close"


@pytest.mark.asyncio
async def test_event_conversion(mock_websockets_connect, model):
    """Test conversion of all OpenAI event types to standard format."""
    _, _ = mock_websockets_connect
    await model.connect()

    # Test audio output (now returns list with BidiAudioStreamEvent)
    from strands.experimental.bidirectional_streaming.types.events import BidiAudioStreamEvent
    audio_event = {
        "type": "response.output_audio.delta",
        "delta": base64.b64encode(b"audio_data").decode()
    }
    converted = model._convert_openai_event(audio_event)
    assert isinstance(converted, list)
    assert len(converted) == 1
    assert isinstance(converted[0], BidiAudioStreamEvent)
    assert converted[0].get("type") == "bidirectional_audio_stream"
    assert converted[0].get("audio") == base64.b64encode(b"audio_data").decode()
    assert converted[0].get("format") == "pcm"

    # Test text output (now returns list with BidiTranscriptStreamEvent)
    from strands.experimental.bidirectional_streaming.types.events import BidiTranscriptStreamEvent
    text_event = {
        "type": "response.output_text.delta",
        "delta": "Hello from OpenAI"
    }
    converted = model._convert_openai_event(text_event)
    assert isinstance(converted, list)
    assert len(converted) == 1
    assert isinstance(converted[0], BidiTranscriptStreamEvent)
    assert converted[0].get("type") == "bidirectional_transcript_stream"
    assert converted[0].get("text") == "Hello from OpenAI"
    assert converted[0].get("role") == "assistant"
    assert converted[0].delta == {"text": "Hello from OpenAI"}
    assert converted[0].is_final is True

    # Test function call sequence
    item_added = {
        "type": "response.output_item.added",
        "item": {
            "type": "function_call",
            "call_id": "call-123",
            "name": "calculator"
        }
    }
    model._convert_openai_event(item_added)

    args_delta = {
        "type": "response.function_call_arguments.delta",
        "call_id": "call-123",
        "delta": '{"expression": "2+2"}'
    }
    model._convert_openai_event(args_delta)

    args_done = {
        "type": "response.function_call_arguments.done",
        "call_id": "call-123"
    }
    converted = model._convert_openai_event(args_done)
    # Now returns list with ToolUseStreamEvent
    assert isinstance(converted, list)
    assert len(converted) == 1
    # ToolUseStreamEvent has delta and current_tool_use, not a "type" field
    assert "delta" in converted[0]
    assert "toolUse" in converted[0]["delta"]
    tool_use = converted[0]["delta"]["toolUse"]
    assert tool_use["toolUseId"] == "call-123"
    assert tool_use["name"] == "calculator"
    assert tool_use["input"]["expression"] == "2+2"

    # Test voice activity (now returns list with BidiInterruptionEvent for speech_started)
    from strands.experimental.bidirectional_streaming.types.events import BidiInterruptionEvent
    speech_started = {
        "type": "input_audio_buffer.speech_started"
    }
    converted = model._convert_openai_event(speech_started)
    assert isinstance(converted, list)
    assert len(converted) == 1
    assert isinstance(converted[0], BidiInterruptionEvent)
    assert converted[0].get("type") == "bidirectional_interruption"
    assert converted[0].get("reason") == "user_speech"

    await model.close()


# Helper Method Tests


def test_config_building(model, system_prompt, tool_spec):
    """Test building session config with various options."""
    # Test basic config
    config_basic = model._build_session_config(None, None)
    assert isinstance(config_basic, dict)
    assert "instructions" in config_basic
    assert "audio" in config_basic

    # Test with system prompt
    config_prompt = model._build_session_config(system_prompt, None)
    assert config_prompt["instructions"] == system_prompt

    # Test with tools
    config_tools = model._build_session_config(None, [tool_spec])
    assert "tools" in config_tools
    assert len(config_tools["tools"]) > 0


def test_tool_conversion(model, tool_spec):
    """Test tool conversion to OpenAI format."""
    # Test with tools
    openai_tools = model._convert_tools_to_openai_format([tool_spec])
    assert len(openai_tools) == 1
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["name"] == "calculator"
    assert openai_tools[0]["description"] == "Calculate mathematical expressions"

    # Test empty list
    openai_empty = model._convert_tools_to_openai_format([])
    assert openai_empty == []


def test_helper_methods(model):
    """Test various helper methods."""
    # Test _require_active
    assert model._require_active() is False
    model._active = True
    assert model._require_active() is True
    model._active = False

    # Test _create_text_event (now returns BidiTranscriptStreamEvent)
    from strands.experimental.bidirectional_streaming.types.events import BidiTranscriptStreamEvent
    text_event = model._create_text_event("Hello", "user")
    assert isinstance(text_event, BidiTranscriptStreamEvent)
    assert text_event.get("type") == "bidirectional_transcript_stream"
    assert text_event.get("text") == "Hello"
    assert text_event.get("role") == "user"
    assert text_event.delta == {"text": "Hello"}
    assert text_event.is_final is True
    assert text_event.current_transcript == "Hello"

    # Test _create_voice_activity_event (now returns BidiInterruptionEvent for speech_started)
    from strands.experimental.bidirectional_streaming.types.events import BidiInterruptionEvent
    voice_event = model._create_voice_activity_event("speech_started")
    assert isinstance(voice_event, BidiInterruptionEvent)
    assert voice_event.get("type") == "bidirectional_interruption"
    assert voice_event.get("reason") == "user_speech"
    
    # Other voice activities return None
    assert model._create_voice_activity_event("speech_stopped") is None


@pytest.mark.asyncio
async def test_send_event_helper(mock_websockets_connect, model):
    """Test _send_event helper method."""
    _, mock_ws = mock_websockets_connect
    await model.connect()

    test_event = {"type": "test.event", "data": "test"}
    await model._send_event(test_event)

    calls = mock_ws.send.call_args_list
    last_call = calls[-1]
    sent_message = json.loads(last_call[0][0])
    assert sent_message == test_event

    await model.close()
