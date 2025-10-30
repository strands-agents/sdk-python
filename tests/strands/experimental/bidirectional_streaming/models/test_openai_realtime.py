"""Unit tests for OpenAI Realtime bidirectional streaming model.

Tests the unified OpenAIRealtimeBidirectionalModel interface including:
- Model initialization and configuration
- Connection establishment with WebSocket
- Unified send() method with different content types
- Event receiving and conversion
- Connection lifecycle management
- Background task management
"""

import asyncio
import base64
import json
import unittest.mock

import pytest

from strands.experimental.bidirectional_streaming.models.openai import OpenAIRealtimeBidirectionalModel
from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import (
    AudioInputEvent,
    ImageInputEvent,
    TextInputEvent,
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
    """Create an OpenAIRealtimeBidirectionalModel instance."""
    return OpenAIRealtimeBidirectionalModel(model=model_name, api_key=api_key)


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


def test_init_default_config():
    """Test model initialization with default configuration."""
    model = OpenAIRealtimeBidirectionalModel(api_key="test-key")
    
    assert model.model == "gpt-realtime"
    assert model.api_key == "test-key"
    assert model._active is False
    assert model.websocket is None


def test_init_with_api_key(api_key, model_name):
    """Test model initialization with API key."""
    model = OpenAIRealtimeBidirectionalModel(model=model_name, api_key=api_key)
    
    assert model.model == model_name
    assert model.api_key == api_key


def test_init_with_custom_config(model_name, api_key):
    """Test model initialization with custom configuration."""
    custom_config = {"organization": "org-123", "project": "proj-456"}
    model = OpenAIRealtimeBidirectionalModel(model=model_name, api_key=api_key, **custom_config)
    
    assert model.config == custom_config


def test_init_without_api_key_raises():
    """Test that initialization without API key raises error."""
    with unittest.mock.patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIRealtimeBidirectionalModel()


def test_init_with_env_api_key():
    """Test initialization with API key from environment."""
    with unittest.mock.patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
        model = OpenAIRealtimeBidirectionalModel()
        assert model.api_key == "env-key"


# Connection Tests


@pytest.mark.asyncio
async def test_connect_basic(mock_websockets_connect, model):
    """Test basic connection establishment."""
    mock_connect, mock_ws = mock_websockets_connect
    
    await model.connect()
    
    assert model._active is True
    assert model.session_id is not None
    assert model.websocket == mock_ws
    assert model._event_queue is not None
    mock_connect.assert_called_once()


@pytest.mark.asyncio
async def test_connect_with_system_prompt(mock_websockets_connect, model, system_prompt):
    """Test connection with system prompt."""
    _, mock_ws = mock_websockets_connect
    
    await model.connect(system_prompt=system_prompt)
    
    # Verify session.update was sent with system prompt
    calls = mock_ws.send.call_args_list
    session_update_call = None
    for call in calls:
        message = json.loads(call[0][0])
        if message.get("type") == "session.update":
            session_update_call = message
            break
    
    assert session_update_call is not None
    assert session_update_call["session"]["instructions"] == system_prompt


@pytest.mark.asyncio
async def test_connect_with_tools(mock_websockets_connect, model, tool_spec):
    """Test connection with tools."""
    _, mock_ws = mock_websockets_connect
    
    await model.connect(tools=[tool_spec])
    
    # Verify tools were included in session config
    calls = mock_ws.send.call_args_list
    session_update_call = None
    for call in calls:
        message = json.loads(call[0][0])
        if message.get("type") == "session.update":
            session_update_call = message
            break
    
    assert session_update_call is not None
    assert "tools" in session_update_call["session"]


@pytest.mark.asyncio
async def test_connect_with_messages(mock_websockets_connect, model, messages):
    """Test connection with message history."""
    _, mock_ws = mock_websockets_connect
    
    await model.connect(messages=messages)
    
    # Verify conversation items were created
    calls = mock_ws.send.call_args_list
    item_create_calls = [
        json.loads(call[0][0]) for call in calls
        if json.loads(call[0][0]).get("type") == "conversation.item.create"
    ]
    
    assert len(item_create_calls) > 0


@pytest.mark.asyncio
async def test_connect_error_handling(mock_websockets_connect, model):
    """Test connection error handling."""
    mock_connect, _ = mock_websockets_connect
    mock_connect.side_effect = Exception("Connection failed")
    
    with pytest.raises(Exception, match="Connection failed"):
        await model.connect()


@pytest.mark.asyncio
async def test_connect_with_organization_header(mock_websockets_connect, api_key):
    """Test connection includes organization header."""
    mock_connect, _ = mock_websockets_connect
    
    model = OpenAIRealtimeBidirectionalModel(
        api_key=api_key,
        organization="org-123"
    )
    await model.connect()
    
    # Verify headers were passed
    call_kwargs = mock_connect.call_args.kwargs
    headers = call_kwargs.get("additional_headers", [])
    org_header = [h for h in headers if h[0] == "OpenAI-Organization"]
    assert len(org_header) == 1
    assert org_header[0][1] == "org-123"


# Send Method Tests


@pytest.mark.asyncio
async def test_send_text_input(mock_websockets_connect, model):
    """Test sending text input through unified send() method."""
    _, mock_ws = mock_websockets_connect
    await model.connect()
    
    text_input: TextInputEvent = {"text": "Hello", "role": "user"}
    await model.send(text_input)
    
    # Verify conversation.item.create and response.create were sent
    calls = mock_ws.send.call_args_list
    messages = [json.loads(call[0][0]) for call in calls]
    
    item_create = [m for m in messages if m.get("type") == "conversation.item.create"]
    response_create = [m for m in messages if m.get("type") == "response.create"]
    
    assert len(item_create) > 0
    assert len(response_create) > 0


@pytest.mark.asyncio
async def test_send_audio_input(mock_websockets_connect, model):
    """Test sending audio input through unified send() method."""
    _, mock_ws = mock_websockets_connect
    await model.connect()
    
    audio_input: AudioInputEvent = {
        "audioData": b"audio_bytes",
        "format": "pcm",
        "sampleRate": 24000,
        "channels": 1,
    }
    await model.send(audio_input)
    
    # Verify input_audio_buffer.append was sent
    calls = mock_ws.send.call_args_list
    messages = [json.loads(call[0][0]) for call in calls]
    
    audio_append = [m for m in messages if m.get("type") == "input_audio_buffer.append"]
    assert len(audio_append) > 0
    
    # Verify audio was base64 encoded
    assert "audio" in audio_append[0]
    decoded = base64.b64decode(audio_append[0]["audio"])
    assert decoded == b"audio_bytes"


@pytest.mark.asyncio
async def test_send_image_input(mock_websockets_connect, model):
    """Test sending image input logs warning (not supported)."""
    _, mock_ws = mock_websockets_connect
    await model.connect()
    
    image_input: ImageInputEvent = {
        "imageData": b"image_bytes",
        "mimeType": "image/jpeg",
        "encoding": "raw",
    }
    
    with unittest.mock.patch("strands.experimental.bidirectional_streaming.models.openai.logger") as mock_logger:
        await model.send(image_input)
        mock_logger.warning.assert_called_with("Image input not supported by OpenAI Realtime API")


@pytest.mark.asyncio
async def test_send_tool_result(mock_websockets_connect, model):
    """Test sending tool result through unified send() method."""
    _, mock_ws = mock_websockets_connect
    await model.connect()
    
    tool_result: ToolResult = {
        "toolUseId": "tool-123",
        "status": "success",
        "content": [{"text": "Result: 42"}],
    }
    await model.send(tool_result)
    
    # Verify function_call_output was created
    calls = mock_ws.send.call_args_list
    messages = [json.loads(call[0][0]) for call in calls]
    
    item_create = [m for m in messages if m.get("type") == "conversation.item.create"]
    assert len(item_create) > 0
    
    # Verify it's a function_call_output
    item = item_create[-1].get("item", {})
    assert item.get("type") == "function_call_output"
    assert item.get("call_id") == "tool-123"


@pytest.mark.asyncio
async def test_send_when_inactive(mock_websockets_connect, model):
    """Test that send() does nothing when connection is inactive."""
    _, mock_ws = mock_websockets_connect
    
    # Don't connect, so _active is False
    text_input: TextInputEvent = {"text": "Hello", "role": "user"}
    await model.send(text_input)
    
    # Verify nothing was sent
    mock_ws.send.assert_not_called()


@pytest.mark.asyncio
async def test_send_unknown_content_type(mock_websockets_connect, model):
    """Test sending unknown content type logs warning."""
    _, _ = mock_websockets_connect
    await model.connect()
    
    unknown_content = {"unknown_field": "value"}
    
    with unittest.mock.patch("strands.experimental.bidirectional_streaming.models.openai.logger") as mock_logger:
        await model.send(unknown_content)
        # Should log warning about unknown content
        assert mock_logger.warning.called


# Receive Method Tests


@pytest.mark.asyncio
async def test_receive_connection_start_event(mock_websockets_connect, model):
    """Test that receive() emits connection start event."""
    _, _ = mock_websockets_connect
    
    await model.connect()
    
    # Get first event
    receive_gen = model.receive()
    first_event = await anext(receive_gen)
    
    # First event should be connection start
    assert "BidirectionalConnectionStart" in first_event
    assert first_event["BidirectionalConnectionStart"]["connectionId"] == model.session_id
    
    # Close to stop the loop
    await model.close()


@pytest.mark.asyncio
async def test_receive_connection_end_event(mock_websockets_connect, model):
    """Test that receive() emits connection end event."""
    _, _ = mock_websockets_connect
    
    await model.connect()
    
    # Collect events until connection ends
    events = []
    async for event in model.receive():
        events.append(event)
        # Close after first event to trigger connection end
        if len(events) == 1:
            await model.close()
    
    # Last event should be connection end
    assert "BidirectionalConnectionEnd" in events[-1]


@pytest.mark.asyncio
async def test_receive_audio_output(mock_websockets_connect, model):
    """Test receiving audio output from model."""
    _, _ = mock_websockets_connect
    await model.connect()
    
    # Create mock OpenAI event
    openai_event = {
        "type": "response.output_audio.delta",
        "delta": base64.b64encode(b"audio_data").decode()
    }
    
    # Test conversion directly
    converted_event = model._convert_openai_event(openai_event)
    
    assert "audioOutput" in converted_event
    assert converted_event["audioOutput"]["audioData"] == b"audio_data"
    assert converted_event["audioOutput"]["format"] == "pcm"


@pytest.mark.asyncio
async def test_receive_text_output(mock_websockets_connect, model):
    """Test receiving text output from model."""
    _, _ = mock_websockets_connect
    await model.connect()
    
    # Create mock OpenAI event
    openai_event = {
        "type": "response.output_text.delta",
        "delta": "Hello from OpenAI"
    }
    
    # Test conversion directly
    converted_event = model._convert_openai_event(openai_event)
    
    assert "textOutput" in converted_event
    assert converted_event["textOutput"]["text"] == "Hello from OpenAI"
    assert converted_event["textOutput"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_receive_function_call(mock_websockets_connect, model):
    """Test receiving function call from model."""
    _, _ = mock_websockets_connect
    await model.connect()
    
    # Simulate function call sequence
    # First: output_item.added with function name
    item_added = {
        "type": "response.output_item.added",
        "item": {
            "type": "function_call",
            "call_id": "call-123",
            "name": "calculator"
        }
    }
    model._convert_openai_event(item_added)
    
    # Second: function_call_arguments.delta
    args_delta = {
        "type": "response.function_call_arguments.delta",
        "call_id": "call-123",
        "delta": '{"expression": "2+2"}'
    }
    model._convert_openai_event(args_delta)
    
    # Third: function_call_arguments.done
    args_done = {
        "type": "response.function_call_arguments.done",
        "call_id": "call-123"
    }
    converted_event = model._convert_openai_event(args_done)
    
    assert "toolUse" in converted_event
    assert converted_event["toolUse"]["toolUseId"] == "call-123"
    assert converted_event["toolUse"]["name"] == "calculator"
    assert converted_event["toolUse"]["input"]["expression"] == "2+2"


@pytest.mark.asyncio
async def test_receive_voice_activity(mock_websockets_connect, model):
    """Test receiving voice activity events."""
    _, _ = mock_websockets_connect
    await model.connect()
    
    # Test speech started
    speech_started = {
        "type": "input_audio_buffer.speech_started"
    }
    converted_event = model._convert_openai_event(speech_started)
    
    assert "voiceActivity" in converted_event
    assert converted_event["voiceActivity"]["activityType"] == "speech_started"


# Close Method Tests


@pytest.mark.asyncio
async def test_close_connection(mock_websockets_connect, model):
    """Test closing connection."""
    _, mock_ws = mock_websockets_connect
    
    await model.connect()
    await model.close()
    
    assert model._active is False
    mock_ws.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_when_not_connected(mock_websockets_connect, model):
    """Test closing when not connected does nothing."""
    _, mock_ws = mock_websockets_connect
    
    # Don't connect
    await model.close()
    
    # Should not raise, and close should not be called
    mock_ws.close.assert_not_called()


@pytest.mark.asyncio
async def test_close_error_handling(mock_websockets_connect, model):
    """Test close error handling."""
    _, mock_ws = mock_websockets_connect
    mock_ws.close.side_effect = Exception("Close failed")
    
    await model.connect()
    
    # Should not raise, just log warning
    await model.close()
    assert model._active is False


@pytest.mark.asyncio
async def test_close_cancels_response_task(mock_websockets_connect, model):
    """Test that close cancels the background response task."""
    _, _ = mock_websockets_connect
    
    await model.connect()
    
    # Verify response task is running
    assert model._response_task is not None
    assert not model._response_task.done()
    
    await model.close()
    
    # Task should be cancelled
    assert model._response_task.cancelled() or model._response_task.done()


# Helper Method Tests


def test_build_session_config_basic(model):
    """Test building basic session config."""
    config = model._build_session_config(None, None)
    
    assert isinstance(config, dict)
    assert "instructions" in config
    assert "audio" in config


def test_build_session_config_with_system_prompt(model, system_prompt):
    """Test building config with system prompt."""
    config = model._build_session_config(system_prompt, None)
    
    assert config["instructions"] == system_prompt


def test_build_session_config_with_tools(model, tool_spec):
    """Test building config with tools."""
    config = model._build_session_config(None, [tool_spec])
    
    assert "tools" in config
    assert len(config["tools"]) > 0


def test_convert_tools_to_openai_format(model, tool_spec):
    """Test tool conversion to OpenAI format."""
    openai_tools = model._convert_tools_to_openai_format([tool_spec])
    
    assert len(openai_tools) == 1
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["name"] == "calculator"
    assert openai_tools[0]["description"] == "Calculate mathematical expressions"


def test_convert_tools_empty_list(model):
    """Test converting empty tool list."""
    openai_tools = model._convert_tools_to_openai_format([])
    
    assert openai_tools == []


@pytest.mark.asyncio
async def test_send_event(mock_websockets_connect, model):
    """Test sending event to WebSocket."""
    _, mock_ws = mock_websockets_connect
    await model.connect()
    
    test_event = {"type": "test.event", "data": "test"}
    await model._send_event(test_event)
    
    # Verify event was sent as JSON
    calls = mock_ws.send.call_args_list
    last_call = calls[-1]
    sent_message = json.loads(last_call[0][0])
    
    assert sent_message == test_event


def test_require_active(model):
    """Test _require_active method."""
    assert model._require_active() is False
    
    model._active = True
    assert model._require_active() is True


def test_create_text_event(model):
    """Test creating text event."""
    event = model._create_text_event("Hello", "user")
    
    assert "textOutput" in event
    assert event["textOutput"]["text"] == "Hello"
    assert event["textOutput"]["role"] == "user"


def test_create_voice_activity_event(model):
    """Test creating voice activity event."""
    event = model._create_voice_activity_event("speech_started")
    
    assert "voiceActivity" in event
    assert event["voiceActivity"]["activityType"] == "speech_started"
