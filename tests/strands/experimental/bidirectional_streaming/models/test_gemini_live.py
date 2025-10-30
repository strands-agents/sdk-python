"""Unit tests for Gemini Live bidirectional streaming model.

Tests the unified GeminiLiveBidirectionalModel interface including:
- Model initialization and configuration
- Connection establishment
- Unified send() method with different content types
- Event receiving and conversion
- Connection lifecycle management
"""

import unittest.mock
import uuid

import pytest
from google import genai
from google.genai import types as genai_types

from strands.experimental.bidirectional_streaming.models.gemini_live import GeminiLiveBidirectionalModel
from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import (
    AudioInputEvent,
    ImageInputEvent,
    TextInputEvent,
)
from strands.types.tools import ToolResult


@pytest.fixture
def mock_genai_client():
    """Mock the Google GenAI client."""
    with unittest.mock.patch("strands.experimental.bidirectional_streaming.models.gemini_live.genai.Client") as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.aio = unittest.mock.MagicMock()
        
        # Mock the live session
        mock_live_session = unittest.mock.AsyncMock()
        
        # Mock the context manager
        mock_live_session_cm = unittest.mock.MagicMock()
        mock_live_session_cm.__aenter__ = unittest.mock.AsyncMock(return_value=mock_live_session)
        mock_live_session_cm.__aexit__ = unittest.mock.AsyncMock(return_value=None)
        
        # Make connect return the context manager
        mock_client.aio.live.connect = unittest.mock.MagicMock(return_value=mock_live_session_cm)
        
        yield mock_client, mock_live_session, mock_live_session_cm


@pytest.fixture
def model_id():
    return "models/gemini-2.0-flash-live-preview-04-09"


@pytest.fixture
def api_key():
    return "test-api-key"


@pytest.fixture
def model(mock_genai_client, model_id, api_key):
    """Create a GeminiLiveBidirectionalModel instance."""
    _ = mock_genai_client
    return GeminiLiveBidirectionalModel(model_id=model_id, api_key=api_key)


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


def test_init_default_config(mock_genai_client):
    """Test model initialization with default configuration."""
    _ = mock_genai_client
    
    model = GeminiLiveBidirectionalModel()
    
    assert model.model_id == "models/gemini-2.0-flash-live-preview-04-09"
    assert model.api_key is None
    assert model._active is False
    assert model.live_session is None


def test_init_with_api_key(mock_genai_client, model_id, api_key):
    """Test model initialization with API key."""
    mock_client, _, _ = mock_genai_client
    
    model = GeminiLiveBidirectionalModel(model_id=model_id, api_key=api_key)
    
    assert model.model_id == model_id
    assert model.api_key == api_key
    
    # Verify client was created with correct parameters
    mock_client_cls = unittest.mock.patch("strands.experimental.bidirectional_streaming.models.gemini_live.genai.Client").start()
    GeminiLiveBidirectionalModel(model_id=model_id, api_key=api_key)
    mock_client_cls.assert_called()


def test_init_with_custom_config(mock_genai_client, model_id):
    """Test model initialization with custom configuration."""
    _ = mock_genai_client
    
    custom_config = {"temperature": 0.7, "top_p": 0.9}
    model = GeminiLiveBidirectionalModel(model_id=model_id, **custom_config)
    
    assert model.config == custom_config


# Connection Tests


@pytest.mark.asyncio
async def test_connect_basic(mock_genai_client, model):
    """Test basic connection establishment."""
    mock_client, mock_live_session, _ = mock_genai_client
    
    await model.connect()
    
    assert model._active is True
    assert model.session_id is not None
    assert model.live_session == mock_live_session
    mock_client.aio.live.connect.assert_called_once()


@pytest.mark.asyncio
async def test_connect_with_system_prompt(mock_genai_client, model, system_prompt):
    """Test connection with system prompt."""
    mock_client, _, _ = mock_genai_client
    
    await model.connect(system_prompt=system_prompt)
    
    # Verify system prompt was included in config
    call_args = mock_client.aio.live.connect.call_args
    config = call_args.kwargs.get("config", {})
    assert config.get("system_instruction") == system_prompt


@pytest.mark.asyncio
async def test_connect_with_tools(mock_genai_client, model, tool_spec):
    """Test connection with tools."""
    mock_client, _, _ = mock_genai_client
    
    await model.connect(tools=[tool_spec])
    
    # Verify tools were formatted and included
    call_args = mock_client.aio.live.connect.call_args
    config = call_args.kwargs.get("config", {})
    assert "tools" in config
    assert len(config["tools"]) > 0


@pytest.mark.asyncio
async def test_connect_with_messages(mock_genai_client, model, messages):
    """Test connection with message history."""
    _, mock_live_session, _ = mock_genai_client
    
    await model.connect(messages=messages)
    
    # Verify message history was sent
    mock_live_session.send_client_content.assert_called()


@pytest.mark.asyncio
async def test_connect_error_handling(mock_genai_client, model):
    """Test connection error handling."""
    mock_client, _, _ = mock_genai_client
    mock_client.aio.live.connect.side_effect = Exception("Connection failed")
    
    with pytest.raises(Exception, match="Connection failed"):
        await model.connect()


# Send Method Tests


@pytest.mark.asyncio
async def test_send_text_input(mock_genai_client, model):
    """Test sending text input through unified send() method."""
    _, mock_live_session, _ = mock_genai_client
    await model.connect()
    
    text_input: TextInputEvent = {"text": "Hello", "role": "user"}
    await model.send(text_input)
    
    # Verify text was sent via send_client_content
    mock_live_session.send_client_content.assert_called_once()
    call_args = mock_live_session.send_client_content.call_args
    content = call_args.kwargs.get("turns")
    assert content.role == "user"
    assert content.parts[0].text == "Hello"


@pytest.mark.asyncio
async def test_send_audio_input(mock_genai_client, model):
    """Test sending audio input through unified send() method."""
    _, mock_live_session, _ = mock_genai_client
    await model.connect()
    
    audio_input: AudioInputEvent = {
        "audioData": b"audio_bytes",
        "format": "pcm",
        "sampleRate": 16000,
        "channels": 1,
    }
    await model.send(audio_input)
    
    # Verify audio was sent via send_realtime_input
    mock_live_session.send_realtime_input.assert_called_once()


@pytest.mark.asyncio
async def test_send_image_input(mock_genai_client, model):
    """Test sending image input through unified send() method."""
    _, mock_live_session, _ = mock_genai_client
    await model.connect()
    
    image_input: ImageInputEvent = {
        "imageData": b"image_bytes",
        "mimeType": "image/jpeg",
        "encoding": "raw",
    }
    await model.send(image_input)
    
    # Verify image was sent
    mock_live_session.send.assert_called_once()


@pytest.mark.asyncio
async def test_send_tool_result(mock_genai_client, model):
    """Test sending tool result through unified send() method."""
    _, mock_live_session, _ = mock_genai_client
    await model.connect()
    
    tool_result: ToolResult = {
        "toolUseId": "tool-123",
        "status": "success",
        "content": [{"text": "Result: 42"}],
    }
    await model.send(tool_result)
    
    # Verify tool result was sent
    mock_live_session.send_tool_response.assert_called_once()


@pytest.mark.asyncio
async def test_send_when_inactive(mock_genai_client, model):
    """Test that send() does nothing when connection is inactive."""
    _, mock_live_session, _ = mock_genai_client
    
    # Don't connect, so _active is False
    text_input: TextInputEvent = {"text": "Hello", "role": "user"}
    await model.send(text_input)
    
    # Verify nothing was sent
    mock_live_session.send_client_content.assert_not_called()


@pytest.mark.asyncio
async def test_send_unknown_content_type(mock_genai_client, model):
    """Test sending unknown content type logs warning."""
    _, _, _ = mock_genai_client
    await model.connect()
    
    unknown_content = {"unknown_field": "value"}
    
    # Should not raise, just log warning
    await model.send(unknown_content)


# Receive Method Tests


@pytest.mark.asyncio
async def test_receive_connection_start_event(mock_genai_client, model, agenerator):
    """Test that receive() emits connection start event."""
    _, mock_live_session, _ = mock_genai_client
    mock_live_session.receive.return_value = agenerator([])
    
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
async def test_receive_connection_end_event(mock_genai_client, model, agenerator):
    """Test that receive() emits connection end event."""
    _, mock_live_session, _ = mock_genai_client
    mock_live_session.receive.return_value = agenerator([])
    
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
async def test_receive_text_output(mock_genai_client, model):
    """Test receiving text output from model."""
    _, mock_live_session, _ = mock_genai_client
    
    mock_message = unittest.mock.Mock()
    mock_message.text = "Hello from Gemini"
    mock_message.data = None
    mock_message.tool_call = None
    mock_message.server_content = None
    
    await model.connect()
    
    # Test the conversion method directly
    converted_event = model._convert_gemini_live_event(mock_message)
    
    assert "textOutput" in converted_event
    assert converted_event["textOutput"]["text"] == "Hello from Gemini"
    assert converted_event["textOutput"]["role"] == "assistant"


@pytest.mark.asyncio
async def test_receive_audio_output(mock_genai_client, model):
    """Test receiving audio output from model."""
    _, mock_live_session, _ = mock_genai_client
    
    mock_message = unittest.mock.Mock()
    mock_message.text = None
    mock_message.data = b"audio_data"
    mock_message.tool_call = None
    mock_message.server_content = None
    
    await model.connect()
    
    # Test the conversion method directly
    converted_event = model._convert_gemini_live_event(mock_message)
    
    assert "audioOutput" in converted_event
    assert converted_event["audioOutput"]["audioData"] == b"audio_data"
    assert converted_event["audioOutput"]["format"] == "pcm"


@pytest.mark.asyncio
async def test_receive_tool_call(mock_genai_client, model):
    """Test receiving tool call from model."""
    _, mock_live_session, _ = mock_genai_client
    
    mock_func_call = unittest.mock.Mock()
    mock_func_call.id = "tool-123"
    mock_func_call.name = "calculator"
    mock_func_call.args = {"expression": "2+2"}
    
    mock_tool_call = unittest.mock.Mock()
    mock_tool_call.function_calls = [mock_func_call]
    
    mock_message = unittest.mock.Mock()
    mock_message.text = None
    mock_message.data = None
    mock_message.tool_call = mock_tool_call
    mock_message.server_content = None
    
    await model.connect()
    
    # Test the conversion method directly
    converted_event = model._convert_gemini_live_event(mock_message)
    
    assert "toolUse" in converted_event
    assert converted_event["toolUse"]["toolUseId"] == "tool-123"
    assert converted_event["toolUse"]["name"] == "calculator"


@pytest.mark.asyncio
async def test_receive_interruption(mock_genai_client, model):
    """Test receiving interruption event."""
    _, mock_live_session, _ = mock_genai_client
    
    mock_server_content = unittest.mock.Mock()
    mock_server_content.interrupted = True
    mock_server_content.input_transcription = None
    mock_server_content.output_transcription = None
    
    mock_message = unittest.mock.Mock()
    mock_message.text = None
    mock_message.data = None
    mock_message.tool_call = None
    mock_message.server_content = mock_server_content
    
    await model.connect()
    
    # Test the conversion method directly
    converted_event = model._convert_gemini_live_event(mock_message)
    
    assert "interruptionDetected" in converted_event
    assert converted_event["interruptionDetected"]["reason"] == "user_input"


# Close Method Tests


@pytest.mark.asyncio
async def test_close_connection(mock_genai_client, model):
    """Test closing connection."""
    _, _, mock_live_session_cm = mock_genai_client
    
    await model.connect()
    await model.close()
    
    assert model._active is False
    mock_live_session_cm.__aexit__.assert_called_once()


@pytest.mark.asyncio
async def test_close_when_not_connected(mock_genai_client, model):
    """Test closing when not connected does nothing."""
    _, _, mock_live_session_cm = mock_genai_client
    
    # Don't connect
    await model.close()
    
    # Should not raise, and __aexit__ should not be called
    mock_live_session_cm.__aexit__.assert_not_called()


@pytest.mark.asyncio
async def test_close_error_handling(mock_genai_client, model):
    """Test close error handling."""
    _, _, mock_live_session_cm = mock_genai_client
    mock_live_session_cm.__aexit__.side_effect = Exception("Close failed")
    
    await model.connect()
    
    with pytest.raises(Exception, match="Close failed"):
        await model.close()


# Helper Method Tests


def test_build_live_config_basic(model):
    """Test building basic live config."""
    config = model._build_live_config()
    
    assert isinstance(config, dict)


def test_build_live_config_with_system_prompt(model, system_prompt):
    """Test building config with system prompt."""
    config = model._build_live_config(system_prompt=system_prompt)
    
    assert config["system_instruction"] == system_prompt


def test_build_live_config_with_tools(model, tool_spec):
    """Test building config with tools."""
    config = model._build_live_config(tools=[tool_spec])
    
    assert "tools" in config
    assert len(config["tools"]) > 0


def test_format_tools_for_live_api(model, tool_spec):
    """Test tool formatting for Gemini Live API."""
    formatted_tools = model._format_tools_for_live_api([tool_spec])
    
    assert len(formatted_tools) == 1
    assert isinstance(formatted_tools[0], genai_types.Tool)


def test_format_tools_empty_list(model):
    """Test formatting empty tool list."""
    formatted_tools = model._format_tools_for_live_api([])
    
    assert formatted_tools == []
