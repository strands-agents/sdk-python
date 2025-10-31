"""Unit tests for Gemini Live bidirectional streaming model.

Tests the unified GeminiLiveModel interface including:
- Model initialization and configuration
- Connection establishment and lifecycle
- Unified send() method with different content types
- Event receiving and conversion
"""

import unittest.mock

import pytest
from google import genai
from google.genai import types as genai_types

from strands.experimental.bidirectional_streaming.models.gemini_live import GeminiLiveModel
from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import (
    AudioInputEvent,
    ImageInputEvent,
    TextInputEvent,
)
from strands.types._events import ToolResultEvent
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
    """Create a GeminiLiveModel instance."""
    _ = mock_genai_client
    return GeminiLiveModel(model_id=model_id, api_key=api_key)


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


def test_model_initialization(mock_genai_client, model_id, api_key):
    """Test model initialization with various configurations."""
    _ = mock_genai_client
    
    # Test default config
    model_default = GeminiLiveModel()
    assert model_default.model_id == "models/gemini-2.0-flash-live-preview-04-09"
    assert model_default.api_key is None
    assert model_default._active is False
    assert model_default.live_session is None
    
    # Test with API key
    model_with_key = GeminiLiveModel(model_id=model_id, api_key=api_key)
    assert model_with_key.model_id == model_id
    assert model_with_key.api_key == api_key
    
    # Test with custom config
    live_config = {"temperature": 0.7, "top_p": 0.9}
    model_custom = GeminiLiveModel(model_id=model_id, live_config=live_config)
    assert model_custom.live_config == live_config


# Connection Tests


@pytest.mark.asyncio
async def test_connection_lifecycle(mock_genai_client, model, system_prompt, tool_spec, messages):
    """Test complete connection lifecycle with various configurations."""
    mock_client, mock_live_session, mock_live_session_cm = mock_genai_client
    
    # Test basic connection
    await model.connect()
    assert model._active is True
    assert model.session_id is not None
    assert model.live_session == mock_live_session
    mock_client.aio.live.connect.assert_called_once()
    
    # Test close
    await model.close()
    assert model._active is False
    mock_live_session_cm.__aexit__.assert_called_once()
    
    # Test connection with system prompt
    await model.connect(system_prompt=system_prompt)
    call_args = mock_client.aio.live.connect.call_args
    config = call_args.kwargs.get("config", {})
    assert config.get("system_instruction") == system_prompt
    await model.close()
    
    # Test connection with tools
    await model.connect(tools=[tool_spec])
    call_args = mock_client.aio.live.connect.call_args
    config = call_args.kwargs.get("config", {})
    assert "tools" in config
    assert len(config["tools"]) > 0
    await model.close()
    
    # Test connection with messages
    await model.connect(messages=messages)
    mock_live_session.send_client_content.assert_called()
    await model.close()


@pytest.mark.asyncio
async def test_connection_edge_cases(mock_genai_client, api_key, model_id):
    """Test connection error handling and edge cases."""
    mock_client, _, mock_live_session_cm = mock_genai_client
    
    # Test connection error
    model1 = GeminiLiveModel(model_id=model_id, api_key=api_key)
    mock_client.aio.live.connect.side_effect = Exception("Connection failed")
    with pytest.raises(Exception, match="Connection failed"):
        await model1.connect()
    
    # Reset mock for next tests
    mock_client.aio.live.connect.side_effect = None
    
    # Test double connection
    model2 = GeminiLiveModel(model_id=model_id, api_key=api_key)
    await model2.connect()
    with pytest.raises(RuntimeError, match="Connection already active"):
        await model2.connect()
    await model2.close()
    
    # Test close when not connected
    model3 = GeminiLiveModel(model_id=model_id, api_key=api_key)
    await model3.close()  # Should not raise
    
    # Test close error handling
    model4 = GeminiLiveModel(model_id=model_id, api_key=api_key)
    await model4.connect()
    mock_live_session_cm.__aexit__.side_effect = Exception("Close failed")
    with pytest.raises(Exception, match="Close failed"):
        await model4.close()


# Send Method Tests


@pytest.mark.asyncio
async def test_send_all_content_types(mock_genai_client, model):
    """Test sending all content types through unified send() method."""
    _, mock_live_session, _ = mock_genai_client
    await model.connect()
    
    # Test text input
    text_input = TextInputEvent(text="Hello", role="user")
    await model.send(text_input)
    mock_live_session.send_client_content.assert_called_once()
    call_args = mock_live_session.send_client_content.call_args
    content = call_args.kwargs.get("turns")
    assert content.role == "user"
    assert content.parts[0].text == "Hello"
    
    # Test audio input
    audio_input = AudioInputEvent(
        audio=b"audio_bytes",
        format="pcm",
        sample_rate=16000,
        channels=1,
    )
    await model.send(audio_input)
    mock_live_session.send_realtime_input.assert_called_once()
    
    # Test image input
    image_input = ImageInputEvent(
        image=b"image_bytes",
        mime_type="image/jpeg",
        encoding="raw",
    )
    await model.send(image_input)
    mock_live_session.send.assert_called_once()
    
    # Test tool result
    from strands.types._events import ToolResultEvent
    tool_result: ToolResult = {
        "toolUseId": "tool-123",
        "status": "success",
        "content": [{"text": "Result: 42"}],
    }
    await model.send(ToolResultEvent(tool_result))
    mock_live_session.send_tool_response.assert_called_once()
    
    await model.close()


@pytest.mark.asyncio
async def test_send_edge_cases(mock_genai_client, model):
    """Test send() edge cases and error handling."""
    _, mock_live_session, _ = mock_genai_client
    
    # Test send when inactive
    text_input = TextInputEvent(text="Hello", role="user")
    await model.send(text_input)
    mock_live_session.send_client_content.assert_not_called()
    
    # Test unknown content type
    await model.connect()
    unknown_content = {"unknown_field": "value"}
    await model.send(unknown_content)  # Should not raise, just log warning
    
    await model.close()


# Receive Method Tests


@pytest.mark.asyncio
async def test_receive_lifecycle_events(mock_genai_client, model, agenerator):
    """Test that receive() emits connection start and end events."""
    from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import (
        SessionStartEvent,
        SessionEndEvent,
    )
    
    _, mock_live_session, _ = mock_genai_client
    mock_live_session.receive.return_value = agenerator([])
    
    await model.connect()
    
    # Collect events
    events = []
    async for event in model.receive():
        events.append(event)
        # Close after first event to trigger connection end
        if len(events) == 1:
            await model.close()
    
    # Verify connection start and end
    assert len(events) >= 2
    assert isinstance(events[0], SessionStartEvent)
    assert events[0].session_id == model.session_id
    assert isinstance(events[-1], SessionEndEvent)


@pytest.mark.asyncio
async def test_event_conversion(mock_genai_client, model):
    """Test conversion of all Gemini Live event types to standard format."""
    from strands.experimental.bidirectional_streaming.types.bidirectional_streaming import (
        TranscriptStreamEvent,
        AudioStreamEvent,
        InterruptionEvent,
    )
    
    _, _, _ = mock_genai_client
    await model.connect()
    
    # Test text output (now converted to transcript)
    mock_text = unittest.mock.Mock()
    mock_text.text = "Hello from Gemini"
    mock_text.data = None
    mock_text.tool_call = None
    mock_text.server_content = None
    
    text_event = model._convert_gemini_live_event(mock_text)
    assert isinstance(text_event, TranscriptStreamEvent)
    assert text_event.text == "Hello from Gemini"
    assert text_event.source == "assistant"
    assert text_event.is_final is True
    
    # Test audio output
    mock_audio = unittest.mock.Mock()
    mock_audio.text = None
    mock_audio.data = b"audio_data"
    mock_audio.tool_call = None
    mock_audio.server_content = None
    
    audio_event = model._convert_gemini_live_event(mock_audio)
    assert isinstance(audio_event, AudioStreamEvent)
    assert audio_event.audio == b"audio_data"
    assert audio_event.format == "pcm"
    
    # Test tool call
    mock_func_call = unittest.mock.Mock()
    mock_func_call.id = "tool-123"
    mock_func_call.name = "calculator"
    mock_func_call.args = {"expression": "2+2"}
    
    mock_tool_call = unittest.mock.Mock()
    mock_tool_call.function_calls = [mock_func_call]
    
    mock_tool = unittest.mock.Mock()
    mock_tool.text = None
    mock_tool.data = None
    mock_tool.tool_call = mock_tool_call
    mock_tool.server_content = None
    
    tool_event = model._convert_gemini_live_event(mock_tool)
    assert "toolUse" in tool_event
    assert tool_event["toolUse"]["toolUseId"] == "tool-123"
    assert tool_event["toolUse"]["name"] == "calculator"
    
    # Test interruption
    mock_server_content = unittest.mock.Mock()
    mock_server_content.interrupted = True
    mock_server_content.input_transcription = None
    mock_server_content.output_transcription = None
    
    mock_interrupt = unittest.mock.Mock()
    mock_interrupt.text = None
    mock_interrupt.data = None
    mock_interrupt.tool_call = None
    mock_interrupt.server_content = mock_server_content
    
    interrupt_event = model._convert_gemini_live_event(mock_interrupt)
    assert isinstance(interrupt_event, InterruptionEvent)
    assert interrupt_event.reason == "user_speech"
    
    await model.close()


# Helper Method Tests


def test_config_building(model, system_prompt, tool_spec):
    """Test building live config with various options."""
    # Test basic config
    config_basic = model._build_live_config()
    assert isinstance(config_basic, dict)
    
    # Test with system prompt
    config_prompt = model._build_live_config(system_prompt=system_prompt)
    assert config_prompt["system_instruction"] == system_prompt
    
    # Test with tools
    config_tools = model._build_live_config(tools=[tool_spec])
    assert "tools" in config_tools
    assert len(config_tools["tools"]) > 0


def test_tool_formatting(model, tool_spec):
    """Test tool formatting for Gemini Live API."""
    # Test with tools
    formatted_tools = model._format_tools_for_live_api([tool_spec])
    assert len(formatted_tools) == 1
    assert isinstance(formatted_tools[0], genai_types.Tool)
    
    # Test empty list
    formatted_empty = model._format_tools_for_live_api([])
    assert formatted_empty == []
