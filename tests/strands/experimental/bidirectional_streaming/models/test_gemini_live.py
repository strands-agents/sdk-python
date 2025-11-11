"""Unit tests for Gemini Live bidirectional streaming model.

Tests the unified BidiGeminiLiveModel interface including:
- Model initialization and configuration
- Connection establishment and lifecycle
- Unified send() method with different content types
- Event receiving and conversion
"""

import base64
import json
import unittest.mock

import pytest
from google import genai
from google.genai import types as genai_types

from strands.experimental.bidirectional_streaming.models.gemini_live import BidiGeminiLiveModel
from strands.experimental.bidirectional_streaming.types.events import (
    BidiAudioInputEvent,
    BidiAudioStreamEvent,
    BidiConnectionCloseEvent,
    BidiConnectionStartEvent,
    BidiImageInputEvent,
    BidiInterruptionEvent,
    BidiTextInputEvent,
    BidiTranscriptStreamEvent,
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
    """Create a BidiGeminiLiveModel instance."""
    _ = mock_genai_client
    return BidiGeminiLiveModel(model_id=model_id, api_key=api_key)


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
    model_default = BidiGeminiLiveModel()
    assert model_default.model_id == "gemini-2.5-flash-native-audio-preview-09-2025"
    assert model_default.api_key is None
    assert model_default._active is False
    assert model_default.live_session is None
    # Check default config includes transcription
    assert model_default.live_config["response_modalities"] == ["AUDIO"]
    assert "outputAudioTranscription" in model_default.live_config
    assert "inputAudioTranscription" in model_default.live_config
    
    # Test with API key
    model_with_key = BidiGeminiLiveModel(model_id=model_id, api_key=api_key)
    assert model_with_key.model_id == model_id
    assert model_with_key.api_key == api_key
    
    # Test with custom config (merges with defaults)
    live_config = {"temperature": 0.7, "top_p": 0.9}
    model_custom = BidiGeminiLiveModel(model_id=model_id, live_config=live_config)
    # Custom config should be merged with defaults
    assert model_custom.live_config["temperature"] == 0.7
    assert model_custom.live_config["top_p"] == 0.9
    # Defaults should still be present
    assert "response_modalities" in model_custom.live_config


# Connection Tests


@pytest.mark.asyncio
async def test_connection_lifecycle(mock_genai_client, model, system_prompt, tool_spec, messages):
    """Test complete connection lifecycle with various configurations."""
    mock_client, mock_live_session, mock_live_session_cm = mock_genai_client
    
    # Test basic connection
    await model.start()
    assert model._active is True
    assert model.connection_id is not None
    assert model.live_session == mock_live_session
    mock_client.aio.live.connect.assert_called_once()
    
    # Test close
    await model.stop()
    assert model._active is False
    mock_live_session_cm.__aexit__.assert_called_once()
    
    # Test connection with system prompt
    await model.start(system_prompt=system_prompt)
    call_args = mock_client.aio.live.connect.call_args
    config = call_args.kwargs.get("config", {})
    assert config.get("system_instruction") == system_prompt
    await model.stop()
    
    # Test connection with tools
    await model.start(tools=[tool_spec])
    call_args = mock_client.aio.live.connect.call_args
    config = call_args.kwargs.get("config", {})
    assert "tools" in config
    assert len(config["tools"]) > 0
    await model.stop()
    
    # Test connection with messages
    await model.start(messages=messages)
    mock_live_session.send_client_content.assert_called()
    await model.stop()


@pytest.mark.asyncio
async def test_connection_edge_cases(mock_genai_client, api_key, model_id):
    """Test connection error handling and edge cases."""
    mock_client, _, mock_live_session_cm = mock_genai_client
    
    # Test connection error
    model1 = BidiGeminiLiveModel(model_id=model_id, api_key=api_key)
    mock_client.aio.live.connect.side_effect = Exception("Connection failed")
    with pytest.raises(Exception, match="Connection failed"):
        await model1.start()
    
    # Reset mock for next tests
    mock_client.aio.live.connect.side_effect = None
    
    # Test double connection
    model2 = BidiGeminiLiveModel(model_id=model_id, api_key=api_key)
    await model2.start()
    with pytest.raises(RuntimeError, match="Connection already active"):
        await model2.start()
    await model2.stop()
    
    # Test close when not connected
    model3 = BidiGeminiLiveModel(model_id=model_id, api_key=api_key)
    await model3.stop()  # Should not raise
    
    # Test close error handling
    model4 = BidiGeminiLiveModel(model_id=model_id, api_key=api_key)
    await model4.start()
    mock_live_session_cm.__aexit__.side_effect = Exception("Close failed")
    with pytest.raises(Exception, match="Close failed"):
        await model4.stop()


# Send Method Tests


@pytest.mark.asyncio
async def test_send_all_content_types(mock_genai_client, model):
    """Test sending all content types through unified send() method."""
    _, mock_live_session, _ = mock_genai_client
    await model.start()
    
    # Test text input
    text_input = BidiTextInputEvent(text="Hello", role="user")
    await model.send(text_input)
    mock_live_session.send_client_content.assert_called_once()
    call_args = mock_live_session.send_client_content.call_args
    content = call_args.kwargs.get("turns")
    assert content.role == "user"
    assert content.parts[0].text == "Hello"
    
    # Test audio input (base64 encoded)
    audio_b64 = base64.b64encode(b"audio_bytes").decode('utf-8')
    audio_input = BidiAudioInputEvent(
        audio=audio_b64,
        format="pcm",
        sample_rate=16000,
        channels=1,
    )
    await model.send(audio_input)
    mock_live_session.send_realtime_input.assert_called_once()
    
    # Test image input (base64 encoded, no encoding parameter)
    image_b64 = base64.b64encode(b"image_bytes").decode('utf-8')
    image_input = BidiImageInputEvent(
        image=image_b64,
        mime_type="image/jpeg",
    )
    await model.send(image_input)
    mock_live_session.send.assert_called_once()
    
    # Test tool result
    tool_result: ToolResult = {
        "toolUseId": "tool-123",
        "status": "success",
        "content": [{"text": "Result: 42"}],
    }
    await model.send(ToolResultEvent(tool_result))
    mock_live_session.send_tool_response.assert_called_once()
    
    await model.stop()


@pytest.mark.asyncio
async def test_send_edge_cases(mock_genai_client, model):
    """Test send() edge cases and error handling."""
    _, mock_live_session, _ = mock_genai_client
    
    # Test send when inactive
    text_input = BidiTextInputEvent(text="Hello", role="user")
    await model.send(text_input)
    mock_live_session.send_client_content.assert_not_called()
    
    # Test unknown content type
    await model.start()
    unknown_content = {"unknown_field": "value"}
    await model.send(unknown_content)  # Should not raise, just log warning
    
    await model.stop()


# Receive Method Tests


@pytest.mark.asyncio
async def test_receive_lifecycle_events(mock_genai_client, model, agenerator):
    """Test that receive() emits connection start and end events."""
    _, mock_live_session, _ = mock_genai_client
    mock_live_session.receive.return_value = agenerator([])
    
    await model.start()
    
    # Collect events
    events = []
    async for event in model.receive():
        events.append(event)
        # Close after first event to trigger connection end
        if len(events) == 1:
            await model.stop()
    
    # Verify connection start and end
    assert len(events) >= 2
    assert isinstance(events[0], BidiConnectionStartEvent)
    assert events[0].get("type") == "bidi_connection_start"
    assert events[0].connection_id == model.connection_id
    assert isinstance(events[-1], BidiConnectionCloseEvent)
    assert events[-1].get("type") == "bidi_connection_close"


@pytest.mark.asyncio
async def test_event_conversion(mock_genai_client, model):
    """Test conversion of all Gemini Live event types to standard format."""
    _, _, _ = mock_genai_client
    await model.start()
    
    # Test text output (converted to transcript via model_turn.parts)
    mock_text = unittest.mock.Mock()
    mock_text.data = None
    mock_text.tool_call = None
    
    # Create proper server_content structure with model_turn
    mock_server_content = unittest.mock.Mock()
    mock_server_content.interrupted = False
    mock_server_content.input_transcription = None
    mock_server_content.output_transcription = None
    
    mock_model_turn = unittest.mock.Mock()
    mock_part = unittest.mock.Mock()
    mock_part.text = "Hello from Gemini"
    mock_model_turn.parts = [mock_part]
    mock_server_content.model_turn = mock_model_turn
    
    mock_text.server_content = mock_server_content
    
    text_events = model._convert_gemini_live_event(mock_text)
    assert isinstance(text_events, list)
    assert len(text_events) == 1
    text_event = text_events[0]
    assert isinstance(text_event, BidiTranscriptStreamEvent)
    assert text_event.get("type") == "bidi_transcript_stream"
    assert text_event.text == "Hello from Gemini"
    assert text_event.role == "assistant"
    assert text_event.is_final is True
    assert text_event.delta == {"text": "Hello from Gemini"}
    assert text_event.current_transcript == "Hello from Gemini"
    
    # Test multiple text parts (should concatenate)
    mock_multi_text = unittest.mock.Mock()
    mock_multi_text.data = None
    mock_multi_text.tool_call = None
    
    mock_server_content_multi = unittest.mock.Mock()
    mock_server_content_multi.interrupted = False
    mock_server_content_multi.input_transcription = None
    mock_server_content_multi.output_transcription = None
    
    mock_model_turn_multi = unittest.mock.Mock()
    mock_part1 = unittest.mock.Mock()
    mock_part1.text = "Hello"
    mock_part2 = unittest.mock.Mock()
    mock_part2.text = "from Gemini"
    mock_model_turn_multi.parts = [mock_part1, mock_part2]
    mock_server_content_multi.model_turn = mock_model_turn_multi
    
    mock_multi_text.server_content = mock_server_content_multi
    
    multi_text_events = model._convert_gemini_live_event(mock_multi_text)
    assert isinstance(multi_text_events, list)
    assert len(multi_text_events) == 1
    multi_text_event = multi_text_events[0]
    assert isinstance(multi_text_event, BidiTranscriptStreamEvent)
    assert multi_text_event.text == "Hello from Gemini"  # Concatenated with space
    
    # Test audio output (base64 encoded)
    mock_audio = unittest.mock.Mock()
    mock_audio.text = None
    mock_audio.data = b"audio_data"
    mock_audio.tool_call = None
    mock_audio.server_content = None
    
    audio_events = model._convert_gemini_live_event(mock_audio)
    assert isinstance(audio_events, list)
    assert len(audio_events) == 1
    audio_event = audio_events[0]
    assert isinstance(audio_event, BidiAudioStreamEvent)
    assert audio_event.get("type") == "bidi_audio_stream"
    # Audio is now base64 encoded
    expected_b64 = base64.b64encode(b"audio_data").decode('utf-8')
    assert audio_event.audio == expected_b64
    assert audio_event.format == "pcm"
    
    # Test single tool call (returns list with one event)
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
    
    tool_events = model._convert_gemini_live_event(mock_tool)
    # Should return a list of ToolUseStreamEvent
    assert isinstance(tool_events, list)
    assert len(tool_events) == 1
    tool_event = tool_events[0]
    # ToolUseStreamEvent has delta and current_tool_use, not a "type" field
    assert "delta" in tool_event
    assert "toolUse" in tool_event["delta"]
    assert tool_event["delta"]["toolUse"]["toolUseId"] == "tool-123"
    assert tool_event["delta"]["toolUse"]["name"] == "calculator"
    
    # Test multiple tool calls (returns list with multiple events)
    mock_func_call_1 = unittest.mock.Mock()
    mock_func_call_1.id = "tool-123"
    mock_func_call_1.name = "calculator"
    mock_func_call_1.args = {"expression": "2+2"}
    
    mock_func_call_2 = unittest.mock.Mock()
    mock_func_call_2.id = "tool-456"
    mock_func_call_2.name = "weather"
    mock_func_call_2.args = {"location": "Seattle"}
    
    mock_tool_call_multi = unittest.mock.Mock()
    mock_tool_call_multi.function_calls = [mock_func_call_1, mock_func_call_2]
    
    mock_tool_multi = unittest.mock.Mock()
    mock_tool_multi.text = None
    mock_tool_multi.data = None
    mock_tool_multi.tool_call = mock_tool_call_multi
    mock_tool_multi.server_content = None
    
    tool_events_multi = model._convert_gemini_live_event(mock_tool_multi)
    # Should return a list with two ToolUseStreamEvent
    assert isinstance(tool_events_multi, list)
    assert len(tool_events_multi) == 2
    
    # Verify first tool call
    assert tool_events_multi[0]["delta"]["toolUse"]["toolUseId"] == "tool-123"
    assert tool_events_multi[0]["delta"]["toolUse"]["name"] == "calculator"
    assert tool_events_multi[0]["delta"]["toolUse"]["input"] == {"expression": "2+2"}
    
    # Verify second tool call
    assert tool_events_multi[1]["delta"]["toolUse"]["toolUseId"] == "tool-456"
    assert tool_events_multi[1]["delta"]["toolUse"]["name"] == "weather"
    assert tool_events_multi[1]["delta"]["toolUse"]["input"] == {"location": "Seattle"}
    
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
    
    interrupt_events = model._convert_gemini_live_event(mock_interrupt)
    assert isinstance(interrupt_events, list)
    assert len(interrupt_events) == 1
    interrupt_event = interrupt_events[0]
    assert isinstance(interrupt_event, BidiInterruptionEvent)
    assert interrupt_event.get("type") == "bidi_interruption"
    assert interrupt_event.reason == "user_speech"
    
    await model.stop()


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
