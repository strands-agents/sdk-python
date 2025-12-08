"""Unit tests for Gemini Live bidirectional streaming model.

Tests the unified BidiGeminiLiveModel interface including:
- Model initialization and configuration
- Connection establishment and lifecycle
- Unified send() method with different content types
- Event receiving and conversion
"""

import base64
import unittest.mock

import pytest
from google.genai import types as genai_types

from strands.experimental.bidi.models.model import BidiModelTimeoutError
from strands.experimental.bidi.models.gemini_live import BidiGeminiLiveModel
from strands.experimental.bidi.types.events import (
    BidiAudioInputEvent,
    BidiAudioStreamEvent,
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
    with unittest.mock.patch("strands.experimental.bidi.models.gemini_live.genai.Client") as mock_client_cls:
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
    return BidiGeminiLiveModel(model_id=model_id, client_config={"api_key": api_key})


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
    assert model_default._live_session is None
    # Check default config includes transcription
    assert model_default.config["inference"]["response_modalities"] == ["AUDIO"]
    assert "outputAudioTranscription" in model_default.config["inference"]
    assert "inputAudioTranscription" in model_default.config["inference"]

    # Test with API key
    model_with_key = BidiGeminiLiveModel(model_id=model_id, client_config={"api_key": api_key})
    assert model_with_key.model_id == model_id
    assert model_with_key.api_key == api_key

    # Test with custom config (merges with defaults)
    provider_config = {"inference": {"temperature": 0.7, "top_p": 0.9}}
    model_custom = BidiGeminiLiveModel(model_id=model_id, provider_config=provider_config)
    # Custom config should be merged with defaults
    assert model_custom.config["inference"]["temperature"] == 0.7
    assert model_custom.config["inference"]["top_p"] == 0.9
    # Defaults should still be present
    assert "response_modalities" in model_custom.config["inference"]


# Connection Tests


@pytest.mark.asyncio
async def test_connection_lifecycle(mock_genai_client, model, system_prompt, tool_spec, messages):
    """Test complete connection lifecycle with various configurations."""
    mock_client, mock_live_session, mock_live_session_cm = mock_genai_client

    # Test basic connection
    await model.start()
    assert model._connection_id is not None
    assert model._live_session == mock_live_session
    mock_client.aio.live.connect.assert_called_once()

    # Test close
    await model.stop()
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
    model1 = BidiGeminiLiveModel(model_id=model_id, client_config={"api_key": api_key})
    mock_client.aio.live.connect.side_effect = Exception("Connection failed")
    with pytest.raises(Exception, match=r"Connection failed"):
        await model1.start()

    # Reset mock for next tests
    mock_client.aio.live.connect.side_effect = None

    # Test double connection
    model2 = BidiGeminiLiveModel(model_id=model_id, client_config={"api_key": api_key})
    await model2.start()
    with pytest.raises(RuntimeError, match="call stop before starting again"):
        await model2.start()
    await model2.stop()

    # Test close when not connected
    model3 = BidiGeminiLiveModel(model_id=model_id, client_config={"api_key": api_key})
    await model3.stop()  # Should not raise

    # Test close error handling
    model4 = BidiGeminiLiveModel(model_id=model_id, client_config={"api_key": api_key})
    await model4.start()
    mock_live_session_cm.__aexit__.side_effect = Exception("Close failed")
    with pytest.raises(ExceptionGroup):
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
    audio_b64 = base64.b64encode(b"audio_bytes").decode("utf-8")
    audio_input = BidiAudioInputEvent(
        audio=audio_b64,
        format="pcm",
        sample_rate=16000,
        channels=1,
    )
    await model.send(audio_input)
    mock_live_session.send_realtime_input.assert_called_once()

    # Test image input (base64 encoded, no encoding parameter)
    image_b64 = base64.b64encode(b"image_bytes").decode("utf-8")
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
    with pytest.raises(RuntimeError, match=r"call start before sending"):
        await model.send(text_input)
    mock_live_session.send_client_content.assert_not_called()

    # Test unknown content type
    await model.start()
    unknown_content = {"unknown_field": "value"}
    with pytest.raises(ValueError, match=r"content not supported"):
        await model.send(unknown_content)

    await model.stop()


# Receive Method Tests


@pytest.mark.asyncio
async def test_receive_lifecycle_events(mock_genai_client, model, agenerator):
    """Test that receive() emits connection start and end events."""
    _, mock_live_session, _ = mock_genai_client
    mock_live_session.receive.return_value = agenerator([])

    await model.start()

    async for event in model.receive():
        _ = event
        break

    # Verify connection start and end
    assert isinstance(event, BidiConnectionStartEvent)
    assert event.get("type") == "bidi_connection_start"
    assert event.connection_id == model._connection_id


@pytest.mark.asyncio
async def test_receive_timeout(mock_genai_client, model, agenerator):
    mock_resumption_response = unittest.mock.Mock()
    mock_resumption_response.go_away = None
    mock_resumption_response.session_resumption_update = unittest.mock.Mock()
    mock_resumption_response.session_resumption_update.resumable = True
    mock_resumption_response.session_resumption_update.new_handle = "h1"

    mock_timeout_response = unittest.mock.Mock()
    mock_timeout_response.go_away = unittest.mock.Mock()
    mock_timeout_response.go_away.model_dump_json.return_value = "test timeout"

    _, mock_live_session, _ = mock_genai_client
    mock_live_session.receive = unittest.mock.Mock(
        return_value=agenerator([mock_resumption_response, mock_timeout_response])
    )

    await model.start()

    with pytest.raises(BidiModelTimeoutError, match=r"test timeout"):
        async for _ in model.receive():
            pass

    tru_handle = model._live_session_handle
    exp_handle = "h1"
    assert tru_handle == exp_handle


@pytest.mark.asyncio
async def test_event_conversion(mock_genai_client, model):
    """Test conversion of all Gemini Live event types to standard format."""
    _, _, _ = mock_genai_client
    await model.start()

    # Test text output (converted to transcript via model_turn.parts)
    mock_text = unittest.mock.Mock()
    mock_text.data = None
    mock_text.go_away = None
    mock_text.session_resumption_update = None
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
    mock_multi_text.go_away = None
    mock_multi_text.session_resumption_update = None
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
    mock_audio.go_away = None
    mock_audio.session_resumption_update = None
    mock_audio.tool_call = None
    mock_audio.server_content = None

    audio_events = model._convert_gemini_live_event(mock_audio)
    assert isinstance(audio_events, list)
    assert len(audio_events) == 1
    audio_event = audio_events[0]
    assert isinstance(audio_event, BidiAudioStreamEvent)
    assert audio_event.get("type") == "bidi_audio_stream"
    # Audio is now base64 encoded
    expected_b64 = base64.b64encode(b"audio_data").decode("utf-8")
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
    mock_tool.go_away = None
    mock_tool.session_resumption_update = None
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
    mock_tool_multi.go_away = None
    mock_tool_multi.session_resumption_update = None
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
    mock_interrupt.go_away = None
    mock_interrupt.session_resumption_update = None
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


# Audio Configuration Tests


def test_audio_config_defaults(mock_genai_client, model_id, api_key):
    """Test default audio configuration."""
    _ = mock_genai_client

    model = BidiGeminiLiveModel(model_id=model_id, client_config={"api_key": api_key})

    assert model.config["audio"]["input_rate"] == 16000
    assert model.config["audio"]["output_rate"] == 24000
    assert model.config["audio"]["channels"] == 1
    assert model.config["audio"]["format"] == "pcm"
    assert "voice" not in model.config["audio"]  # No default voice


def test_audio_config_partial_override(mock_genai_client, model_id, api_key):
    """Test partial audio configuration override."""
    _ = mock_genai_client

    provider_config = {"audio": {"output_rate": 48000, "voice": "Puck"}}
    model = BidiGeminiLiveModel(model_id=model_id, client_config={"api_key": api_key}, provider_config=provider_config)

    # Overridden values
    assert model.config["audio"]["output_rate"] == 48000
    assert model.config["audio"]["voice"] == "Puck"

    # Default values preserved
    assert model.config["audio"]["input_rate"] == 16000
    assert model.config["audio"]["channels"] == 1
    assert model.config["audio"]["format"] == "pcm"


def test_audio_config_full_override(mock_genai_client, model_id, api_key):
    """Test full audio configuration override."""
    _ = mock_genai_client

    provider_config = {
        "audio": {
            "input_rate": 48000,
            "output_rate": 48000,
            "channels": 2,
            "format": "pcm",
            "voice": "Aoede",
        }
    }
    model = BidiGeminiLiveModel(model_id=model_id, client_config={"api_key": api_key}, provider_config=provider_config)

    assert model.config["audio"]["input_rate"] == 48000
    assert model.config["audio"]["output_rate"] == 48000
    assert model.config["audio"]["channels"] == 2
    assert model.config["audio"]["format"] == "pcm"
    assert model.config["audio"]["voice"] == "Aoede"


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



# Tool Result Content Tests


@pytest.mark.asyncio
async def test_custom_audio_rates_in_events(mock_genai_client, model_id, api_key):
    """Test that audio events use configured sample rates and channels."""
    _, _, _ = mock_genai_client

    # Create model with custom audio configuration
    provider_config = {"audio": {"output_rate": 48000, "channels": 2}}
    model = BidiGeminiLiveModel(model_id=model_id, client_config={"api_key": api_key}, provider_config=provider_config)
    await model.start()

    # Test audio output event uses custom configuration
    mock_audio = unittest.mock.Mock()
    mock_audio.text = None
    mock_audio.data = b"audio_data"
    mock_audio.go_away = None
    mock_audio.session_resumption_update = None
    mock_audio.tool_call = None
    mock_audio.server_content = None

    audio_events = model._convert_gemini_live_event(mock_audio)
    assert len(audio_events) == 1
    audio_event = audio_events[0]
    assert isinstance(audio_event, BidiAudioStreamEvent)
    # Should use configured rates, not constants
    assert audio_event.sample_rate == 48000  # Custom config
    assert audio_event.channels == 2         # Custom config
    assert audio_event.format == "pcm"

    await model.stop()


@pytest.mark.asyncio
async def test_default_audio_rates_in_events(mock_genai_client, model_id, api_key):
    """Test that audio events use default sample rates when no custom config."""
    _, _, _ = mock_genai_client

    # Create model without custom audio configuration
    model = BidiGeminiLiveModel(model_id=model_id, client_config={"api_key": api_key})
    await model.start()

    # Test audio output event uses defaults
    mock_audio = unittest.mock.Mock()
    mock_audio.text = None
    mock_audio.data = b"audio_data"
    mock_audio.go_away = None
    mock_audio.session_resumption_update = None
    mock_audio.tool_call = None
    mock_audio.server_content = None

    audio_events = model._convert_gemini_live_event(mock_audio)
    assert len(audio_events) == 1
    audio_event = audio_events[0]
    assert isinstance(audio_event, BidiAudioStreamEvent)
    # Should use default rates
    assert audio_event.sample_rate == 24000  # Default output rate
    assert audio_event.channels == 1         # Default channels
    assert audio_event.format == "pcm"

    await model.stop()


# Tool Result Content Tests


@pytest.mark.asyncio
async def test_tool_result_single_content_unwrapped(mock_genai_client, model):
    """Test that single content item is unwrapped (optimization)."""
    _, mock_live_session, _ = mock_genai_client
    await model.start()

    tool_result: ToolResult = {
        "toolUseId": "tool-123",
        "status": "success",
        "content": [{"text": "Single result"}],
    }

    await model.send(ToolResultEvent(tool_result))

    # Verify the tool response was sent
    mock_live_session.send_tool_response.assert_called_once()
    call_args = mock_live_session.send_tool_response.call_args
    function_responses = call_args.kwargs.get("function_responses", [])

    assert len(function_responses) == 1
    func_response = function_responses[0]
    assert func_response.id == "tool-123"
    # Single content should be unwrapped (not in array)
    assert func_response.response == {"text": "Single result"}

    await model.stop()


@pytest.mark.asyncio
async def test_tool_result_multiple_content_as_array(mock_genai_client, model):
    """Test that multiple content items are sent as array."""
    _, mock_live_session, _ = mock_genai_client
    await model.start()

    tool_result: ToolResult = {
        "toolUseId": "tool-456",
        "status": "success",
        "content": [{"text": "Part 1"}, {"json": {"data": "value"}}],
    }

    await model.send(ToolResultEvent(tool_result))

    # Verify the tool response was sent
    mock_live_session.send_tool_response.assert_called_once()
    call_args = mock_live_session.send_tool_response.call_args
    function_responses = call_args.kwargs.get("function_responses", [])

    assert len(function_responses) == 1
    func_response = function_responses[0]
    assert func_response.id == "tool-456"
    # Multiple content should be in array format
    assert "result" in func_response.response
    assert isinstance(func_response.response["result"], list)
    assert len(func_response.response["result"]) == 2
    assert func_response.response["result"][0] == {"text": "Part 1"}
    assert func_response.response["result"][1] == {"json": {"data": "value"}}

    await model.stop()


@pytest.mark.asyncio
async def test_tool_result_unsupported_content_type(mock_genai_client, model):
    """Test that unsupported content types raise ValueError."""
    _, _, _ = mock_genai_client
    await model.start()

    # Test with image content (unsupported)
    tool_result_image: ToolResult = {
        "toolUseId": "tool-999",
        "status": "success",
        "content": [{"image": {"format": "jpeg", "source": {"bytes": b"image_data"}}}],
    }

    with pytest.raises(ValueError, match=r"Content type not supported by Gemini Live API"):
        await model.send(ToolResultEvent(tool_result_image))

    # Test with document content (unsupported)
    tool_result_doc: ToolResult = {
        "toolUseId": "tool-888",
        "status": "success",
        "content": [{"document": {"format": "pdf", "source": {"bytes": b"doc_data"}}}],
    }

    with pytest.raises(ValueError, match=r"Content type not supported by Gemini Live API"):
        await model.send(ToolResultEvent(tool_result_doc))

    # Test with mixed content (one unsupported)
    tool_result_mixed: ToolResult = {
        "toolUseId": "tool-777",
        "status": "success",
        "content": [{"text": "Valid text"}, {"image": {"format": "jpeg", "source": {"bytes": b"image_data"}}}],
    }

    with pytest.raises(ValueError, match=r"Content type not supported by Gemini Live API"):
        await model.send(ToolResultEvent(tool_result_mixed))

    await model.stop()


# Helper fixture for async generator
@pytest.fixture
def agenerator():
    """Helper to create async generators for testing."""

    async def _agenerator(items):
        for item in items:
            yield item

    return _agenerator
