import asyncio
import base64
import unittest.mock

import pytest

from strands.experimental.bidi.io import BidiAudioIO
from strands.experimental.bidi.types.events import BidiAudioInputEvent, BidiAudioStreamEvent


@pytest.fixture
def py_audio():
    with unittest.mock.patch("strands.experimental.bidi.io.audio.pyaudio") as mock:
        yield mock.PyAudio()


@pytest.fixture
def audio_io():
    return BidiAudioIO()


@pytest.fixture
def mock_agent():
    """Create a mock agent with model that has default audio_config."""
    agent = unittest.mock.MagicMock()
    agent.model.audio_config = {
        "input_rate": 16000,
        "output_rate": 16000,
        "channels": 1,
        "format": "pcm",
        "voice": "matthew",
    }
    return agent


@pytest.fixture
def mock_agent_custom_config():
    """Create a mock agent with custom audio_config."""
    agent = unittest.mock.MagicMock()
    agent.model.audio_config = {
        "input_rate": 48000,
        "output_rate": 24000,
        "channels": 2,
        "format": "pcm",
        "voice": "alloy",
    }
    return agent


@pytest.fixture
def audio_input(audio_io):
    return audio_io.input()


@pytest.fixture
def audio_output(audio_io):
    return audio_io.output()


@pytest.mark.asyncio
async def test_bidi_audio_io_input(py_audio, audio_input, mock_agent):
    """Test basic audio input functionality."""
    microphone = unittest.mock.Mock()
    microphone.read.return_value = b"test-audio"

    py_audio.open.return_value = microphone

    await audio_input.start(mock_agent)
    tru_event = await audio_input()
    await audio_input.stop()

    exp_event = BidiAudioInputEvent(
        audio=base64.b64encode(b"test-audio").decode("utf-8"),
        channels=1,
        format="pcm",
        sample_rate=16000,
    )
    assert tru_event == exp_event

    microphone.read.assert_called_once_with(512, exception_on_overflow=False)


@pytest.mark.asyncio
async def test_bidi_audio_io_output(py_audio, audio_output, mock_agent):
    """Test basic audio output functionality."""
    write_future = asyncio.Future()
    write_event = asyncio.Event()

    def write(data):
        write_future.set_result(data)
        write_event.set()

    speaker = unittest.mock.Mock()
    speaker.write.side_effect = write

    py_audio.open.return_value = speaker

    await audio_output.start(mock_agent)

    audio_event = BidiAudioStreamEvent(
        audio=base64.b64encode(b"test-audio").decode("utf-8"),
        channels=1,
        format="pcm",
        sample_rate=1600,
    )
    await audio_output(audio_event)
    await write_event.wait()

    await audio_output.stop()

    speaker.write.assert_called_once_with(write_future.result())


# Audio Configuration Tests


@pytest.mark.asyncio
async def test_audio_input_uses_model_config(py_audio, audio_io, mock_agent):
    """Test that audio input uses model's audio_config."""
    audio_input = audio_io.input()

    microphone = unittest.mock.Mock()
    microphone.read.return_value = b"test-audio"
    py_audio.open.return_value = microphone

    await audio_input.start(mock_agent)

    # Model config should be used
    py_audio.open.assert_called_once()
    call_kwargs = py_audio.open.call_args.kwargs
    assert call_kwargs["rate"] == 16000  # From mock_agent.model.audio_config
    assert call_kwargs["channels"] == 1  # From mock_agent.model.audio_config

    await audio_input.stop()


@pytest.mark.asyncio
async def test_audio_input_uses_custom_model_config(py_audio, audio_io, mock_agent_custom_config):
    """Test that audio input uses custom model audio_config."""
    audio_input = audio_io.input()

    microphone = unittest.mock.Mock()
    microphone.read.return_value = b"test-audio"
    py_audio.open.return_value = microphone

    await audio_input.start(mock_agent_custom_config)

    # Custom model config should be used
    py_audio.open.assert_called_once()
    call_kwargs = py_audio.open.call_args.kwargs
    assert call_kwargs["rate"] == 48000  # From custom config
    assert call_kwargs["channels"] == 2  # From custom config

    await audio_input.stop()


@pytest.mark.asyncio
async def test_audio_output_uses_model_config(py_audio, audio_io, mock_agent):
    """Test that audio output uses model's audio_config."""
    audio_output = audio_io.output()

    speaker = unittest.mock.Mock()
    py_audio.open.return_value = speaker

    await audio_output.start(mock_agent)

    # Model config should be used
    py_audio.open.assert_called_once()
    call_kwargs = py_audio.open.call_args.kwargs
    assert call_kwargs["rate"] == 16000  # From mock_agent.model.audio_config
    assert call_kwargs["channels"] == 1  # From mock_agent.model.audio_config

    await audio_output.stop()


@pytest.mark.asyncio
async def test_audio_output_uses_custom_model_config(py_audio, audio_io, mock_agent_custom_config):
    """Test that audio output uses custom model audio_config."""
    audio_output = audio_io.output()

    speaker = unittest.mock.Mock()
    py_audio.open.return_value = speaker

    await audio_output.start(mock_agent_custom_config)

    # Custom model config should be used
    py_audio.open.assert_called_once()
    call_kwargs = py_audio.open.call_args.kwargs
    assert call_kwargs["rate"] == 24000  # From custom config
    assert call_kwargs["channels"] == 2  # From custom config

    await audio_output.stop()


# Device Configuration Tests


@pytest.mark.asyncio
async def test_audio_input_respects_user_device_config(py_audio, mock_agent):
    """Test that user-provided device config overrides defaults."""
    audio_io = BidiAudioIO(input_device_index=5, input_frames_per_buffer=1024)
    audio_input = audio_io.input()

    microphone = unittest.mock.Mock()
    microphone.read.return_value = b"test-audio"
    py_audio.open.return_value = microphone

    await audio_input.start(mock_agent)

    # User device config should be used
    py_audio.open.assert_called_once()
    call_kwargs = py_audio.open.call_args.kwargs
    assert call_kwargs["input_device_index"] == 5  # User config
    assert call_kwargs["frames_per_buffer"] == 1024  # User config
    # Model config still used for audio parameters
    assert call_kwargs["rate"] == 16000  # From model
    assert call_kwargs["channels"] == 1  # From model

    await audio_input.stop()


@pytest.mark.asyncio
async def test_audio_output_respects_user_device_config(py_audio, mock_agent):
    """Test that user-provided device config overrides defaults."""
    audio_io = BidiAudioIO(output_device_index=3, output_frames_per_buffer=2048, output_buffer_size=50)
    audio_output = audio_io.output()

    speaker = unittest.mock.Mock()
    py_audio.open.return_value = speaker

    await audio_output.start(mock_agent)

    # User device config should be used
    py_audio.open.assert_called_once()
    call_kwargs = py_audio.open.call_args.kwargs
    assert call_kwargs["output_device_index"] == 3  # User config
    assert call_kwargs["frames_per_buffer"] == 2048  # User config
    # Model config still used for audio parameters
    assert call_kwargs["rate"] == 16000  # From model
    assert call_kwargs["channels"] == 1  # From model
    # Buffer size should be set
    assert audio_output._buffer_size == 50  # User config

    await audio_output.stop()


@pytest.mark.asyncio
async def test_audio_io_uses_defaults_when_no_config(py_audio, mock_agent):
    """Test that defaults are used when no config provided."""
    audio_io = BidiAudioIO()  # No config
    audio_input = audio_io.input()

    microphone = unittest.mock.Mock()
    microphone.read.return_value = b"test-audio"
    py_audio.open.return_value = microphone

    await audio_input.start(mock_agent)

    # Defaults should be used
    py_audio.open.assert_called_once()
    call_kwargs = py_audio.open.call_args.kwargs
    assert call_kwargs["input_device_index"] is None  # Default
    assert call_kwargs["frames_per_buffer"] == 512  # Default

    await audio_input.stop()
