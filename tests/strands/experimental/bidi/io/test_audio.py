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
def audio_input(audio_io):
    return audio_io.input()


@pytest.fixture
def audio_output(audio_io):
    return audio_io.output()


@pytest.mark.asyncio
async def test_bidi_audio_io_input(py_audio, audio_input):
    microphone = unittest.mock.Mock()
    microphone.read.return_value = b"test-audio"

    py_audio.open.return_value = microphone

    await audio_input.start()
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
async def test_bidi_audio_io_output(py_audio, audio_output):
    write_future = asyncio.Future()
    write_event = asyncio.Event()

    def write(data):
        write_future.set_result(data)
        write_event.set()

    speaker = unittest.mock.Mock()
    speaker.write.side_effect = write

    py_audio.open.return_value = speaker

    await audio_output.start()

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
async def test_audio_input_respects_user_config(py_audio):
    """Test that user-provided config takes precedence over model config."""
    audio_io = BidiAudioIO(input_rate=48000, input_channels=2)
    audio_input = audio_io.input()

    microphone = unittest.mock.Mock()
    microphone.read.return_value = b"test-audio"
    py_audio.open.return_value = microphone

    # Model provides different config
    model_audio_config = {"input_rate": 16000, "channels": 1}

    await audio_input.start(audio_config=model_audio_config)

    # User config should be used
    py_audio.open.assert_called_once()
    call_kwargs = py_audio.open.call_args.kwargs
    assert call_kwargs["rate"] == 48000  # User config
    assert call_kwargs["channels"] == 2  # User config

    await audio_input.stop()


@pytest.mark.asyncio
async def test_audio_input_applies_model_config_when_user_not_set(py_audio):
    """Test that model config is applied when user doesn't provide values."""
    audio_io = BidiAudioIO()  # No user config
    audio_input = audio_io.input()

    microphone = unittest.mock.Mock()
    microphone.read.return_value = b"test-audio"
    py_audio.open.return_value = microphone

    # Model provides config
    model_audio_config = {"input_rate": 24000, "channels": 2}

    await audio_input.start(audio_config=model_audio_config)

    # Model config should be used
    py_audio.open.assert_called_once()
    call_kwargs = py_audio.open.call_args.kwargs
    assert call_kwargs["rate"] == 24000  # Model config
    assert call_kwargs["channels"] == 2  # Model config

    await audio_input.stop()


@pytest.mark.asyncio
async def test_audio_output_respects_user_config(py_audio):
    """Test that user-provided config takes precedence over model config."""
    audio_io = BidiAudioIO(output_rate=48000, output_channels=2)
    audio_output = audio_io.output()

    speaker = unittest.mock.Mock()
    py_audio.open.return_value = speaker

    # Model provides different config
    model_audio_config = {"output_rate": 16000, "channels": 1}

    await audio_output.start(audio_config=model_audio_config)

    # User config should be used
    py_audio.open.assert_called_once()
    call_kwargs = py_audio.open.call_args.kwargs
    assert call_kwargs["rate"] == 48000  # User config
    assert call_kwargs["channels"] == 2  # User config

    await audio_output.stop()


@pytest.mark.asyncio
async def test_audio_output_applies_model_config_when_user_not_set(py_audio):
    """Test that model config is applied when user doesn't provide values."""
    audio_io = BidiAudioIO()  # No user config
    audio_output = audio_io.output()

    speaker = unittest.mock.Mock()
    py_audio.open.return_value = speaker

    # Model provides config
    model_audio_config = {"output_rate": 24000, "channels": 2}

    await audio_output.start(audio_config=model_audio_config)

    # Model config should be used
    py_audio.open.assert_called_once()
    call_kwargs = py_audio.open.call_args.kwargs
    assert call_kwargs["rate"] == 24000  # Model config
    assert call_kwargs["channels"] == 2  # Model config

    await audio_output.stop()


@pytest.mark.asyncio
async def test_audio_partial_user_config(py_audio):
    """Test that partial user config works correctly."""
    # User only sets rate, not channels
    audio_io = BidiAudioIO(input_rate=48000)
    audio_input = audio_io.input()

    microphone = unittest.mock.Mock()
    microphone.read.return_value = b"test-audio"
    py_audio.open.return_value = microphone

    # Model provides both rate and channels
    model_audio_config = {"input_rate": 16000, "channels": 2}

    await audio_input.start(audio_config=model_audio_config)

    # User rate should be used, model channels should be applied
    py_audio.open.assert_called_once()
    call_kwargs = py_audio.open.call_args.kwargs
    assert call_kwargs["rate"] == 48000  # User config
    assert call_kwargs["channels"] == 2  # Model config

    await audio_input.stop()
