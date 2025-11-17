import asyncio
import base64
import unittest.mock

import pytest

from strands.experimental.bidi.io import BidiAudioIO
from strands.experimental.bidi.types.events import BidiAudioInputEvent, BidiAudioStreamEvent, BidiInterruptionEvent


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
