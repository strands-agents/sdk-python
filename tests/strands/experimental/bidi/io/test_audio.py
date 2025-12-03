import base64
import unittest.mock

import pyaudio
import pytest
import pytest_asyncio

from strands.experimental.bidi.io.audio import BidiAudioIO, _BidiAudioBuffer
from strands.experimental.bidi.types.events import BidiAudioInputEvent, BidiAudioStreamEvent, BidiInterruptionEvent


@pytest.fixture
def audio_buffer():
    buffer = _BidiAudioBuffer(size=1)
    buffer.start()
    yield buffer
    buffer.stop()


@pytest.fixture
def agent():
    mock = unittest.mock.MagicMock()
    mock.model.config = {
        "audio": {
            "input_rate": 24000,
            "output_rate": 16000,
            "channels": 2,
            "format": "test-format",
            "voice": "test-voice",
        },
    }
    return mock 


@pytest.fixture
def py_audio():
    with unittest.mock.patch("strands.experimental.bidi.io.audio.pyaudio.PyAudio") as mock:
        yield mock.return_value


@pytest.fixture
def config():
    return {
        "input_buffer_size": 1,
        "input_device_index": 1,
        "input_frames_per_buffer": 1024,
        "output_buffer_size": 2,
        "output_device_index": 2,
        "output_frames_per_buffer": 2048,
    }

@pytest.fixture
def audio_io(py_audio, config):
    _ = py_audio
    return BidiAudioIO(**config)


@pytest_asyncio.fixture
async def audio_input(audio_io, agent):
    input_ = audio_io.input()
    await input_.start(agent)
    yield input_
    await input_.stop()


@pytest_asyncio.fixture
async def audio_output(audio_io, agent):
    output = audio_io.output()
    await output.start(agent)
    yield output
    await output.stop()


def test_bidi_audio_buffer_put(audio_buffer):
    audio_buffer.put(b"test-chunk")

    tru_chunk = audio_buffer.get()
    exp_chunk = b"test-chunk"
    assert tru_chunk == exp_chunk


def test_bidi_audio_buffer_put_full(audio_buffer):
    audio_buffer.put(b"test-chunk-1")
    audio_buffer.put(b"test-chunk-2")

    tru_chunk = audio_buffer.get()
    exp_chunk = b"test-chunk-2"
    assert tru_chunk == exp_chunk


def test_bidi_audio_buffer_get_padding(audio_buffer):
    audio_buffer.put(b"test-chunk")

    tru_chunk = audio_buffer.get(11)
    exp_chunk = b"test-chunk\x00"
    assert tru_chunk == exp_chunk


def test_bidi_audio_buffer_clear(audio_buffer):
    audio_buffer.put(b"test-chunk")
    audio_buffer.clear()

    tru_byte = audio_buffer.get(1)
    exp_byte = b"\x00"
    assert tru_byte == exp_byte


@pytest.mark.asyncio
async def test_bidi_audio_io_input(audio_input):
    audio_input._callback(b"test-audio")

    tru_event = await audio_input()
    exp_event = BidiAudioInputEvent(
        audio=base64.b64encode(b"test-audio").decode("utf-8"),
        channels=2,
        format="test-format",
        sample_rate=24000,
    )
    assert tru_event == exp_event


def test_bidi_audio_io_input_configs(py_audio, audio_input):
    py_audio.open.assert_called_once_with(
        channels=2,
        format=pyaudio.paInt16,
        frames_per_buffer=1024,
        input=True,
        input_device_index=1,
        rate=24000,
        stream_callback=audio_input._callback,
    )


@pytest.mark.asyncio
async def test_bidi_audio_io_output(audio_output):
    audio_event = BidiAudioStreamEvent(
        audio=base64.b64encode(b"test-audio").decode("utf-8"),
        channels=2,
        format="test-format",
        sample_rate=16000,
    )
    await audio_output(audio_event)

    tru_data, _ = audio_output._callback(None, frame_count=4)
    exp_data = b"test-aud"
    assert tru_data == exp_data


@pytest.mark.asyncio
async def test_bidi_audio_io_output_interrupt(audio_output):
    audio_event = BidiAudioStreamEvent(
        audio=base64.b64encode(b"test-audio").decode("utf-8"),
        channels=2,
        format="test-format",
        sample_rate=16000,
    )
    await audio_output(audio_event)
    interrupt_event = BidiInterruptionEvent(reason="user_speech")
    await audio_output(interrupt_event)

    tru_data, _ = audio_output._callback(None, frame_count=1)
    exp_data = b"\x00\x00"
    assert tru_data == exp_data


def test_bidi_audio_io_output_configs(py_audio, audio_output):
    py_audio.open.assert_called_once_with(
        channels=2,
        format=pyaudio.paInt16,
        frames_per_buffer=2048,
        output=True,
        output_device_index=2,
        rate=16000,
        stream_callback=audio_output._callback,
    )
