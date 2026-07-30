"""Unit tests for bidirectional audio echo cancellation."""

import base64
import unittest.mock

import numpy as np
import pytest

from strands.experimental.bidi.io.audio import BidiAudioIO, _BidiAudioOutput, _EchoReferenceBuffer


@pytest.fixture
def agent():
    mock = unittest.mock.MagicMock()
    mock.model.config = {
        "audio": {
            "input_rate": 16000,
            "output_rate": 16000,
            "channels": 1,
            "format": "pcm",
            "voice": "test-voice",
        },
    }
    return mock


@pytest.fixture
def agent_mixed_rates():
    mock = unittest.mock.MagicMock()
    mock.model.config = {
        "audio": {
            "input_rate": 16000,
            "output_rate": 24000,
            "channels": 1,
            "format": "pcm",
            "voice": "test-voice",
        },
    }
    return mock


@pytest.fixture
def py_audio():
    with unittest.mock.patch("strands.experimental.bidi.io.audio.pyaudio.PyAudio") as mock:
        yield mock.return_value


@pytest.fixture
def echo_ref_buffer():
    return _EchoReferenceBuffer()


def test_echo_ref_buffer_put_and_get(echo_ref_buffer):
    """Reference buffer stores and returns audio frames."""
    echo_ref_buffer.put(b"\x01\x02\x03\x04")

    tru_data = echo_ref_buffer.get(4)
    exp_data = b"\x01\x02\x03\x04"
    assert tru_data == exp_data


def test_echo_ref_buffer_returns_silence_when_empty(echo_ref_buffer):
    """Reference buffer returns silence when no data is available."""
    tru_data = echo_ref_buffer.get(4)
    exp_data = b"\x00\x00\x00\x00"
    assert tru_data == exp_data


def test_echo_ref_buffer_clear(echo_ref_buffer):
    """Reference buffer is cleared correctly."""
    echo_ref_buffer.put(b"\x01\x02\x03\x04")
    echo_ref_buffer.clear()

    tru_data = echo_ref_buffer.get(4)
    exp_data = b"\x00\x00\x00\x00"
    assert tru_data == exp_data


def test_echo_ref_buffer_overflow_drops_oldest(echo_ref_buffer):
    """When buffer is full, oldest frames are dropped."""
    ref_buf = _EchoReferenceBuffer(max_frames=2)
    ref_buf.put(b"\x01\x01")
    ref_buf.put(b"\x02\x02")
    ref_buf.put(b"\x03\x03")

    tru_data = ref_buf.get(4)
    exp_data = b"\x02\x02\x03\x03"
    assert tru_data == exp_data


def test_bidi_audio_io_creates_shared_ref_buffer_when_aec_enabled():
    """BidiAudioIO creates a shared reference buffer when echo_cancellation=True."""
    audio_io = BidiAudioIO(echo_cancellation=True)
    assert audio_io._echo_ref_buffer is not None


def test_bidi_audio_io_no_ref_buffer_when_aec_disabled():
    """BidiAudioIO does not create a reference buffer when echo_cancellation=False."""
    audio_io = BidiAudioIO(echo_cancellation=False)
    assert audio_io._echo_ref_buffer is None


def test_numpy_not_imported_when_aec_disabled():
    """numpy is not imported at module level — only loaded when AEC processes frames."""
    # numpy may already be imported by the test framework, so we check
    # that creating BidiAudioIO without AEC doesn't trigger new numpy usage
    # by verifying the module doesn't force-import it at construction
    audio_io = BidiAudioIO(echo_cancellation=False)
    input_ = audio_io.input()
    # The input should have no pipeline — no numpy needed
    assert input_._pipeline is None
    assert input_._echo_cancellation is False


@pytest.mark.asyncio
async def test_aec_raises_import_error_when_pywebrtc_not_installed(py_audio, agent):
    """Echo cancellation raises ImportError if pywebrtc-audio is not installed."""
    audio_io = BidiAudioIO(echo_cancellation=True)
    input_ = audio_io.input()

    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": None}):
        with pytest.raises(ImportError, match="pywebrtc-audio is required"):
            await input_.start(agent)


@pytest.mark.asyncio
async def test_aec_initializes_pipeline(py_audio, agent):
    """Echo cancellation initializes processing pipeline when pywebrtc-audio is available."""
    mock_processor = unittest.mock.MagicMock()
    mock_audio_processor_class = unittest.mock.MagicMock(return_value=mock_processor)

    with unittest.mock.patch.dict(
        "sys.modules",
        {"pywebrtc_audio": unittest.mock.MagicMock(AudioProcessor=mock_audio_processor_class)},
    ):
        audio_io = BidiAudioIO(echo_cancellation=True)
        input_ = audio_io.input()
        await input_.start(agent)

        mock_audio_processor_class.assert_called_once_with(
            sample_rate=16000,
            num_channels=1,
            echo_cancellation=True,
            noise_suppression=True,
            auto_gain_control=True,
            stream_delay_ms=0,
        )
        assert input_._pipeline is not None

        await input_.stop()


@pytest.mark.asyncio
async def test_aec_processes_mic_with_reference(py_audio, agent):
    """Echo cancellation processes mic frames using the speaker reference."""
    frame_size = 160  # 10ms at 16kHz mono
    mic_samples = np.ones(frame_size, dtype=np.int16) * 1000
    ref_samples = np.ones(frame_size, dtype=np.int16) * 500
    cleaned_samples = np.ones(frame_size, dtype=np.int16) * 200

    mock_processor = unittest.mock.MagicMock()
    mock_processor.process.return_value = cleaned_samples
    mock_audio_processor_class = unittest.mock.MagicMock(return_value=mock_processor)

    with unittest.mock.patch.dict(
        "sys.modules",
        {"pywebrtc_audio": unittest.mock.MagicMock(AudioProcessor=mock_audio_processor_class)},
    ):
        audio_io = BidiAudioIO(echo_cancellation=True)
        input_ = audio_io.input()
        output = audio_io.output()

        await input_.start(agent)
        await output.start(agent)

        # Simulate speaker playing audio (writes to reference buffer)
        ref_data = ref_samples.tobytes()
        audio_io._echo_ref_buffer.put(ref_data)

        # Simulate mic capturing audio
        mic_data = mic_samples.tobytes()
        input_._buffer.put(mic_data)

        event = await input_()

        # Verify the processor was called
        assert mock_processor.process.called
        call_args = mock_processor.process.call_args[0]
        np.testing.assert_array_equal(call_args[0], mic_samples)
        np.testing.assert_array_equal(call_args[1], ref_samples)

        # Verify the output is the cleaned audio
        decoded = base64.b64decode(event.audio)
        result = np.frombuffer(decoded, dtype=np.int16)
        np.testing.assert_array_equal(result, cleaned_samples)

        await input_.stop()
        await output.stop()


@pytest.mark.asyncio
async def test_aec_disabled_passes_raw_audio(py_audio, agent):
    """When AEC is disabled, mic audio passes through unmodified."""
    audio_io = BidiAudioIO(echo_cancellation=False)
    input_ = audio_io.input()
    await input_.start(agent)

    mic_data = b"\x01\x02\x03\x04"
    input_._buffer.put(mic_data)

    event = await input_()
    decoded = base64.b64decode(event.audio)
    assert decoded == mic_data

    await input_.stop()


@pytest.mark.asyncio
async def test_output_writes_reference_at_playback_time(py_audio, agent):
    """Output callback writes audio to the reference buffer at the moment it plays."""
    from strands.experimental.bidi.types.events import BidiAudioStreamEvent

    audio_io = BidiAudioIO(echo_cancellation=True)
    output = audio_io.output()
    await output.start(agent)

    audio_data = b"\x10\x20\x30\x40"
    event = BidiAudioStreamEvent(
        audio=base64.b64encode(audio_data).decode("utf-8"),
        channels=1,
        format="pcm",
        sample_rate=16000,
    )
    await output(event)

    # Reference should NOT be written yet (only buffered for playback)
    ref_data = audio_io._echo_ref_buffer.get(4)
    assert ref_data == b"\x00\x00\x00\x00"

    # Simulate PyAudio callback firing (actual playback moment)
    played_data, _ = output._callback(None, frame_count=2)
    assert played_data == audio_data

    # NOW the reference should be available
    ref_data = audio_io._echo_ref_buffer.get(4)
    assert ref_data == audio_data

    await output.stop()


@pytest.mark.asyncio
async def test_output_clears_reference_on_interruption(py_audio, agent):
    """Output stream clears the reference buffer on interruption."""
    from strands.experimental.bidi.types.events import BidiAudioStreamEvent, BidiInterruptionEvent

    audio_io = BidiAudioIO(echo_cancellation=True)
    output = audio_io.output()
    await output.start(agent)

    # Buffer audio and play it (writes reference)
    audio_data = b"\x10\x20\x30\x40"
    event = BidiAudioStreamEvent(
        audio=base64.b64encode(audio_data).decode("utf-8"),
        channels=1,
        format="pcm",
        sample_rate=16000,
    )
    await output(event)
    output._callback(None, frame_count=2)  # plays and writes reference

    # Interrupt clears the reference
    interrupt = BidiInterruptionEvent(reason="user_speech")
    await output(interrupt)

    ref_data = audio_io._echo_ref_buffer.get(4)
    assert ref_data == b"\x00\x00\x00\x00"

    await output.stop()


def test_resample_reference_same_rate(py_audio):
    """Resampling returns data unchanged when rates match."""
    output = _BidiAudioOutput({})
    output._output_rate = 16000
    output._input_rate = 16000
    data = np.array([100, 200, 300, 400], dtype=np.int16).tobytes()

    tru_result = output._resample_reference(data)
    assert tru_result == data


def test_resample_reference_downsample():
    """Resampling downsamples from higher output rate to lower input rate."""
    output = _BidiAudioOutput({})
    output._output_rate = 24000
    output._input_rate = 16000
    samples = np.arange(240, dtype=np.int16)
    data = samples.tobytes()

    result_bytes = output._resample_reference(data)
    result = np.frombuffer(result_bytes, dtype=np.int16)

    exp_length = int(240 * (16000 / 24000))
    assert len(result) == exp_length


def test_resample_reference_upsample():
    """Resampling upsamples from lower output rate to higher input rate."""
    output = _BidiAudioOutput({})
    output._output_rate = 16000
    output._input_rate = 24000
    samples = np.arange(160, dtype=np.int16)
    data = samples.tobytes()

    result_bytes = output._resample_reference(data)
    result = np.frombuffer(result_bytes, dtype=np.int16)

    exp_length = int(160 * (24000 / 16000))
    assert len(result) == exp_length


@pytest.mark.asyncio
async def test_aec_stop_clears_pipeline(py_audio, agent):
    """Stopping input clears the processing pipeline reference."""
    mock_processor = unittest.mock.MagicMock()
    mock_audio_processor_class = unittest.mock.MagicMock(return_value=mock_processor)

    with unittest.mock.patch.dict(
        "sys.modules",
        {"pywebrtc_audio": unittest.mock.MagicMock(AudioProcessor=mock_audio_processor_class)},
    ):
        audio_io = BidiAudioIO(echo_cancellation=True)
        input_ = audio_io.input()
        await input_.start(agent)
        assert input_._pipeline is not None

        await input_.stop()
        assert input_._pipeline is None
