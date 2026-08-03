"""Unit tests for bidirectional audio processing (echo cancellation, noise suppression, AGC)."""

import base64
import unittest.mock

import numpy as np
import pytest

from strands.experimental.bidi.io.audio import (
    AudioProcessingConfig,
    BidiAudioIO,
    _AudioProcessor,
)


def _fake_pywebrtc(processor=None):
    """Build a fake pywebrtc_audio module for patching into sys.modules."""
    processor = processor or unittest.mock.MagicMock()
    processor_class = unittest.mock.MagicMock(return_value=processor)
    module = unittest.mock.MagicMock(AudioProcessor=processor_class)
    return module, processor_class, processor


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


# ---------------------------------------------------------------------------
# AudioProcessingConfig validation
# ---------------------------------------------------------------------------


def test_config_defaults():
    config = AudioProcessingConfig()
    assert config.noise_suppression is True
    assert config.auto_gain_control is True
    assert config.ns_level == 1
    assert config.stream_delay_ms == 0


@pytest.mark.parametrize("bad_level", [-1, 4, 10])
def test_config_rejects_bad_ns_level(bad_level):
    with pytest.raises(ValueError, match="ns_level"):
        AudioProcessingConfig(ns_level=bad_level)


def test_config_rejects_negative_delay():
    with pytest.raises(ValueError, match="stream_delay_ms"):
        AudioProcessingConfig(stream_delay_ms=-5)


# ---------------------------------------------------------------------------
# BidiAudioIO construction and enablement
# ---------------------------------------------------------------------------


def test_no_processor_when_processing_disabled():
    audio_io = BidiAudioIO()
    assert audio_io._processor is None
    assert audio_io.input()._processor is None
    assert audio_io.output()._processor is None


def test_processor_shared_between_input_and_output():
    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig())

    assert audio_io._processor is not None
    assert audio_io.input()._processor is audio_io._processor
    assert audio_io.output()._processor is audio_io._processor


def test_construction_raises_import_error_without_pywebrtc():
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": None}):
        with pytest.raises(ImportError, match="pywebrtc-audio is required"):
            BidiAudioIO(audio_processing=AudioProcessingConfig())


def test_audio_processing_is_keyword_only():
    # Passing positionally must not be interpreted as audio_processing.
    with pytest.raises(TypeError):
        BidiAudioIO(AudioProcessingConfig())  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _AudioProcessor.start — sample rate gate and processor construction
# ---------------------------------------------------------------------------


def test_processor_start_builds_audio_processor_with_config():
    module, processor_class, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        proc = _AudioProcessor(AudioProcessingConfig(noise_suppression=False, ns_level=2, stream_delay_ms=20))
        proc.start(input_rate=16000, output_rate=16000, num_channels=1)

    processor_class.assert_called_once_with(
        sample_rate=16000,
        num_channels=1,
        echo_cancellation=True,
        noise_suppression=False,
        auto_gain_control=True,
        ns_level=2,
        stream_delay_ms=20,
    )


@pytest.mark.parametrize("rate", [16000, 32000, 48000])
def test_processor_start_accepts_supported_rates(rate):
    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        proc = _AudioProcessor(AudioProcessingConfig())
        proc.start(input_rate=rate, output_rate=rate, num_channels=1)


@pytest.mark.parametrize("rate", [8000, 24000, 44100])
def test_processor_start_rejects_unsupported_rates(rate):
    proc = _AudioProcessor(AudioProcessingConfig())
    with pytest.raises(ValueError, match="audio processing supports sample rates"):
        proc.start(input_rate=rate, output_rate=rate, num_channels=1)


# ---------------------------------------------------------------------------
# _AudioProcessor reference buffer behaviour
# ---------------------------------------------------------------------------


def test_process_capture_passes_reference_to_processor():
    frame = np.ones(160, dtype=np.int16) * 1000
    ref = np.ones(160, dtype=np.int16) * 500
    cleaned = np.ones(160, dtype=np.int16) * 200

    processor = unittest.mock.MagicMock()
    processor.process.return_value = cleaned
    module, _, _ = _fake_pywebrtc(processor)

    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        proc = _AudioProcessor(AudioProcessingConfig())
        proc.start(input_rate=16000, output_rate=16000, num_channels=1)

        proc.record_playback(ref.tobytes())
        result = proc.process_capture(frame.tobytes())

    near_arg, far_arg = processor.process.call_args[0]
    np.testing.assert_array_equal(near_arg, frame)
    np.testing.assert_array_equal(far_arg, ref)
    np.testing.assert_array_equal(np.frombuffer(result, dtype=np.int16), cleaned)


def test_process_capture_passes_silence_when_no_reference():
    # The WebRTC processor rejects a None far-end frame when echo cancellation is on, so an empty
    # reference buffer must be filled with zeros of the same length as the mic frame.
    frame = np.ones(160, dtype=np.int16) * 1000
    cleaned = np.zeros(160, dtype=np.int16)

    processor = unittest.mock.MagicMock()
    processor.process.return_value = cleaned
    module, _, _ = _fake_pywebrtc(processor)

    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        proc = _AudioProcessor(AudioProcessingConfig())
        proc.start(input_rate=16000, output_rate=16000, num_channels=1)
        proc.process_capture(frame.tobytes())

    near_arg, far_arg = processor.process.call_args[0]
    assert far_arg is not None
    assert far_arg.shape == near_arg.shape
    np.testing.assert_array_equal(far_arg, np.zeros(160, dtype=np.int16))


def test_reference_overflow_drops_oldest():
    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        proc = _AudioProcessor(AudioProcessingConfig(), max_ref_frames=2)
        proc.start(input_rate=16000, output_rate=16000, num_channels=1)

        proc.record_playback(b"\x01\x01")
        proc.record_playback(b"\x02\x02")
        proc.record_playback(b"\x03\x03")

        # Mic frame is 2 samples (4 bytes); oldest frame (\x01\x01) was dropped on overflow.
        ref = proc._take_reference(np.zeros(2, dtype=np.int16))

    np.testing.assert_array_equal(ref, np.frombuffer(b"\x02\x02\x03\x03", dtype=np.int16))


def test_reset_reference_drains_buffer_but_keeps_filter():
    processor = unittest.mock.MagicMock()
    module, _, _ = _fake_pywebrtc(processor)

    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        proc = _AudioProcessor(AudioProcessingConfig())
        proc.start(input_rate=16000, output_rate=16000, num_channels=1)
        proc.record_playback(b"\x01\x02\x03\x04")
        proc.reset_reference()

        # Buffer drained: reference for a 2-sample mic frame is silence.
        ref = proc._take_reference(np.zeros(2, dtype=np.int16))
        np.testing.assert_array_equal(ref, np.zeros(2, dtype=np.int16))

    # Interruption must NOT reset the converged AEC filter.
    processor.reset.assert_not_called()


def test_reset_drains_buffer_and_resets_filter():
    processor = unittest.mock.MagicMock()
    module, _, _ = _fake_pywebrtc(processor)

    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        proc = _AudioProcessor(AudioProcessingConfig())
        proc.start(input_rate=16000, output_rate=16000, num_channels=1)
        proc.record_playback(b"\x01\x02\x03\x04")
        proc.reset()

        ref = proc._take_reference(np.zeros(2, dtype=np.int16))
        np.testing.assert_array_equal(ref, np.zeros(2, dtype=np.int16))

    processor.reset.assert_called_once()


# ---------------------------------------------------------------------------
# Reference resampling (edge case: output rate != input rate)
# ---------------------------------------------------------------------------


def test_resample_same_rate_unchanged():
    proc = _AudioProcessor(AudioProcessingConfig())
    proc._input_rate = 16000
    proc._output_rate = 16000
    data = np.array([100, 200, 300, 400], dtype=np.int16).tobytes()
    assert proc._resample_reference(data) == data


def test_resample_downsample_length():
    proc = _AudioProcessor(AudioProcessingConfig())
    proc._input_rate = 16000
    proc._output_rate = 24000
    samples = np.arange(240, dtype=np.int16)
    result = np.frombuffer(proc._resample_reference(samples.tobytes()), dtype=np.int16)
    assert len(result) == round(240 * (16000 / 24000))


def test_resample_upsample_length():
    proc = _AudioProcessor(AudioProcessingConfig())
    proc._input_rate = 24000
    proc._output_rate = 16000
    samples = np.arange(160, dtype=np.int16)
    result = np.frombuffer(proc._resample_reference(samples.tobytes()), dtype=np.int16)
    assert len(result) == round(160 * (24000 / 16000))


# ---------------------------------------------------------------------------
# Stream wiring: buffer alignment and end-to-end callback flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_input_aligns_buffer_to_10ms_when_processing_on(py_audio, agent):
    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig(), input_frames_per_buffer=999)
        input_ = audio_io.input()
        await input_.start(agent)

    # 10ms at 16kHz = 160 samples, overriding the configured 999.
    assert py_audio.open.call_args.kwargs["frames_per_buffer"] == 160
    await input_.stop()


@pytest.mark.asyncio
async def test_input_keeps_configured_buffer_when_processing_off(py_audio, agent):
    audio_io = BidiAudioIO(input_frames_per_buffer=1024)
    input_ = audio_io.input()
    await input_.start(agent)

    assert py_audio.open.call_args.kwargs["frames_per_buffer"] == 1024
    await input_.stop()


@pytest.mark.asyncio
async def test_output_records_reference_at_playback(py_audio, agent):
    from strands.experimental.bidi.types.events import BidiAudioStreamEvent

    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig())
        output = audio_io.output()
        await output.start(agent)

        audio_data = b"\x10\x20\x30\x40"
        await output(
            BidiAudioStreamEvent(
                audio=base64.b64encode(audio_data).decode("utf-8"),
                channels=1,
                format="pcm",
                sample_rate=16000,
            )
        )

        # Reference is written only when audio actually exits the speaker (callback).
        mic = np.zeros(2, dtype=np.int16)
        np.testing.assert_array_equal(audio_io._processor._take_reference(mic), np.zeros(2, dtype=np.int16))

        played, _ = output._callback(None, frame_count=2)
        assert played == audio_data

        ref = audio_io._processor._take_reference(mic)
        np.testing.assert_array_equal(ref, np.frombuffer(audio_data, dtype=np.int16))

        await output.stop()


@pytest.mark.asyncio
async def test_output_clears_reference_on_interruption(py_audio, agent):
    from strands.experimental.bidi.types.events import BidiAudioStreamEvent, BidiInterruptionEvent

    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig())
        output = audio_io.output()
        await output.start(agent)

        audio_data = b"\x10\x20\x30\x40"
        await output(
            BidiAudioStreamEvent(
                audio=base64.b64encode(audio_data).decode("utf-8"),
                channels=1,
                format="pcm",
                sample_rate=16000,
            )
        )
        output._callback(None, frame_count=2)

        await output(BidiInterruptionEvent(reason="user_speech"))

        mic = np.zeros(2, dtype=np.int16)
        np.testing.assert_array_equal(audio_io._processor._take_reference(mic), np.zeros(2, dtype=np.int16))
        await output.stop()


@pytest.mark.asyncio
async def test_end_to_end_capture_cancels_echo(py_audio, agent):
    frame = np.ones(160, dtype=np.int16) * 1000
    cleaned = np.ones(160, dtype=np.int16) * 200

    processor = unittest.mock.MagicMock()
    processor.process.return_value = cleaned
    module, _, _ = _fake_pywebrtc(processor)

    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig())
        input_ = audio_io.input()
        await input_.start(agent)

        input_._buffer.put(frame.tobytes())
        event = await input_()

    result = np.frombuffer(base64.b64decode(event.audio), dtype=np.int16)
    np.testing.assert_array_equal(result, cleaned)
