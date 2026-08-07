import base64
import unittest.mock

import numpy as np
import pyaudio
import pytest
import pytest_asyncio

from strands.experimental.bidi.io.audio import AudioProcessingConfig, BidiAudioIO, _AudioProcessor, _BidiAudioBuffer
from strands.experimental.bidi.types.events import BidiAudioInputEvent, BidiAudioStreamEvent, BidiInterruptionEvent


def _fake_pywebrtc(processor=None):
    """Build a fake pywebrtc_audio module for patching into sys.modules."""
    processor = processor or unittest.mock.MagicMock()
    processor_class = unittest.mock.MagicMock(return_value=processor)
    module = unittest.mock.MagicMock(AudioProcessor=processor_class)
    return module, processor_class, processor


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
def aec_agent():
    # Audio processing requires a supported input rate (16k/32k/48k) and mono.
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


# ===========================================================================
# Audio processing (echo cancellation, noise suppression, AGC)
# ===========================================================================

# ---------------------------------------------------------------------------
# AudioProcessingConfig validation
# ---------------------------------------------------------------------------


def test_config_defaults():
    assert AudioProcessingConfig() == AudioProcessingConfig(
        echo_cancellation=True,
        noise_suppression=True,
        auto_gain_control=True,
        ns_level=1,
        stream_delay_ms=0,
    )


@pytest.mark.parametrize("bad_level", [-1, 4, 10, 99])
def test_config_rejects_bad_ns_level(bad_level):
    with pytest.raises(ValueError, match="ns_level"):
        AudioProcessingConfig(ns_level=bad_level)


def test_config_rejects_negative_delay():
    with pytest.raises(ValueError, match="stream_delay_ms"):
        AudioProcessingConfig(stream_delay_ms=-5)


def test_config_rejects_excessive_delay():
    with pytest.raises(ValueError, match="stream_delay_ms"):
        AudioProcessingConfig(stream_delay_ms=5000)


def test_config_allows_disabling_echo_cancellation():
    # Headset case: noise suppression / AGC without echo cancellation.
    config = AudioProcessingConfig(echo_cancellation=False)
    assert config.echo_cancellation is False
    assert config.noise_suppression is True


# ---------------------------------------------------------------------------
# echo_cancellation toggle behaviour
# ---------------------------------------------------------------------------


def test_processor_start_respects_echo_cancellation_flag():
    module, processor_class, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        proc = _AudioProcessor(AudioProcessingConfig(echo_cancellation=False))
        proc.start(input_rate=16000, output_rate=16000, num_channels=1)

    assert processor_class.call_args.kwargs["echo_cancellation"] is False


def test_ec_off_passes_none_far_and_skips_reference():
    frame = np.ones(160, dtype=np.int16) * 1000
    cleaned = np.zeros(160, dtype=np.int16)

    processor = unittest.mock.MagicMock()
    processor.process.return_value = cleaned
    module, _, _ = _fake_pywebrtc(processor)

    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        proc = _AudioProcessor(AudioProcessingConfig(echo_cancellation=False))
        proc.start(input_rate=16000, output_rate=16000, num_channels=1)

        # record_playback is a no-op when EC is off.
        proc.record_playback(np.ones(160, dtype=np.int16).tobytes())
        proc.process_capture(frame.tobytes())

    _, far_arg = processor.process.call_args[0]
    assert far_arg is None


def test_process_capture_empty_input_returns_empty():
    # b"" is the shutdown sentinel emitted by _BidiAudioBuffer.stop(); it must not reach the C extension.
    processor = unittest.mock.MagicMock()
    module, _, _ = _fake_pywebrtc(processor)

    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        proc = _AudioProcessor(AudioProcessingConfig())
        proc.start(input_rate=16000, output_rate=16000, num_channels=1)
        result = proc.process_capture(b"")

    assert result == b""
    processor.process.assert_not_called()


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


def test_start_drains_stale_reference():
    # Restarting the processor (e.g. after a model reconnect) must not pair reference frames left over
    # from the previous session with fresh mic input.
    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        proc = _AudioProcessor(AudioProcessingConfig())
        proc.start(input_rate=16000, output_rate=16000, num_channels=1)
        proc.record_playback(b"\x01\x02\x03\x04")

        # Simulate a session restart.
        proc.start(input_rate=16000, output_rate=16000, num_channels=1)

        ref = proc._take_reference(np.zeros(2, dtype=np.int16))
        np.testing.assert_array_equal(ref, np.zeros(2, dtype=np.int16))


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
async def test_input_aligns_buffer_to_10ms_when_processing_on(py_audio, aec_agent):
    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig(), input_frames_per_buffer=999)
        input_ = audio_io.input()
        await input_.start(aec_agent)

    # 10ms at 16kHz = 160 samples, overriding the configured 999.
    assert py_audio.open.call_args.kwargs["frames_per_buffer"] == 160
    await input_.stop()


@pytest.mark.asyncio
async def test_input_keeps_configured_buffer_when_processing_off(py_audio, aec_agent):
    audio_io = BidiAudioIO(input_frames_per_buffer=1024)
    input_ = audio_io.input()
    await input_.start(aec_agent)

    assert py_audio.open.call_args.kwargs["frames_per_buffer"] == 1024
    await input_.stop()


def test_mic_buffer_bounded_to_reference_horizon_when_ec_on():
    # The mic buffer must share the reference buffer's frame bound so both evict in lockstep under a stall,
    # even if the user passes a larger input_buffer_size.
    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig(), input_buffer_size=9999)
        input_ = audio_io.input()

    assert input_._buffer._size == audio_io._processor._max_ref_frames


def test_mic_buffer_uses_configured_size_when_ec_off():
    # With echo cancellation off there is no reference to align to, so the user's sizing is respected.
    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig(echo_cancellation=False), input_buffer_size=7)
        input_ = audio_io.input()

    assert input_._buffer._size == 7


@pytest.mark.asyncio
async def test_output_aligns_buffer_to_10ms_at_output_rate(py_audio, agent_mixed_rates):
    # Output stream runs at output_rate (24k here). frame_count is measured in samples at the stream's
    # own rate, so the buffer must be sized off output_rate (240 = 10ms@24k), NOT input_rate (which would
    # give 160 = 6.67ms@24k and produce short, zero-padded reference frames).
    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig(), output_frames_per_buffer=999)
        output = audio_io.output()
        await output.start(agent_mixed_rates)

    assert py_audio.open.call_args.kwargs["frames_per_buffer"] == 240
    await output.stop()


@pytest.mark.asyncio
async def test_output_aligns_buffer_to_10ms_matched_rates(py_audio, aec_agent):
    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig())
        output = audio_io.output()
        await output.start(aec_agent)

    # 10ms at 16kHz = 160 samples.
    assert py_audio.open.call_args.kwargs["frames_per_buffer"] == 160
    await output.stop()


@pytest.mark.asyncio
async def test_output_keeps_configured_buffer_when_processing_off(py_audio, aec_agent):
    audio_io = BidiAudioIO(output_frames_per_buffer=2048)
    output = audio_io.output()
    await output.start(aec_agent)

    assert py_audio.open.call_args.kwargs["frames_per_buffer"] == 2048
    await output.stop()


@pytest.mark.asyncio
async def test_mixed_rate_reference_matches_mic_frame_length(py_audio, agent_mixed_rates):
    # End-to-end regression for the output-rate bug: with a correctly sized output buffer, a 10ms speaker
    # frame at 24k resamples to exactly a 10ms mic frame at 16k (320 bytes), so the reference is fully real
    # audio with no zero-padding.
    from strands.experimental.bidi.types.events import BidiAudioStreamEvent

    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig())
        output = audio_io.output()
        await output.start(agent_mixed_rates)

        # One 10ms playback frame at 24k = 240 samples = 480 bytes.
        speaker_frame = (np.arange(240, dtype=np.int16)).tobytes()
        await output(
            BidiAudioStreamEvent(
                audio=base64.b64encode(speaker_frame).decode("utf-8"),
                channels=1,
                format="pcm",
                sample_rate=24000,
            )
        )
        output._callback(None, frame_count=240)

        # A 10ms mic frame at 16k = 160 samples = 320 bytes. The resampled reference must fill it with real
        # audio (not zero-padded silence).
        mic = np.zeros(160, dtype=np.int16)
        ref = audio_io._processor._take_reference(mic)
        assert ref.shape == mic.shape
        # With the bug, only ~107 samples of real reference arrive and the trailing ~53 are zero-padded.
        # With the fix, the resampled ramp fills the whole frame, so the final samples are non-zero.
        assert ref[-1] != 0 and ref[-10] != 0

        await output.stop()


@pytest.mark.asyncio
async def test_output_records_reference_at_playback(py_audio, aec_agent):
    from strands.experimental.bidi.types.events import BidiAudioStreamEvent

    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig())
        output = audio_io.output()
        await output.start(aec_agent)

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
async def test_output_clears_reference_on_interruption(py_audio, aec_agent):
    from strands.experimental.bidi.types.events import BidiAudioStreamEvent, BidiInterruptionEvent

    module, _, _ = _fake_pywebrtc()
    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig())
        output = audio_io.output()
        await output.start(aec_agent)

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
async def test_end_to_end_capture_cancels_echo(py_audio, aec_agent):
    frame = np.ones(160, dtype=np.int16) * 1000
    cleaned = np.ones(160, dtype=np.int16) * 200

    processor = unittest.mock.MagicMock()
    processor.process.return_value = cleaned
    module, _, _ = _fake_pywebrtc(processor)

    with unittest.mock.patch.dict("sys.modules", {"pywebrtc_audio": module}):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig())
        input_ = audio_io.input()
        await input_.start(aec_agent)

        input_._buffer.put(frame.tobytes())
        event = await input_()

    result = np.frombuffer(base64.b64decode(event.audio), dtype=np.int16)
    np.testing.assert_array_equal(result, cleaned)


# ---------------------------------------------------------------------------
# Real pywebrtc-audio library contract (skipped when the extra is not installed)
#
# The tests above mock the C extension. These exercise the actual library — pure
# compute, no audio hardware — to catch contract drift the mocks cannot (e.g. the
# far=None-when-EC-on rejection, empty-frame rejection, and output length).
# ---------------------------------------------------------------------------


def test_real_library_roundtrip_and_contract():
    pytest.importorskip("pywebrtc_audio")

    # Echo cancellation on: first frame with an empty reference buffer must not crash (the empty buffer is
    # zero-filled, never passed as None), and output length matches input.
    proc = _AudioProcessor(AudioProcessingConfig())
    proc.start(input_rate=16000, output_rate=16000, num_channels=1)

    mic = (np.ones(160, dtype=np.int16) * 1000).tobytes()
    out = proc.process_capture(mic)
    assert len(out) == len(mic)

    # With a recorded reference the pairing still yields a matching-length frame.
    proc.record_playback((np.ones(160, dtype=np.int16) * 500).tobytes())
    assert len(proc.process_capture(mic)) == len(mic)

    # The empty shutdown sentinel is returned unchanged rather than reaching the C extension.
    assert proc.process_capture(b"") == b""


def test_real_library_echo_cancellation_off():
    pytest.importorskip("pywebrtc_audio")

    # Noise suppression / AGC without echo cancellation: no reference is used (far=None) and it still runs.
    proc = _AudioProcessor(AudioProcessingConfig(echo_cancellation=False))
    proc.start(input_rate=16000, output_rate=16000, num_channels=1)

    mic = (np.ones(160, dtype=np.int16) * 800).tobytes()
    assert len(proc.process_capture(mic)) == len(mic)
