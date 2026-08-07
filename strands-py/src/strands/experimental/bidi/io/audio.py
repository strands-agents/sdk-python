"""Send and receive audio data from devices.

Reads user audio from input device and sends agent audio to output device using PyAudio. If a user interrupts the agent,
the output buffer is cleared to stop playback.

Audio configuration is provided by the model via agent.model.config["audio"].

Audio processing (acoustic echo cancellation, noise suppression, and automatic gain control) is available when
pywebrtc-audio is installed (pip install strands-agents[bidi-aec]). Pass an ``AudioProcessingConfig`` to
``BidiAudioIO`` to enable it. When enabled, the agent's speaker output is used as a reference signal to cancel echo
from the microphone input so the model does not hear its own voice.
"""

import asyncio
import base64
import logging
import queue
from typing import TYPE_CHECKING, Any

import pyaudio

from ..types.events import BidiAudioInputEvent, BidiAudioStreamEvent, BidiInterruptionEvent, BidiOutputEvent
from ..types.io import AudioProcessingConfig, BidiInput, BidiOutput

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from ..agent.agent import BidiAgent

logger = logging.getLogger(__name__)

_SUPPORTED_SAMPLE_RATES = (16000, 32000, 48000)
"""Sample rates supported by pywebrtc-audio's AudioProcessor."""

_FRAME_DURATION_MS = 10
"""WebRTC audio processing operates on 10ms frames."""

_DEFAULT_MAX_REF_FRAMES = 100
"""Reference/mic buffer bound in frames (~1s at 10ms/frame). Both buffers share this bound so that under a
sustained input stall they evict oldest data in lockstep, keeping the newest mic frame paired with the newest
reference frame instead of letting the pairing drift apart."""


def _check_pywebrtc_available() -> None:
    """Verify pywebrtc-audio is importable.

    Raises:
        ImportError: If pywebrtc-audio is not installed.
    """
    try:
        import pywebrtc_audio  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "pywebrtc-audio is required for audio processing. Install it with: pip install strands-agents[bidi-aec]"
        ) from error


class _AudioProcessor:
    """Owns the echo-cancellation concern: reference buffer, WebRTC processor, and resampling.

    A single instance is shared between the output and input streams of a ``BidiAudioIO``. The output stream
    records every played frame as the far-end reference via ``record_playback``; the input stream cancels echo
    from captured mic audio via ``process_capture``.

    The underlying pywebrtc-audio AudioProcessor is not thread-safe. It is only ever invoked from
    ``process_capture`` on the input path, and those calls are serialized by the agent's ``await`` on the
    executor, so no two ``process()`` calls overlap even though ``asyncio.to_thread`` uses a thread pool. The
    reference buffer is a thread-safe queue shared across the input and output PyAudio callback threads.
    """

    def __init__(self, config: AudioProcessingConfig, max_ref_frames: int = _DEFAULT_MAX_REF_FRAMES) -> None:
        """Initialize the processor.

        Args:
            config: Audio processing configuration.
            max_ref_frames: Maximum number of reference frames to retain before dropping oldest.
        """
        self._config = config
        self._max_ref_frames = max_ref_frames
        self._ref_buffer: queue.Queue[bytes] = queue.Queue(maxsize=max_ref_frames)
        self._processor: Any = None
        self._input_rate: int | None = None
        self._output_rate: int | None = None

    def start(self, input_rate: int, output_rate: int, num_channels: int) -> None:
        """Build the underlying WebRTC processor for the given audio format.

        Args:
            input_rate: Microphone sample rate in Hz. Must be a supported rate.
            output_rate: Speaker sample rate in Hz (reference is resampled to input_rate if different).
            num_channels: Number of audio channels.

        Raises:
            ImportError: If pywebrtc-audio is not installed.
            ValueError: If input_rate is not supported by pywebrtc-audio.
        """
        if input_rate not in _SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"input_rate=<{input_rate}> | audio processing supports sample rates "
                f"{_SUPPORTED_SAMPLE_RATES}. Configure the model's audio input_rate accordingly."
            )

        from pywebrtc_audio import AudioProcessor  # type: ignore[import-not-found]

        # Discard any reference frames left over from a previous session so a restart never pairs stale
        # far-end audio with fresh mic input. Safe without a lock: start() runs before the PyAudio stream
        # and processing threads exist.
        _drain(self._ref_buffer)

        self._input_rate = input_rate
        self._output_rate = output_rate
        self._processor = AudioProcessor(
            sample_rate=input_rate,
            num_channels=num_channels,
            echo_cancellation=self._config.echo_cancellation,
            noise_suppression=self._config.noise_suppression,
            auto_gain_control=self._config.auto_gain_control,
            ns_level=self._config.ns_level,
            stream_delay_ms=self._config.stream_delay_ms,
        )
        logger.debug(
            "input_rate=<%d>, output_rate=<%d>, channels=<%d> | audio processor started",
            input_rate,
            output_rate,
            num_channels,
        )

    def record_playback(self, frame: bytes) -> None:
        """Record a played speaker frame as the far-end reference.

        Called from the output stream callback at the moment audio exits the speaker — the correct temporal
        alignment point for echo cancellation. The frame is resampled to the input rate when the speaker and
        mic rates differ. Drops the oldest reference frame on overflow.

        No-op when echo cancellation is disabled, since no reference signal is needed.

        Args:
            frame: PCM int16 speaker audio at the output sample rate.
        """
        if not self._config.echo_cancellation:
            return

        frame = self._resample_reference(frame)
        if self._ref_buffer.full():
            logger.debug("ref_buffer_full=<True> | echo reference overflow, dropping oldest frame")
            try:
                self._ref_buffer.get_nowait()
            except queue.Empty:
                pass
        self._ref_buffer.put_nowait(frame)

    def process_capture(self, mic_data: bytes) -> bytes:
        """Cancel echo from captured mic audio using the buffered reference.

        Args:
            mic_data: PCM int16 microphone audio.

        Returns:
            Cleaned PCM int16 audio of the same length. Empty input is returned unchanged (the WebRTC
            processor rejects empty frames, and ``_BidiAudioBuffer.stop`` emits an empty shutdown sentinel).
        """
        import numpy as np_

        if not mic_data:
            return mic_data

        near = np_.frombuffer(mic_data, dtype=np_.int16)
        # With echo cancellation off there is no reference signal; pass far=None (noise suppression and
        # auto gain control still run on the mic signal alone).
        far = self._take_reference(near) if self._config.echo_cancellation else None
        cleaned = self._processor.process(near, far)
        return cleaned.astype(np_.int16).tobytes()

    def reset_reference(self) -> None:
        """Drain the reference buffer. Called on interruption to preserve time alignment.

        The AEC filter itself is intentionally left converged — the acoustic path is unchanged by a barge-in.
        """
        _drain(self._ref_buffer)

    def _take_reference(self, mic: "npt.NDArray[np.int16]") -> "npt.NDArray[np.int16]":
        """Pull a reference frame aligned to the mic frame.

        The WebRTC AudioProcessor requires a non-null far-end frame of the same length as the near-end
        (mic) frame when echo cancellation is enabled. Any shortfall — including an empty reference buffer
        because the speaker is silent — is filled with zeros (silence).

        Args:
            mic: int16 numpy array of mic samples for the current frame.

        Returns:
            int16 numpy array of reference samples, same length as ``mic``.
        """
        import numpy as np_

        byte_count = mic.nbytes
        data = bytearray()
        while len(data) < byte_count:
            try:
                data.extend(self._ref_buffer.get_nowait())
            except queue.Empty:
                break

        if len(data) < byte_count:
            data.extend(b"\x00" * (byte_count - len(data)))

        ref: npt.NDArray[np.int16] = np_.frombuffer(bytes(data[:byte_count]), dtype=np_.int16)
        return ref

    def _resample_reference(self, data: bytes) -> bytes:
        """Resample speaker audio to the input rate via linear interpolation.

        Returns data unchanged when rates match.
        """
        if self._output_rate is None or self._input_rate is None or self._output_rate == self._input_rate:
            return data

        import numpy as np_

        samples = np_.frombuffer(data, dtype=np_.int16)
        if len(samples) == 0:
            return data

        ratio = self._input_rate / self._output_rate
        new_length = max(int(round(len(samples) * ratio)), 1)
        positions = np_.linspace(0, len(samples) - 1, new_length)
        resampled = np_.interp(positions, np_.arange(len(samples)), samples.astype(np_.float32))
        return resampled.astype(np_.int16).tobytes()


def _drain(buffer: "queue.Queue[Any]") -> None:
    """Remove all items from a queue without blocking."""
    while True:
        try:
            buffer.get_nowait()
        except queue.Empty:
            break


class _BidiAudioBuffer:
    """Buffer chunks of audio data between agent and PyAudio."""

    _buffer: queue.Queue[bytes]
    _data: bytearray

    def __init__(self, size: int | None = None):
        """Initialize buffer settings.

        Args:
            size: Size of the buffer (default: unbounded).
        """
        self._size = size or 0

    def start(self) -> None:
        """Setup buffer."""
        self._buffer = queue.Queue(self._size)
        self._data = bytearray()

    def stop(self) -> None:
        """Tear down buffer."""
        if hasattr(self, "_data"):
            self._data.clear()
        if hasattr(self, "_buffer"):
            # Unblocking waited get calls by putting an empty chunk
            # Note, Queue.shutdown exists but is a 3.13+ only feature
            # We simulate shutdown with the below logic
            self._buffer.put_nowait(b"")
            self._buffer = queue.Queue(self._size)

    def put(self, chunk: bytes) -> None:
        """Put data chunk into buffer.

        If full, removes the oldest chunk.
        """
        if self._buffer.full():
            logger.debug("buffer is full | removing oldest chunk")
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                logger.debug("buffer already empty")
                pass

        self._buffer.put_nowait(chunk)

    def get(self, byte_count: int | None = None) -> bytes:
        """Get the number of bytes specified from the buffer.

        Args:
            byte_count: Number of bytes to get from buffer.

                - If the number of bytes specified is not available, the return is padded with silence.
                - If the number of bytes is not specified, get the first chunk put in the buffer.

        Returns:
            Specified number of bytes.
        """
        if not byte_count:
            self._data.extend(self._buffer.get())
            byte_count = len(self._data)

        while len(self._data) < byte_count:
            try:
                self._data.extend(self._buffer.get_nowait())
            except queue.Empty:
                break

        padding_bytes = b"\x00" * max(byte_count - len(self._data), 0)
        self._data.extend(padding_bytes)

        data = self._data[:byte_count]
        del self._data[:byte_count]

        return bytes(data)

    def clear(self) -> None:
        """Clear the buffer."""
        while True:
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                break


class _BidiAudioInput(BidiInput):
    """Handle audio input from user.

    Attributes:
        _audio: PyAudio instance for audio system access.
        _stream: Audio input stream.
        _buffer: Buffer for sharing audio data between agent and PyAudio.
    """

    _audio: pyaudio.PyAudio
    _stream: pyaudio.Stream

    _BUFFER_SIZE = None
    _DEVICE_INDEX = None
    _FRAMES_PER_BUFFER = 512

    def __init__(self, config: dict[str, Any], processor: _AudioProcessor | None = None) -> None:
        """Extract configs.

        Args:
            config: Audio device configuration.
            processor: Shared audio processor for echo cancellation, or None to disable processing.
        """
        self._buffer_size = config.get("input_buffer_size", _BidiAudioInput._BUFFER_SIZE)
        self._device_index = config.get("input_device_index", _BidiAudioInput._DEVICE_INDEX)
        self._configured_frames_per_buffer = config.get("input_frames_per_buffer", _BidiAudioInput._FRAMES_PER_BUFFER)
        self._processor = processor

        # When echo cancellation is on, bound the mic buffer to the same frame horizon as the reference
        # buffer so that under a sustained input stall both evict their oldest frames in lockstep, keeping
        # the newest mic frame paired with the newest reference frame. An unbounded (or differently sized)
        # mic buffer would let the reference saturate and drop frames first, inverting the pairing and
        # collapsing echo cancellation. This overrides input_buffer_size while processing is on.
        if processor is not None and processor._config.echo_cancellation:
            buffer_size = processor._max_ref_frames
        else:
            buffer_size = self._buffer_size

        self._buffer = _BidiAudioBuffer(buffer_size)

    async def start(self, agent: "BidiAgent") -> None:
        """Start input stream.

        Args:
            agent: The BidiAgent instance, providing access to model configuration.

        Raises:
            ValueError: If audio processing is enabled but the input rate is unsupported.
        """
        logger.debug("starting audio input stream")

        self._channels = agent.model.config["audio"]["channels"]
        self._format = agent.model.config["audio"]["format"]
        self._rate = agent.model.config["audio"]["input_rate"]

        if self._processor is not None:
            self._processor.start(
                input_rate=self._rate,
                output_rate=agent.model.config["audio"]["output_rate"],
                num_channels=self._channels,
            )
            # WebRTC processes 10ms frames; align the device buffer so each callback delivers whole frames.
            # NOTE (follow-up): this makes __call__ emit one BidiAudioInputEvent per 10ms frame = ~100
            # events/s to the model, vs ~31/s at the non-processing default of 512 samples. Processing must
            # stay at 10ms frames for AEC quality, but the wire cadence could be decoupled by accumulating
            # several processed frames into one event before sending. Deferred until the 100/s cadence is
            # measured against a live provider for throttling/cost impact.
            frames_per_buffer = self._rate * _FRAME_DURATION_MS // 1000
        else:
            frames_per_buffer = self._configured_frames_per_buffer

        self._buffer.start()
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            channels=self._channels,
            format=pyaudio.paInt16,
            frames_per_buffer=frames_per_buffer,
            input=True,
            input_device_index=self._device_index,
            rate=self._rate,
            stream_callback=self._callback,
        )

        logger.debug("audio input stream started")

    async def stop(self) -> None:
        """Stop input stream."""
        logger.debug("stopping audio input stream")

        if hasattr(self, "_stream"):
            self._stream.close()
        if hasattr(self, "_audio"):
            self._audio.terminate()
        if hasattr(self, "_buffer"):
            self._buffer.stop()

        logger.debug("audio input stream stopped")

    async def __call__(self) -> BidiAudioInputEvent:
        """Read audio from input stream, applying echo cancellation if enabled."""
        data = await asyncio.to_thread(self._buffer.get)

        if self._processor is not None:
            data = await asyncio.to_thread(self._processor.process_capture, data)

        return BidiAudioInputEvent(
            audio=base64.b64encode(data).decode("utf-8"),
            channels=self._channels,
            format=self._format,
            sample_rate=self._rate,
        )

    def _callback(self, in_data: bytes, *_: Any) -> tuple[None, Any]:
        """Callback to receive audio data from PyAudio."""
        self._buffer.put(in_data)
        return (None, pyaudio.paContinue)


class _BidiAudioOutput(BidiOutput):
    """Handle audio output from bidi agent.

    Attributes:
        _audio: PyAudio instance for audio system access.
        _stream: Audio output stream.
        _buffer: Buffer for sharing audio data between agent and PyAudio.
    """

    _audio: pyaudio.PyAudio
    _stream: pyaudio.Stream

    _BUFFER_SIZE = None
    _DEVICE_INDEX = None
    _FRAMES_PER_BUFFER = 512

    def __init__(self, config: dict[str, Any], processor: _AudioProcessor | None = None) -> None:
        """Extract configs.

        Args:
            config: Audio device configuration.
            processor: Shared audio processor — output records played frames as the AEC reference.
        """
        self._buffer_size = config.get("output_buffer_size", _BidiAudioOutput._BUFFER_SIZE)
        self._device_index = config.get("output_device_index", _BidiAudioOutput._DEVICE_INDEX)
        self._configured_frames_per_buffer = config.get("output_frames_per_buffer", _BidiAudioOutput._FRAMES_PER_BUFFER)

        self._buffer = _BidiAudioBuffer(self._buffer_size)
        self._processor = processor

    async def start(self, agent: "BidiAgent") -> None:
        """Start output stream.

        Args:
            agent: The BidiAgent instance, providing access to model configuration.
        """
        logger.debug("starting audio output stream")

        self._channels = agent.model.config["audio"]["channels"]
        self._rate = agent.model.config["audio"]["output_rate"]

        if self._processor is not None:
            # Align the output buffer to a 10ms frame at the OUTPUT rate. frame_count in the PyAudio callback
            # is measured in samples at the stream's own (output) rate, so sizing off input_rate would open a
            # buffer of the wrong duration when output_rate != input_rate (e.g. Gemini 16k in / 24k out),
            # yielding short reference frames that get zero-padded and degrade echo cancellation.
            frames_per_buffer = self._rate * _FRAME_DURATION_MS // 1000
        else:
            frames_per_buffer = self._configured_frames_per_buffer

        self._buffer.start()
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            channels=self._channels,
            format=pyaudio.paInt16,
            frames_per_buffer=frames_per_buffer,
            output=True,
            output_device_index=self._device_index,
            rate=self._rate,
            stream_callback=self._callback,
        )

        logger.debug("audio output stream started")

    async def stop(self) -> None:
        """Stop output stream."""
        logger.debug("stopping audio output stream")

        if hasattr(self, "_stream"):
            self._stream.close()
        if hasattr(self, "_audio"):
            self._audio.terminate()
        if hasattr(self, "_buffer"):
            self._buffer.stop()

        logger.debug("audio output stream stopped")

    async def __call__(self, event: BidiOutputEvent) -> None:
        """Send audio to output stream."""
        if isinstance(event, BidiAudioStreamEvent):
            data = base64.b64decode(event["audio"])
            self._buffer.put(data)
            logger.debug("audio_bytes=<%d> | audio chunk buffered for playback", len(data))

        elif isinstance(event, BidiInterruptionEvent):
            logger.debug("reason=<%s> | clearing audio buffer due to interruption", event["reason"])
            self._buffer.clear()
            if self._processor is not None:
                self._processor.reset_reference()

    def _callback(self, _in_data: None, frame_count: int, *_: Any) -> tuple[bytes, Any]:
        """Callback to send audio data to PyAudio.

        When processing is enabled, records the played audio as the echo reference at the moment it exits the
        speaker — the correct temporal alignment point for echo cancellation.
        """
        byte_count = frame_count * pyaudio.get_sample_size(pyaudio.paInt16)
        data = self._buffer.get(byte_count)

        if self._processor is not None:
            self._processor.record_playback(data)

        return (data, pyaudio.paContinue)


class BidiAudioIO:
    """Send and receive audio data from devices using PyAudio.

    Reads microphone audio via ``input()`` and plays agent audio via ``output()``. On interruption, the
    playback buffer is cleared to stop the agent mid-response.

    When an ``AudioProcessingConfig`` is provided, the agent's speaker output is used as a reference signal to
    cancel echo from the microphone input, preventing the model from hearing its own voice (and optionally
    applying noise suppression and automatic gain control). The same processor instance is shared between the
    input and output channels this factory produces, so echo cancellation only works when both ``input()`` and
    ``output()`` come from the *same* ``BidiAudioIO`` instance.

    Audio processing requires pywebrtc-audio (``pip install strands-agents[bidi-aec]``) and a microphone
    sample rate of 16000, 32000, or 48000 Hz (set via the model's audio config).

    Example:
        ```python
        from strands.experimental.bidi.io import AudioProcessingConfig, BidiAudioIO

        # Plain mic/speaker, no processing (a headset is recommended to avoid echo):
        audio_io = BidiAudioIO()
        await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

        # With echo cancellation, noise suppression, and auto gain control:
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig())
        await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

        # Noise suppression and auto gain control without echo cancellation (e.g. headset users):
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig(echo_cancellation=False))
        await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

        # Tuned processing on a specific input device:
        audio_io = BidiAudioIO(
            audio_processing=AudioProcessingConfig(noise_suppression=False, ns_level=2),
            input_device_index=1,
        )
        await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])
        ```
    """

    def __init__(self, *, audio_processing: AudioProcessingConfig | None = None, **config: Any) -> None:
        """Initialize audio devices.

        Args:
            audio_processing: Enable microphone audio processing (echo cancellation, and optionally noise
                suppression and automatic gain control) by passing an ``AudioProcessingConfig``. Requires
                pywebrtc-audio (pip install strands-agents[bidi-aec]). Defaults to None (processing disabled).
            **config: Optional device configuration:

                - input_buffer_size (int): Maximum input buffer size (default: None)
                - input_device_index (int): Specific input device (default: None = system default)
                - input_frames_per_buffer (int): Input buffer size (default: 512, ignored when processing is on)
                - output_buffer_size (int): Maximum output buffer size (default: None)
                - output_device_index (int): Specific output device (default: None = system default)
                - output_frames_per_buffer (int): Output buffer size (default: 512, ignored when processing is on)

        Raises:
            ImportError: If audio_processing is set but pywebrtc-audio is not installed.
        """
        self._config = config

        if audio_processing is not None:
            _check_pywebrtc_available()
            self._processor: _AudioProcessor | None = _AudioProcessor(audio_processing)
        else:
            self._processor = None

    def input(self) -> _BidiAudioInput:
        """Return audio processing BidiInput."""
        return _BidiAudioInput(self._config, processor=self._processor)

    def output(self) -> _BidiAudioOutput:
        """Return audio processing BidiOutput."""
        return _BidiAudioOutput(self._config, processor=self._processor)
