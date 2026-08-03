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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pyaudio

from ..types.events import BidiAudioInputEvent, BidiAudioStreamEvent, BidiInterruptionEvent, BidiOutputEvent
from ..types.io import BidiInput, BidiOutput

if TYPE_CHECKING:
    from ..agent.agent import BidiAgent

logger = logging.getLogger(__name__)

_SUPPORTED_SAMPLE_RATES = (16000, 32000, 48000)
"""Sample rates supported by pywebrtc-audio's AudioProcessor."""

_FRAME_DURATION_MS = 10
"""WebRTC audio processing operates on 10ms frames."""


@dataclass
class AudioProcessingConfig:
    """Configuration for microphone audio processing.

    Passing an instance of this config to ``BidiAudioIO`` enables acoustic echo cancellation (AEC). Echo
    cancellation is always on when processing is enabled; noise suppression and automatic gain control are
    tunable extras. Requires pywebrtc-audio (pip install strands-agents[bidi-aec]).

    Attributes:
        noise_suppression: Enable noise suppression.
        auto_gain_control: Enable automatic gain control.
        ns_level: Noise suppression aggressiveness, 0-3 (6dB, 12dB, 18dB, 21dB).
        stream_delay_ms: Playback-to-capture delay hint in ms for AEC. 0 lets AEC3 auto-estimate the delay.
    """

    noise_suppression: bool = True
    auto_gain_control: bool = True
    ns_level: int = 1
    stream_delay_ms: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If ns_level is not in 0-3 or stream_delay_ms is negative.
        """
        if self.ns_level not in (0, 1, 2, 3):
            raise ValueError(f"ns_level=<{self.ns_level}> | must be 0, 1, 2, or 3")
        if self.stream_delay_ms < 0:
            raise ValueError(f"stream_delay_ms=<{self.stream_delay_ms}> | must be non-negative")


def _check_pywebrtc_available() -> None:
    """Verify pywebrtc-audio is importable.

    Raises:
        ImportError: If pywebrtc-audio is not installed.
    """
    try:
        import pywebrtc_audio  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "pywebrtc-audio is required for audio processing. "
            "Install it with: pip install strands-agents[bidi-aec]"
        ) from error


class _AudioProcessor:
    """Owns the echo-cancellation concern: reference buffer, WebRTC processor, and resampling.

    A single instance is shared between the output and input streams of a ``BidiAudioIO``. The output stream
    records every played frame as the far-end reference via ``record_playback``; the input stream cancels echo
    from captured mic audio via ``process_capture``.

    The underlying pywebrtc-audio AudioProcessor is not thread-safe. It is only ever invoked from the input
    path (``process_capture``), so all processing stays on a single thread. The reference buffer is a
    thread-safe queue shared across the input and output callback threads.
    """

    def __init__(self, config: AudioProcessingConfig, max_ref_frames: int = 100) -> None:
        """Initialize the processor.

        Args:
            config: Audio processing configuration.
            max_ref_frames: Maximum number of reference frames to retain before dropping oldest.
        """
        self._config = config
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

        self._input_rate = input_rate
        self._output_rate = output_rate
        self._processor = AudioProcessor(
            sample_rate=input_rate,
            num_channels=num_channels,
            echo_cancellation=True,
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

        Args:
            frame: PCM int16 speaker audio at the output sample rate.
        """
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
            Cleaned PCM int16 audio of the same length.
        """
        import numpy as np_

        near = np_.frombuffer(mic_data, dtype=np_.int16)
        far = self._take_reference(near)
        cleaned = self._processor.process(near, far)
        return cleaned.astype(np_.int16).tobytes()

    def reset_reference(self) -> None:
        """Drain the reference buffer. Called on interruption to preserve time alignment.

        The AEC filter itself is intentionally left converged — the acoustic path is unchanged by a barge-in.
        """
        _drain(self._ref_buffer)

    def reset(self) -> None:
        """Reset the AEC filter and drain the reference buffer.

        For genuine session resets (e.g. a new conversation), not interruptions. Kept available for future
        lifecycle wiring; not invoked automatically today.
        """
        _drain(self._ref_buffer)
        if self._processor is not None:
            self._processor.reset()

    def _take_reference(self, mic: "Any") -> "Any":
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

        return np_.frombuffer(bytes(data[:byte_count]), dtype=np_.int16)

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

        self._buffer = _BidiAudioBuffer(self._buffer_size)
        self._processor = processor

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
        self._configured_frames_per_buffer = config.get(
            "output_frames_per_buffer", _BidiAudioOutput._FRAMES_PER_BUFFER
        )

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
            input_rate = agent.model.config["audio"]["input_rate"]
            # Align the output buffer to a 10ms frame at the input rate so recorded reference frames match
            # the mic frame cadence used by the processor.
            frames_per_buffer = input_rate * _FRAME_DURATION_MS // 1000
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
        # Plain mic/speaker, no processing (a headset is recommended to avoid echo):
        audio_io = BidiAudioIO()
        await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

        # With echo cancellation, noise suppression, and auto gain control:
        audio_io = BidiAudioIO(audio_processing=AudioProcessingConfig())
        await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

        # Tuned processing:
        audio_io = BidiAudioIO(
            audio_processing=AudioProcessingConfig(noise_suppression=False, ns_level=2),
            input_device_index=1,
        )
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
