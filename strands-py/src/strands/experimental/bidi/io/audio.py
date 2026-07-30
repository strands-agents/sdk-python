"""Send and receive audio data from devices.

Reads user audio from input device and sends agent audio to output device using PyAudio. If a user interrupts the agent,
the output buffer is cleared to stop playback.

Audio configuration is provided by the model via agent.model.config["audio"].

Echo cancellation is available when pywebrtc-audio is installed (pip install strands-agents[bidi-aec]).
When enabled, the agent's speaker output is subtracted from the microphone input so the model
does not hear its own voice echoed back.
"""

import asyncio
import base64
import logging
import queue
from typing import TYPE_CHECKING, Any, Protocol

import pyaudio

from ..types.events import BidiAudioInputEvent, BidiAudioStreamEvent, BidiInterruptionEvent, BidiOutputEvent
from ..types.io import BidiInput, BidiOutput

if TYPE_CHECKING:
    import numpy as np

    from ..agent.agent import BidiAgent

logger = logging.getLogger(__name__)

_AEC_FRAME_DURATION_MS = 10
"""Frame duration in milliseconds for WebRTC audio processing (fixed at 10ms)."""


class AudioProcessingStage(Protocol):
    """Protocol for an audio processing stage in the pipeline.

    Each stage receives a mic frame and an optional reference frame, and returns
    the processed mic frame. Stages are composed in order by the pipeline.
    """

    def process(self, mic_frame: "np.ndarray", ref_frame: "np.ndarray | None") -> "np.ndarray":
        """Process a single audio frame.

        Args:
            mic_frame: int16 numpy array of mic samples (one 10ms frame).
            ref_frame: int16 numpy array of speaker reference samples, or None if silent.

        Returns:
            Processed int16 numpy array of the same length.
        """
        ...


class _WebRTCProcessingStage:
    """Processing stage wrapping pywebrtc-audio's AudioProcessor.

    Provides echo cancellation, noise suppression, and automatic gain control
    via WebRTC's production audio processing algorithms.
    """

    def __init__(
        self,
        sample_rate: int,
        num_channels: int,
        noise_suppression: bool = True,
        auto_gain_control: bool = True,
        stream_delay_ms: int = 0,
    ) -> None:
        """Initialize the WebRTC audio processor.

        Args:
            sample_rate: Audio sample rate in Hz.
            num_channels: Number of audio channels.
            noise_suppression: Enable noise suppression.
            auto_gain_control: Enable automatic gain control.
            stream_delay_ms: Estimated playback-to-capture delay in ms.

        Raises:
            ImportError: If pywebrtc-audio is not installed.
        """
        try:
            from pywebrtc_audio import AudioProcessor
        except ImportError as error:
            raise ImportError(
                "pywebrtc-audio is required for echo cancellation. "
                "Install it with: pip install strands-agents[bidi-aec]"
            ) from error

        self._processor = AudioProcessor(
            sample_rate=sample_rate,
            num_channels=num_channels,
            echo_cancellation=True,
            noise_suppression=noise_suppression,
            auto_gain_control=auto_gain_control,
            stream_delay_ms=stream_delay_ms,
        )

    def process(self, mic_frame: "np.ndarray", ref_frame: "np.ndarray | None") -> "np.ndarray":
        """Process a frame through WebRTC AEC/NS/AGC.

        Args:
            mic_frame: int16 mic samples.
            ref_frame: int16 speaker reference samples (silence if None).

        Returns:
            Cleaned int16 samples.
        """
        import numpy as np_

        if ref_frame is None:
            ref_frame = np_.zeros_like(mic_frame)

        return self._processor.process(mic_frame, ref_frame)


class _AudioProcessingPipeline:
    """Composable pipeline that runs audio through a sequence of processing stages.

    Handles 10ms frame chunking (required by WebRTC) and delegates per-frame
    processing to each stage in order.
    """

    def __init__(self, stages: list[AudioProcessingStage], sample_rate: int, num_channels: int) -> None:
        """Initialize the pipeline.

        Args:
            stages: Ordered list of processing stages to apply.
            sample_rate: Audio sample rate in Hz.
            num_channels: Number of audio channels.
        """
        self._stages = stages
        self._frame_size = (sample_rate * _AEC_FRAME_DURATION_MS // 1000) * num_channels

    def process(self, mic_data: bytes, ref_data: bytes) -> bytes:
        """Process mic audio through all pipeline stages.

        Splits audio into 10ms frames, runs each through every stage, and
        reassembles the result.

        Args:
            mic_data: Raw PCM int16 microphone audio.
            ref_data: Raw PCM int16 reference (speaker) audio of equal length.

        Returns:
            Processed PCM int16 audio.
        """
        import numpy as np_

        mic_array = np_.frombuffer(mic_data, dtype=np_.int16)
        ref_array = np_.frombuffer(ref_data, dtype=np_.int16)

        output_frames = []
        for offset in range(0, len(mic_array), self._frame_size):
            mic_frame = mic_array[offset : offset + self._frame_size]
            ref_frame = ref_array[offset : offset + self._frame_size]

            if len(mic_frame) < self._frame_size:
                mic_frame = np_.pad(mic_frame, (0, self._frame_size - len(mic_frame)))
                ref_frame = np_.pad(ref_frame, (0, self._frame_size - len(ref_frame)))

            for stage in self._stages:
                mic_frame = stage.process(mic_frame, ref_frame)

            output_frames.append(mic_frame)

        result = np_.concatenate(output_frames)
        return result[: len(mic_array)].astype(np_.int16).tobytes()


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


class _EchoReferenceBuffer:
    """Thread-safe ring buffer storing the far-end (speaker) reference signal for AEC.

    The output stream writes every frame it plays into this buffer. The input stream
    reads the corresponding reference frame to feed the echo canceller. If no reference
    is available (speaker is silent), silence is returned.
    """

    def __init__(self, max_frames: int = 100) -> None:
        """Initialize the reference buffer.

        Args:
            max_frames: Maximum number of reference frames to retain.
        """
        self._buffer: queue.Queue[bytes] = queue.Queue(maxsize=max_frames)

    def put(self, frame: bytes) -> None:
        """Store a reference frame (called from the output playback path)."""
        if self._buffer.full():
            logger.debug("ref_buffer_full=<True> | echo reference overflow, dropping oldest frame")
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                pass
        self._buffer.put_nowait(frame)

    def get(self, byte_count: int) -> bytes:
        """Retrieve reference audio matching the requested byte count.

        Returns silence if no reference is available.
        """
        data = bytearray()
        while len(data) < byte_count:
            try:
                data.extend(self._buffer.get_nowait())
            except queue.Empty:
                break

        if len(data) < byte_count:
            data.extend(b"\x00" * (byte_count - len(data)))

        return bytes(data[:byte_count])

    def clear(self) -> None:
        """Discard all buffered reference frames."""
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

    def __init__(
        self,
        config: dict[str, Any],
        echo_ref_buffer: _EchoReferenceBuffer | None = None,
        echo_cancellation: bool = False,
        noise_suppression: bool = True,
        auto_gain_control: bool = True,
        stream_delay_ms: int = 0,
    ) -> None:
        """Extract configs.

        Args:
            config: Audio device configuration.
            echo_ref_buffer: Shared reference buffer from the output stream for AEC.
            echo_cancellation: Whether to apply echo cancellation.
            noise_suppression: Whether to apply noise suppression (requires echo_cancellation).
            auto_gain_control: Whether to apply automatic gain control (requires echo_cancellation).
            stream_delay_ms: Estimated delay between playback and capture in milliseconds.
        """
        self._buffer_size = config.get("input_buffer_size", _BidiAudioInput._BUFFER_SIZE)
        self._device_index = config.get("input_device_index", _BidiAudioInput._DEVICE_INDEX)
        self._frames_per_buffer = config.get("input_frames_per_buffer", _BidiAudioInput._FRAMES_PER_BUFFER)

        self._buffer = _BidiAudioBuffer(self._buffer_size)
        self._echo_ref_buffer = echo_ref_buffer
        self._echo_cancellation = echo_cancellation
        self._noise_suppression = noise_suppression
        self._auto_gain_control = auto_gain_control
        self._stream_delay_ms = stream_delay_ms
        self._pipeline: _AudioProcessingPipeline | None = None

    async def start(self, agent: "BidiAgent") -> None:
        """Start input stream.

        Args:
            agent: The BidiAgent instance, providing access to model configuration.

        Raises:
            ImportError: If echo cancellation is enabled but pywebrtc-audio is not installed.
        """
        logger.debug("starting audio input stream")

        self._channels = agent.model.config["audio"]["channels"]
        self._format = agent.model.config["audio"]["format"]
        self._rate = agent.model.config["audio"]["input_rate"]

        if self._echo_cancellation:
            webrtc_stage = _WebRTCProcessingStage(
                sample_rate=self._rate,
                num_channels=self._channels,
                noise_suppression=self._noise_suppression,
                auto_gain_control=self._auto_gain_control,
                stream_delay_ms=self._stream_delay_ms,
            )
            self._pipeline = _AudioProcessingPipeline(
                stages=[webrtc_stage],
                sample_rate=self._rate,
                num_channels=self._channels,
            )
            logger.debug(
                "sample_rate=<%d>, channels=<%d>, stream_delay_ms=<%d> | echo cancellation initialized",
                self._rate,
                self._channels,
                self._stream_delay_ms,
            )

        self._buffer.start()
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            channels=self._channels,
            format=pyaudio.paInt16,
            frames_per_buffer=self._frames_per_buffer,
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

        self._pipeline = None
        logger.debug("audio input stream stopped")

    async def __call__(self) -> BidiAudioInputEvent:
        """Read audio from input stream, applying echo cancellation if enabled."""
        data = await asyncio.to_thread(self._buffer.get)

        if self._pipeline is not None and self._echo_ref_buffer is not None:
            ref_data = self._echo_ref_buffer.get(len(data))
            data = self._pipeline.process(data, ref_data)

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

    def __init__(
        self,
        config: dict[str, Any],
        echo_ref_buffer: _EchoReferenceBuffer | None = None,
    ) -> None:
        """Extract configs.

        Args:
            config: Audio device configuration.
            echo_ref_buffer: Shared reference buffer for AEC — output writes reference frames here.
        """
        self._buffer_size = config.get("output_buffer_size", _BidiAudioOutput._BUFFER_SIZE)
        self._device_index = config.get("output_device_index", _BidiAudioOutput._DEVICE_INDEX)
        self._frames_per_buffer = config.get("output_frames_per_buffer", _BidiAudioOutput._FRAMES_PER_BUFFER)

        self._buffer = _BidiAudioBuffer(self._buffer_size)
        self._echo_ref_buffer = echo_ref_buffer
        self._output_rate: int | None = None
        self._input_rate: int | None = None

    async def start(self, agent: "BidiAgent") -> None:
        """Start output stream.

        Args:
            agent: The BidiAgent instance, providing access to model configuration.
        """
        logger.debug("starting audio output stream")

        self._channels = agent.model.config["audio"]["channels"]
        self._rate = agent.model.config["audio"]["output_rate"]

        if self._echo_ref_buffer is not None:
            self._output_rate = agent.model.config["audio"]["output_rate"]
            self._input_rate = agent.model.config["audio"]["input_rate"]
            if self._output_rate != self._input_rate:
                logger.debug(
                    "output_rate=<%d>, input_rate=<%d> | reference will be resampled for AEC",
                    self._output_rate,
                    self._input_rate,
                )

        self._buffer.start()
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            channels=self._channels,
            format=pyaudio.paInt16,
            frames_per_buffer=self._frames_per_buffer,
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
            if self._echo_ref_buffer is not None:
                self._echo_ref_buffer.clear()

    def _resample_reference(self, data: bytes) -> bytes:
        """Resample output audio to the input sample rate for AEC reference.

        If rates match, returns data unchanged.

        Args:
            data: PCM int16 audio at the output sample rate.

        Returns:
            PCM int16 audio resampled to the input sample rate.
        """
        if self._output_rate is None or self._input_rate is None or self._output_rate == self._input_rate:
            return data

        import numpy as np_

        samples = np_.frombuffer(data, dtype=np_.int16)
        ratio = self._input_rate / self._output_rate
        new_length = int(len(samples) * ratio)
        indices = np_.linspace(0, len(samples) - 1, new_length).astype(np_.int32)
        resampled = samples[indices]
        return resampled.astype(np_.int16).tobytes()

    def _callback(self, _in_data: None, frame_count: int, *_: Any) -> tuple[bytes, Any]:
        """Callback to send audio data to PyAudio.

        When AEC is enabled, writes the played audio to the echo reference buffer
        at the moment it exits the speaker — this is the correct temporal alignment
        point for echo cancellation.
        """
        byte_count = frame_count * pyaudio.get_sample_size(pyaudio.paInt16)
        data = self._buffer.get(byte_count)

        if self._echo_ref_buffer is not None:
            ref_data = self._resample_reference(data)
            self._echo_ref_buffer.put(ref_data)

        return (data, pyaudio.paContinue)


class BidiAudioIO:
    """Send and receive audio data from devices.

    When echo_cancellation is enabled, the agent's speaker output is used as a reference
    signal to cancel echo from the microphone input, preventing the model from hearing
    its own voice.
    """

    def __init__(self, echo_cancellation: bool = False, **config: Any) -> None:
        """Initialize audio devices.

        Args:
            echo_cancellation: Enable acoustic echo cancellation. Requires pywebrtc-audio
                to be installed (pip install strands-agents[bidi-aec]).
            **config: Optional device configuration:

                - input_buffer_size (int): Maximum input buffer size (default: None)
                - input_device_index (int): Specific input device (default: None = system default)
                - input_frames_per_buffer (int): Input buffer size (default: 512)
                - output_buffer_size (int): Maximum output buffer size (default: None)
                - output_device_index (int): Specific output device (default: None = system default)
                - output_frames_per_buffer (int): Output buffer size (default: 512)
                - noise_suppression (bool): Enable noise suppression with AEC (default: True)
                - auto_gain_control (bool): Enable auto gain control with AEC (default: True)
                - stream_delay_ms (int): Estimated playback-to-capture delay in ms (default: 0)
        """
        self._config = config
        self._echo_cancellation = echo_cancellation
        self._echo_ref_buffer = _EchoReferenceBuffer() if echo_cancellation else None

    def input(self) -> _BidiAudioInput:
        """Return audio processing BidiInput."""
        return _BidiAudioInput(
            self._config,
            echo_ref_buffer=self._echo_ref_buffer,
            echo_cancellation=self._echo_cancellation,
            noise_suppression=self._config.get("noise_suppression", True),
            auto_gain_control=self._config.get("auto_gain_control", True),
            stream_delay_ms=self._config.get("stream_delay_ms", 0),
        )

    def output(self) -> _BidiAudioOutput:
        """Return audio processing BidiOutput."""
        return _BidiAudioOutput(
            self._config,
            echo_ref_buffer=self._echo_ref_buffer,
        )
