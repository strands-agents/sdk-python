"""Send and receive audio data from devices.

Reads user audio from input device and sends agent audio to output device using PyAudio. If a user interrupts the agent,
the output buffer is cleared to stop playback.

Audio configuration is provided by the model via agent.model.config["audio"].

Optional microphone audio processing (acoustic echo cancellation, noise suppression, and automatic gain
control) is enabled by passing an ``AudioProcessorConfig`` (from ``strands.experimental.bidi.audio``) to
``BidiAudioIO``. It requires pywebrtc-audio (pip install strands-agents[bidi-aec]); see that module for the
processing implementation.
"""

import asyncio
import base64
import logging
import queue
from typing import TYPE_CHECKING, Any

import pyaudio

from ..types.events import BidiAudioInputEvent, BidiAudioStreamEvent, BidiInterruptionEvent, BidiOutputEvent
from ..types.io import BidiInput, BidiOutput

if TYPE_CHECKING:
    from ..agent.agent import BidiAgent
    from ..audio import AudioProcessorConfig, _AudioProcessor

logger = logging.getLogger(__name__)


def _check_pywebrtc_available() -> None:
    """Verify pywebrtc-audio is importable.

    Raises:
        ImportError: If pywebrtc-audio is not installed.
    """
    try:
        import pywebrtc_audio  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "pywebrtc-audio is required for audio processing. Install it with: pip install strands-agents[bidi-aec]"
        ) from error


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
            # Unblock waited get calls by putting an empty chunk.
            # Note, Queue.shutdown exists but is a 3.13+ only feature; we simulate shutdown with the below
            # logic. Drop the oldest chunk first when full (a bounded buffer can be full during teardown,
            # e.g. after a stall) so the sentinel always lands and stop() never raises queue.Full over the
            # real teardown error.
            if self._buffer.full():
                try:
                    self._buffer.get_nowait()
                except queue.Empty:
                    pass
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

    def __init__(self, config: dict[str, Any], processor: "_AudioProcessor | None" = None) -> None:
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
        buffer_size: int | None
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
            self._processor.configure(
                input_rate=self._rate,
                output_rate=agent.model.config["audio"]["output_rate"],
                num_channels=self._channels,
            )
            # WebRTC processes 10ms frames; align the device buffer so each callback delivers whole frames.
            # NOTE (follow-up): this makes __call__ emit one BidiAudioInputEvent per 10ms frame = ~100
            # events/s to the model, vs ~31/s at the non-processing default of 512 samples. Any multiple of
            # 10ms preserves AEC quality (ERLE is unchanged at 20/30/40ms), so the wire cadence could be
            # decoupled by batching several processed frames into one event (e.g. 20-30ms). Deferred until
            # the cadence is measured against a live provider for throttling/cost impact.
            frames_per_buffer = self._rate * self._processor._FRAME_DURATION_MS // 1000
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

    def __init__(self, config: dict[str, Any], processor: "_AudioProcessor | None" = None) -> None:
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
            frames_per_buffer = self._rate * self._processor._FRAME_DURATION_MS // 1000
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
                self._processor.clear_reference()

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

    When an ``AudioProcessorConfig`` is passed as ``processor``, the microphone signal gets noise suppression
    and automatic gain control, and (when echo cancellation is enabled) the agent's speaker output is used as a
    reference to cancel echo from the mic input, preventing the model from hearing its own voice. The same
    processor instance is shared between the input and output channels this factory produces, so echo
    cancellation only works when both ``input()`` and ``output()`` come from the *same* ``BidiAudioIO``
    instance.

    Audio processing requires pywebrtc-audio (``pip install strands-agents[bidi-aec]``) and a microphone
    sample rate of 16000, 32000, or 48000 Hz (set via the model's audio config).

    Example:
        ```python
        from strands.experimental.bidi.audio import AudioProcessorConfig
        from strands.experimental.bidi.io import BidiAudioIO

        # Plain mic/speaker, no processing (a headset is recommended to avoid echo):
        audio_io = BidiAudioIO()
        await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

        # Full processing: echo cancellation, noise suppression, and auto gain control:
        audio_io = BidiAudioIO(processor=AudioProcessorConfig())
        await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

        # Noise suppression and auto gain control without echo cancellation (e.g. headset users):
        audio_io = BidiAudioIO(processor=AudioProcessorConfig(echo_cancellation=False))
        await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

        # Processing on a specific input device:
        audio_io = BidiAudioIO(processor=AudioProcessorConfig(), input_device_index=1)
        await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])
        ```
    """

    def __init__(self, **config: Any) -> None:
        """Initialize audio devices.

        Args:
            **config: Optional configuration:

                - processor (AudioProcessorConfig): Enable microphone audio processing (noise suppression,
                  automatic gain control, and optionally echo cancellation) by passing an
                  ``AudioProcessorConfig`` (from ``strands.experimental.bidi.audio``). Requires pywebrtc-audio
                  (pip install strands-agents[bidi-aec]). Defaults to None (processing disabled).
                - input_buffer_size (int): Maximum input buffer size (default: None). Ignored when echo
                  cancellation is on — the mic buffer is bound to the reference horizon so the two stay aligned.
                - input_device_index (int): Specific input device (default: None = system default)
                - input_frames_per_buffer (int): Input buffer size (default: 512, ignored when processing is on)
                - output_buffer_size (int): Maximum output buffer size (default: None)
                - output_device_index (int): Specific output device (default: None = system default)
                - output_frames_per_buffer (int): Output buffer size (default: 512, ignored when processing is on)

        Raises:
            ImportError: If a processor config is set but pywebrtc-audio is not installed.
        """
        processor_config: AudioProcessorConfig | None = config.pop("processor", None)
        self._config = config

        if processor_config is not None:
            _check_pywebrtc_available()
            from ..audio import _AudioProcessor

            self._processor: _AudioProcessor | None = _AudioProcessor(processor_config)
        else:
            self._processor = None

    def input(self) -> _BidiAudioInput:
        """Return audio processing BidiInput."""
        return _BidiAudioInput(self._config, processor=self._processor)

    def output(self) -> _BidiAudioOutput:
        """Return audio processing BidiOutput."""
        return _BidiAudioOutput(self._config, processor=self._processor)
