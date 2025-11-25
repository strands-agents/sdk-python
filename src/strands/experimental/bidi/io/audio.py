"""Send and receive audio data from devices.

Reads user audio from input device and sends agent audio to output device using PyAudio. If a user interrupts the agent,
the output buffer is cleared to stop playback.

Audio configuration is provided by the model via agent.model.audio_config.
"""

import asyncio
import base64
import logging
from collections import deque
from typing import TYPE_CHECKING, Any

import pyaudio

from ..types.events import BidiAudioInputEvent, BidiAudioStreamEvent, BidiInterruptionEvent, BidiOutputEvent
from ..types.io import BidiInput, BidiOutput

if TYPE_CHECKING:
    from ..agent.agent import BidiAgent

logger = logging.getLogger(__name__)


class _BidiAudioInput(BidiInput):
    """Handle audio input from user.

    Attributes:
        _audio: PyAudio instance for audio system access.
        _stream: Audio input stream.
    """

    _audio: pyaudio.PyAudio
    _stream: pyaudio.Stream

    # Audio device constants
    _DEVICE_INDEX: int | None = None
    _PYAUDIO_FORMAT: int = pyaudio.paInt16
    _FRAMES_PER_BUFFER: int = 512

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize audio input handler.
        
        Args:
            config: Configuration dictionary with optional overrides:
                - input_device_index: Specific input device to use
                - input_pyaudio_format: PyAudio format (default: paInt16)
                - input_frames_per_buffer: Number of frames per buffer
        """
        # Initialize instance variables from config or class constants
        self._device_index = config.get("input_device_index", self._DEVICE_INDEX)
        self._pyaudio_format = config.get("input_pyaudio_format", self._PYAUDIO_FORMAT)
        self._frames_per_buffer = config.get("input_frames_per_buffer", self._FRAMES_PER_BUFFER)

    async def start(self, agent: "BidiAgent") -> None:
        """Start input stream.

        Args:
            agent: The BidiAgent instance, providing access to model configuration.
        """
        # Get audio parameters from model config
        self._rate = agent.model.config["audio"]["input_rate"]
        self._channels = agent.model.config["audio"]["channels"]
        self._format = agent.model.config["audio"].get("format", "pcm")  # Encoding format for events

        logger.debug(
            "rate=<%d>, channels=<%d>, device_index=<%s> | starting audio input stream",
            self._rate,
            self._channels,
            self._device_index,
        )
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            channels=self._channels,
            format=self._pyaudio_format,
            frames_per_buffer=self._frames_per_buffer,
            input=True,
            input_device_index=self._device_index,
            rate=self._rate,
        )
        logger.info("rate=<%d>, channels=<%d> | audio input stream started", self._rate, self._channels)

    async def stop(self) -> None:
        """Stop input stream."""
        logger.debug("stopping audio input stream")
        # TODO: Provide time for streaming thread to exit cleanly to prevent conflicts with the Nova threads.
        #       See if we can remove after properly handling cancellation for agent.
        await asyncio.sleep(0.1)

        self._stream.close()
        self._audio.terminate()

        logger.debug("audio input stream stopped")

    async def __call__(self) -> BidiAudioInputEvent:
        """Read audio from input stream."""
        audio_bytes = await asyncio.to_thread(self._stream.read, self._frames_per_buffer, exception_on_overflow=False)

        return BidiAudioInputEvent(
            audio=base64.b64encode(audio_bytes).decode("utf-8"),
            channels=self._channels,
            format=self._format,
            sample_rate=self._rate,
        )


class _BidiAudioOutput(BidiOutput):
    """Handle audio output from bidi agent.

    Attributes:
        _audio: PyAudio instance for audio system access.
        _stream: Audio output stream.
        _buffer: Deque buffer for queuing audio data.
        _buffer_event: Event to signal when buffer has data.
        _output_task: Background task for processing audio output.
    """

    _audio: pyaudio.PyAudio
    _stream: pyaudio.Stream
    _buffer: deque
    _buffer_event: asyncio.Event
    _output_task: asyncio.Task

    # Audio device constants
    _BUFFER_SIZE: int | None = None
    _DEVICE_INDEX: int | None = None
    _PYAUDIO_FORMAT: int = pyaudio.paInt16
    _FRAMES_PER_BUFFER: int = 512

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize audio output handler.
        
        Args:
            config: Configuration dictionary with optional overrides:
                - output_device_index: Specific output device to use
                - output_pyaudio_format: PyAudio format (default: paInt16)
                - output_frames_per_buffer: Number of frames per buffer
                - output_buffer_size: Maximum buffer size (None = unlimited)
        """
        # Initialize instance variables from config or class constants
        self._buffer_size = config.get("output_buffer_size", self._BUFFER_SIZE)
        self._device_index = config.get("output_device_index", self._DEVICE_INDEX)
        self._pyaudio_format = config.get("output_pyaudio_format", self._PYAUDIO_FORMAT)
        self._frames_per_buffer = config.get("output_frames_per_buffer", self._FRAMES_PER_BUFFER)

    async def start(self, agent: "BidiAgent") -> None:
        """Start output stream.

        Args:
            agent: The BidiAgent instance, providing access to model configuration.
        """
        # Get audio parameters from model config
        self._rate = agent.model.config["audio"]["output_rate"]
        self._channels = agent.model.config["audio"]["channels"]

        logger.debug(
            "rate=<%d>, channels=<%d> | starting audio output stream",
            self._rate,
            self._channels,
        )
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            channels=self._channels,
            format=self._pyaudio_format,
            frames_per_buffer=self._frames_per_buffer,
            output=True,
            output_device_index=self._device_index,
            rate=self._rate,
        )
        self._buffer = deque(maxlen=self._buffer_size)
        self._buffer_event = asyncio.Event()
        self._output_task = asyncio.create_task(self._output())
        logger.info("rate=<%d>, channels=<%d> | audio output stream started", self._rate, self._channels)

    async def stop(self) -> None:
        """Stop output stream."""
        logger.debug("stopping audio output stream")
        self._buffer.clear()
        self._buffer.append(None)
        self._buffer_event.set()
        await self._output_task

        self._stream.close()
        self._audio.terminate()

        logger.debug("audio output stream stopped")

    async def __call__(self, event: BidiOutputEvent) -> None:
        """Handle audio events with direct stream writing."""
        if isinstance(event, BidiAudioStreamEvent):
            audio_bytes = base64.b64decode(event["audio"])
            self._buffer.append(audio_bytes)
            self._buffer_event.set()
            logger.debug("audio_bytes=<%d> | audio chunk buffered for playback", len(audio_bytes))

        elif isinstance(event, BidiInterruptionEvent):
            logger.debug("reason=<%s> | clearing audio buffer due to interruption", event["reason"])
            self._buffer.clear()
            self._buffer_event.clear()

    async def _output(self) -> None:
        while True:
            await self._buffer_event.wait()
            self._buffer_event.clear()

            while self._buffer:
                audio_bytes = self._buffer.popleft()
                if not audio_bytes:
                    return

                await asyncio.to_thread(self._stream.write, audio_bytes)


class BidiAudioIO:
    """Send and receive audio data from devices."""

    def __init__(self, **config: Any) -> None:
        """Initialize audio devices.
        
        Args:
            **config: Optional device configuration:
                - input_device_index (int): Specific input device (default: None = system default)
                - output_device_index (int): Specific output device (default: None = system default)
                - input_pyaudio_format (int): PyAudio format for input (default: pyaudio.paInt16)
                - output_pyaudio_format (int): PyAudio format for output (default: pyaudio.paInt16)
                - input_frames_per_buffer (int): Input buffer size (default: 512)
                - output_frames_per_buffer (int): Output buffer size (default: 512)
                - output_buffer_size (int | None): Max output queue size (default: None = unlimited)
        """
        self._config = config

    def input(self) -> _BidiAudioInput:
        """Return audio processing BidiInput."""
        return _BidiAudioInput(self._config)

    def output(self) -> _BidiAudioOutput:
        """Return audio processing BidiOutput."""
        return _BidiAudioOutput(self._config)
