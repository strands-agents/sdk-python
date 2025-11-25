"""Send and receive audio data from devices.

Reads user audio from input device and sends agent audio to output device using PyAudio. If a user interrupts the agent,
the output buffer is cleared to stop playback.
"""

import asyncio
import base64
import logging
from collections import deque
from typing import TYPE_CHECKING, Any

import pyaudio

from ..types.events import BidiAudioInputEvent, BidiAudioStreamEvent, BidiInterruptionEvent, BidiOutputEvent
from ..types.io import AudioConfig, BidiInput, BidiOutput

if TYPE_CHECKING:
    from ..agent.agent import BidiAgent

logger = logging.getLogger(__name__)


class _BidiAudioInput(BidiInput):
    """Handle audio input from user.

    Attributes:
        _audio: PyAudio instance for audio system access.
        _stream: Audio input stream.
        _user_config_set: Track which config values were explicitly set by user.
    """

    _audio: pyaudio.PyAudio
    _stream: pyaudio.Stream
    _user_config_set: set[str]

    _CHANNELS: int = 1
    _DEVICE_INDEX: int | None = None
    _ENCODING: str = "pcm"
    _FORMAT: int = pyaudio.paInt16
    _FRAMES_PER_BUFFER: int = 512
    _RATE: int = 16000

    def __init__(self, config: dict[str, Any]) -> None:
        """Extract configs and track which were explicitly set by user."""
        # Track which config values were explicitly provided by user
        self._user_config_set = set(config.keys())

        self._channels = config.get("input_channels", _BidiAudioInput._CHANNELS)
        self._device_index = config.get("input_device_index", _BidiAudioInput._DEVICE_INDEX)
        self._format = config.get("input_format", _BidiAudioInput._FORMAT)
        self._frames_per_buffer = config.get("input_frames_per_buffer", _BidiAudioInput._FRAMES_PER_BUFFER)
        self._rate = config.get("input_rate", _BidiAudioInput._RATE)

    async def start(self, agent: "BidiAgent") -> None:
        """Start input stream.

        Args:
            agent: The BidiAgent instance, providing access to model configuration.
        """
        # Extract audio config from agent's model
        audio_config = getattr(agent.model, "audio_config", None)
        
        # Apply audio config overrides only if user didn't explicitly set them
        if audio_config:
            if "input_rate" in audio_config and "input_rate" not in self._user_config_set:
                self._rate = audio_config["input_rate"]
                logger.debug("audio_config | applying model input rate: %d Hz", self._rate)
            if "channels" in audio_config and "input_channels" not in self._user_config_set:
                self._channels = audio_config["channels"]
                logger.debug("audio_config | applying model channels: %d", self._channels)

        logger.debug(
            "rate=<%d>, channels=<%d>, device_index=<%s> | starting audio input stream",
            self._rate,
            self._channels,
            self._device_index,
        )
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            channels=self._channels,
            format=self._format,
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
            format=_BidiAudioInput._ENCODING,
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
        _user_config_set: Track which config values were explicitly set by user.
    """

    _audio: pyaudio.PyAudio
    _stream: pyaudio.Stream
    _buffer: deque
    _buffer_event: asyncio.Event
    _output_task: asyncio.Task
    _user_config_set: set[str]

    _BUFFER_SIZE: int | None = None
    _CHANNELS: int = 1
    _DEVICE_INDEX: int | None = None
    _FORMAT: int = pyaudio.paInt16
    _FRAMES_PER_BUFFER: int = 512
    _RATE: int = 16000

    def __init__(self, config: dict[str, Any]) -> None:
        """Extract configs and track which were explicitly set by user."""
        # Track which config values were explicitly provided by user
        self._user_config_set = set(config.keys())

        self._buffer_size = config.get("output_buffer_size", _BidiAudioOutput._BUFFER_SIZE)
        self._channels = config.get("output_channels", _BidiAudioOutput._CHANNELS)
        self._device_index = config.get("output_device_index", _BidiAudioOutput._DEVICE_INDEX)
        self._format = config.get("output_format", _BidiAudioOutput._FORMAT)
        self._frames_per_buffer = config.get("output_frames_per_buffer", _BidiAudioOutput._FRAMES_PER_BUFFER)
        self._rate = config.get("output_rate", _BidiAudioOutput._RATE)

    async def start(self, agent: "BidiAgent") -> None:
        """Start output stream.

        Args:
            agent: The BidiAgent instance, providing access to model configuration.
        """
        # Extract audio config from agent's model
        audio_config = getattr(agent.model, "audio_config", None)
        
        # Apply audio config overrides only if user didn't explicitly set them
        if audio_config:
            if "output_rate" in audio_config and "output_rate" not in self._user_config_set:
                self._rate = audio_config["output_rate"]
                logger.debug("audio_config | applying model output rate: %d Hz", self._rate)
            if "channels" in audio_config and "output_channels" not in self._user_config_set:
                self._channels = audio_config["channels"]
                logger.debug("audio_config | applying model channels: %d", self._channels)

        logger.debug(
            "rate=<%d>, channels=<%d>, device_index=<%s>, buffer_size=<%s> | starting audio output stream",
            self._rate,
            self._channels,
            self._device_index,
            self._buffer_size,
        )
        self._audio = pyaudio.PyAudio()
        self._stream = self._audio.open(
            channels=self._channels,
            format=self._format,
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
            **config: Dictionary containing audio configuration:
                - input_channels (int): Input channels (default: 1)
                - input_device_index (int): Specific input device (optional)
                - input_format (int): Audio format (default: paInt16)
                - input_frames_per_buffer (int): Frames per buffer (default: 512)
                - input_rate (int): Input sample rate (default: 16000)
                - output_buffer_size (int): Maximum output buffer size (default: None)
                - output_channels (int): Output channels (default: 1)
                - output_device_index (int): Specific output device (optional)
                - output_format (int): Audio format (default: paInt16)
                - output_frames_per_buffer (int): Frames per buffer (default: 512)
                - output_rate (int): Output sample rate (default: 16000)
        """
        self._config = config

    def input(self) -> _BidiAudioInput:
        """Return audio processing BidiInput."""
        return _BidiAudioInput(self._config)

    def output(self) -> _BidiAudioOutput:
        """Return audio processing BidiOutput."""
        return _BidiAudioOutput(self._config)
