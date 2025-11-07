"""AudioIO - Clean separation of audio functionality from core BidirectionalAgent.

Provides audio input/output capabilities for BidirectionalAgent through the BidirectionalIO protocol.
Handles all PyAudio setup, streaming, and cleanup while keeping the core agent data-agnostic.
"""

import asyncio
import base64
import logging
from typing import Any, Callable, Optional

from .bidirectional_io import BidirectionalIO

try:
    import pyaudio
except ImportError:
    pyaudio = None

logger = logging.getLogger(__name__)


class AudioIO(BidirectionalIO):
    """Audio IO channel for BidirectionalAgent with direct stream processing."""

    def __init__(
        self,
        audio_config: Optional[dict] = None,
    ):
        """Initialize AudioIO with clean audio configuration.

        Args:
            audio_config: Dictionary containing audio configuration:
                - input_sample_rate (int): Microphone sample rate (default: 24000)
                - output_sample_rate (int): Speaker sample rate (default: 24000)
                - chunk_size (int): Audio chunk size in bytes (default: 1024)
                - input_device_index (int): Specific input device (optional)
                - output_device_index (int): Specific output device (optional)
                - input_channels (int): Input channels (default: 1)
                - output_channels (int): Output channels (default: 1)
        """
        if pyaudio is None:
            raise ImportError("PyAudio is required for AudioIO. Install with: pip install pyaudio")

        # Default audio configuration
        default_config = {
            "input_sample_rate": 24000,
            "output_sample_rate": 24000,
            "chunk_size": 1024,
            "input_device_index": None,
            "output_device_index": None,
            "input_channels": 1,
            "output_channels": 1,
        }

        # Merge user config with defaults
        if audio_config:
            default_config.update(audio_config)

        # Set audio configuration attributes
        self.input_sample_rate = default_config["input_sample_rate"]
        self.output_sample_rate = default_config["output_sample_rate"]
        self.chunk_size = default_config["chunk_size"]
        self.input_device_index = default_config["input_device_index"]
        self.output_device_index = default_config["output_device_index"]
        self.input_channels = default_config["input_channels"]
        self.output_channels = default_config["output_channels"]

        # Audio infrastructure
        self.audio = None
        self.input_stream = None
        self.output_stream = None
        self.interrupted = False

    def _setup_audio(self) -> None:
        """Setup PyAudio streams for input and output."""
        if self.audio:
            return

        self.audio = pyaudio.PyAudio()

        try:
            # Input stream
            self.input_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.input_channels,
                rate=self.input_sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                input_device_index=self.input_device_index,
            )

            # Output stream
            self.output_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.output_channels,
                rate=self.output_sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size,
                output_device_index=self.output_device_index,
            )

            # Start streams
            self.input_stream.start_stream()
            self.output_stream.start_stream()

        except Exception as e:
            logger.error(f"AudioIO: Audio setup failed: {e}")
            self._cleanup_audio()
            raise

    def _cleanup_audio(self) -> None:
        """Clean up PyAudio resources."""
        try:
            if self.input_stream:
                if self.input_stream.is_active():
                    self.input_stream.stop_stream()
                self.input_stream.close()

            if self.output_stream:
                if self.output_stream.is_active():
                    self.output_stream.stop_stream()
                self.output_stream.close()

            if self.audio:
                self.audio.terminate()

            self.input_stream = None
            self.output_stream = None
            self.audio = None

        except Exception as e:
            logger.warning(f"Audio cleanup error: {e}")

    async def input_channel(self) -> dict:
        """Read audio from microphone."""
        if not self.input_stream:
            self._setup_audio()

        try:
            audio_bytes = self.input_stream.read(self.chunk_size, exception_on_overflow=False)
            return {
                "audioData": audio_bytes,
                "format": "pcm",
                "sampleRate": self.input_sample_rate,
                "channels": self.input_channels,
            }
        except Exception as e:
            logger.warning(f"Audio input error: {e}")
            return {
                "audioData": b"",
                "format": "pcm",
                "sampleRate": self.input_sample_rate,
                "channels": self.input_channels,
            }

    async def output_channel(self, event: dict) -> None:
        """Handle audio events with direct stream writing."""
        if not self.output_stream:
            self._setup_audio()

        # Handle audio output
        if "audioOutput" in event and not self.interrupted:
            audio_data = event["audioOutput"]["audioData"]

            # Handle both base64 and raw bytes
            if isinstance(audio_data, str):
                audio_data = base64.b64decode(audio_data)

            if audio_data:
                chunk_size = 2048
                for i in range(0, len(audio_data), chunk_size):
                    # Check for interruption before each chunk
                    if self.interrupted:
                        break

                    chunk = audio_data[i : i + chunk_size]
                    try:
                        self.output_stream.write(chunk, exception_on_underflow=False)
                        await asyncio.sleep(0)
                    except Exception as e:
                        logger.warning(f"Audio playback error: {e}")
                        break

        elif "interruptionDetected" in event or "interrupted" in event:
            self.interrupted = True
            logger.debug("Interruption detected")

            # Stop and restart stream for immediate interruption
            if self.output_stream:
                try:
                    self.output_stream.stop_stream()
                    self.output_stream.start_stream()
                except Exception as e:
                    logger.debug(f"Error clearing audio buffer: {e}")

            self.interrupted = False

        elif "textOutput" in event:
            text = event["textOutput"].get("text", "").strip()
            role = event["textOutput"].get("role", "")
            if text:
                if role.upper() == "ASSISTANT":
                    print(f"🤖 {text}")
                elif role.upper() == "USER":
                    print(f"User: {text}")

    def cleanup(self) -> None:
        """Clean up IO channel resources."""
        self._cleanup_audio()
