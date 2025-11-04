"""AudioAdapter - Clean separation of audio functionality from core BidirectionalAgent.

Provides audio input/output capabilities for BidirectionalAgent through the adapter pattern.
Handles all PyAudio setup, streaming, and cleanup while keeping the core agent data-agnostic.
"""

import asyncio
import base64
import logging
from typing import Any, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..agent import BidirectionalAgent

try:
    import pyaudio
except ImportError:
    pyaudio = None

logger = logging.getLogger(__name__)


class AudioAdapter:
    """Audio adapter for BidirectionalAgent with queue-based processing."""
    
    def __init__(
        self,
        agent: "BidirectionalAgent",
        audio_config: Optional[dict] = None,
    ):
        """Initialize AudioAdapter with clean audio configuration.
        
        Args:
            agent: The BidirectionalAgent instance to wrap
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
            raise ImportError("PyAudio is required for AudioAdapter. Install with: pip install pyaudio")
        
        self.agent = agent
        
        # Default audio configuration
        default_config = {
            "input_sample_rate": 24000,
            "output_sample_rate": 24000,
            "chunk_size": 1024,
            "input_device_index": None,
            "output_device_index": None,
            "input_channels": 1,
            "output_channels": 1
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
        
        # Audio infrastructure (lazy initialization)
        self.audio = None
        self.input_stream = None
        self.output_stream = None
        self.interrupted = False
        
        # Audio output queue for background processing
        self.audio_output_queue = asyncio.Queue()

    def _setup_audio(self) -> None:
        """Setup PyAudio streams for input and output."""
        if self.audio:
            return  # Already setup
        
        self.audio = pyaudio.PyAudio()
        
        try:
            # Input stream (microphone)
            self.input_stream = self.audio.open(
                format=pyaudio.paInt16, 
                channels=self.input_channels, 
                rate=self.input_sample_rate,
                input=True, 
                frames_per_buffer=self.chunk_size,
                input_device_index=self.input_device_index
            )
            
            # Output stream (speakers)
            self.output_stream = self.audio.open(
                format=pyaudio.paInt16, 
                channels=self.output_channels, 
                rate=self.output_sample_rate,
                output=True, 
                frames_per_buffer=self.chunk_size,
                output_device_index=self.output_device_index
            )
            
            # Start streams - required for audio to flow
            self.input_stream.start_stream()
            self.output_stream.start_stream()
            
        except Exception as e:
            logger.error(f"AudioAdapter: Audio setup failed: {e}")
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

    def create_input(self) -> Callable[[], dict]:
        """Create audio input function for agent.run()."""
        async def audio_receiver() -> dict:
            """Read audio from microphone."""
            if not self.input_stream:
                self._setup_audio()
            
            try:
                audio_bytes = self.input_stream.read(self.chunk_size, exception_on_overflow=False)
                return {
                    "audioData": audio_bytes,
                    "format": "pcm", 
                    "sampleRate": self.input_sample_rate,
                    "channels": self.input_channels  # Use configured channels
                }
            except Exception as e:
                logger.warning(f"Audio input error: {e}")
                return {"audioData": b"", "format": "pcm", "sampleRate": self.input_sample_rate, "channels": self.input_channels}
        
        return audio_receiver
    
    def create_output(self) -> Callable[[dict], None]:
        """Create output function that queues audio for background processing."""
        
        # Start background audio processor once
        if not hasattr(self, '_audio_task') or self._audio_task.done():
            self._audio_task = asyncio.create_task(self._process_audio_queue())
        
        events_queued = 0
        
        async def audio_sender(event: dict) -> None:
            """Queue audio events with minimal debug."""
            nonlocal events_queued
            
            if "audioOutput" in event:
                if not self.interrupted:
                    audio_data = event["audioOutput"]["audioData"]
                    self.audio_output_queue.put_nowait(audio_data)
                    events_queued += 1

            elif "interruptionDetected" in event or "interrupted" in event:
                self.interrupted = True
                cleared = 0
                while not self.audio_output_queue.empty():
                    try:
                        self.audio_output_queue.get_nowait()
                        cleared += 1
                    except asyncio.QueueEmpty:
                        break
                logger.debug(f"Cleared {cleared} audio chunks on interruption")
                self.interrupted = False

            elif "textOutput" in event:
                text = event["textOutput"].get("text", "")
                role = event["textOutput"].get("role", "")
                if role.upper() == "ASSISTANT":
                    logger.info(f"Assistant: {text}")
                elif role.upper() == "USER":
                    logger.info(f"User: {text}")
        
        return audio_sender

    async def _process_audio_queue(self):
        """Audio processor without performance-killing delays."""
        logger.debug("Audio processor started - optimized")
        
        # Separate PyAudio instance for background processing
        audio = pyaudio.PyAudio()
        speaker = audio.open(
            channels=self.output_channels, 
            format=pyaudio.paInt16, 
            output=True,
            rate=self.output_sample_rate, 
            frames_per_buffer=self.chunk_size,
            output_device_index=self.output_device_index
        )

        try:
            chunks = 0
            while True:
                try:
                    # Get audio from queue
                    audio_data = await asyncio.wait_for(self.audio_output_queue.get(), timeout=0.1)

                    if audio_data and not self.interrupted:
                        chunks += 1
                        
                        # Use chunked playback like working test_bidi_openai.py
                        chunk_size = 1024
                        for i in range(0, len(audio_data), chunk_size):
                            if self.interrupted:
                                break
                            
                            chunk = audio_data[i:i + chunk_size]
                            speaker.write(chunk)
                            await asyncio.sleep(0.001)  # Same as working implementation

                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break

        finally:
            logger.debug(f"AudioAdapter finished processing {chunks} chunks")
            speaker.close()
            audio.terminate()

    async def chat(self, duration: Optional[float] = None) -> None:
        """Start voice conversation using agent.run() pattern."""
        try:
            self._setup_audio()
            
            if duration:
                await asyncio.wait_for(
                    self.agent.run(
                        sender=self.create_output(),
                        receiver=self.create_input()
                    ),
                    timeout=duration
                )
            else:
                await self.agent.run(
                    sender=self.create_output(),
                    receiver=self.create_input()
                )
                
        except KeyboardInterrupt:
            logger.info("Conversation ended by user")
        except asyncio.TimeoutError:
            logger.info(f"Conversation ended after {duration}s timeout")
        finally:
            if hasattr(self, '_audio_task'):
                self._audio_task.cancel()
            self._cleanup_audio()

    # Context manager support
    async def __aenter__(self) -> "AudioAdapter":
        """Async context manager entry."""
        self._setup_audio()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit with cleanup."""
        self._cleanup_audio()

