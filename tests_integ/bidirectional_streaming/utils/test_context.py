"""Test context manager for bidirectional streaming tests.

Provides a high-level interface for testing bidirectional streaming agents
with continuous background threads that mimic real-world usage patterns.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strands.experimental.bidirectional_streaming.agent.agent import BidirectionalAgent
    from .audio_generator import AudioGenerator

logger = logging.getLogger(__name__)


class BidirectionalTestContext:
    """Manages threads and generators for bidirectional streaming tests.

    Mimics real-world usage with continuous background threads:
    - Audio input thread (microphone simulation with silence padding)
    - Event collection thread (captures all model outputs)

    Generators feed data into threads via queues for natural conversation flow.

    Example:
        async with BidirectionalTestContext(agent, audio_generator) as ctx:
            await ctx.say("What is 5 plus 3?")
            await ctx.wait_for_response()
            assert "8" in " ".join(ctx.get_text_outputs())
    """

    def __init__(
        self,
        agent: "BidirectionalAgent",
        audio_generator: "AudioGenerator | None" = None,
        silence_chunk_size: int = 1024,
        audio_chunk_size: int = 1024,
    ):
        """Initialize test context.

        Args:
            agent: BidirectionalAgent instance.
            audio_generator: AudioGenerator for text-to-speech.
            silence_chunk_size: Size of silence chunks in bytes.
            audio_chunk_size: Size of audio chunks for streaming.
        """
        self.agent = agent
        self.audio_generator = audio_generator
        self.silence_chunk_size = silence_chunk_size
        self.audio_chunk_size = audio_chunk_size

        # Queue for thread communication
        self.input_queue = asyncio.Queue()  # Handles both audio and text input

        # Event storage
        self.events = []  # All collected events
        self.last_event_time = None

        # Control flags
        self.active = False
        self.threads = []

    async def __aenter__(self):
        """Start context manager, agent session, and background threads."""
        # Start agent session
        await self.agent.start()
        logger.debug("Agent session started")
        
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop context manager, cleanup threads, and end agent session."""
        await self.stop()
        
        # End agent session
        if self.agent._session and self.agent._session.active:
            await self.agent.end()
            logger.debug("Agent session ended")
        
        return False

    async def start(self):
        """Start all background threads."""
        self.active = True
        self.last_event_time = asyncio.get_event_loop().time()

        self.threads = [
            asyncio.create_task(self._input_thread()),
            asyncio.create_task(self._event_collection_thread()),
        ]

        logger.debug("Test context started with %d threads", len(self.threads))

    async def stop(self):
        """Stop all threads gracefully."""
        self.active = False

        # Cancel all threads
        for task in self.threads:
            if not task.done():
                task.cancel()

        # Wait for cancellation
        await asyncio.gather(*self.threads, return_exceptions=True)

        logger.debug("Test context stopped")

    # === User-facing methods ===

    async def say(self, text: str):
        """Queue text to be converted to audio and sent to model.

        Args:
            text: Text to convert to speech and send as audio.
        """
        await self.input_queue.put({"type": "audio", "text": text})
        logger.debug(f"Queued speech: {text[:50]}...")

    async def send(self, data: str | dict) -> None:
        """Send data directly to model (text, image, etc.).

        Args:
            data: Data to send to model. Can be:
                - str: Text input
                - dict: Custom event (e.g., image, audio)
        """
        await self.input_queue.put({"type": "direct", "data": data})
        logger.debug(f"Queued direct send: {type(data).__name__}")

    async def wait_for_response(
        self,
        timeout: float = 15.0,
        silence_threshold: float = 2.0,
        min_events: int = 1,
    ):
        """Wait for model to finish responding.

        Uses silence detection (no events for silence_threshold seconds)
        combined with minimum event count to determine response completion.

        Args:
            timeout: Maximum time to wait in seconds.
            silence_threshold: Seconds of silence to consider response complete.
            min_events: Minimum events before silence detection activates.
        """
        start_time = asyncio.get_event_loop().time()
        initial_event_count = len(self.events)

        while asyncio.get_event_loop().time() - start_time < timeout:
            # Check if we have minimum events
            if len(self.events) - initial_event_count >= min_events:
                # Check silence
                elapsed_since_event = asyncio.get_event_loop().time() - self.last_event_time
                if elapsed_since_event >= silence_threshold:
                    logger.debug(
                        f"Response complete: {len(self.events) - initial_event_count} events, "
                        f"{elapsed_since_event:.1f}s silence"
                    )
                    return

            await asyncio.sleep(0.1)

        logger.warning(f"Response timeout after {timeout}s")

    def get_events(self, event_type: str | None = None) -> list[dict]:
        """Get collected events, optionally filtered by type.

        Args:
            event_type: Optional event type to filter by (e.g., "textOutput").

        Returns:
            List of events, filtered if event_type specified.
        """
        if event_type:
            return [e for e in self.events if event_type in e]
        return self.events.copy()

    def get_text_outputs(self) -> list[str]:
        """Extract text outputs from collected events.

        Returns:
            List of text content strings.
        """
        texts = []
        for event in self.events:
            if "textOutput" in event:
                text = event["textOutput"].get("text", "")
                if text:
                    texts.append(text)
        return texts

    def get_audio_outputs(self) -> list[bytes]:
        """Extract audio outputs from collected events.

        Returns:
            List of audio data bytes.
        """
        audio_data = []
        for event in self.events:
            if "audioOutput" in event:
                data = event["audioOutput"].get("audioData")
                if data:
                    audio_data.append(data)
        return audio_data

    def get_tool_uses(self) -> list[dict]:
        """Extract tool use events from collected events.

        Returns:
            List of tool use events.
        """
        return [event["toolUse"] for event in self.events if "toolUse" in event]

    def has_interruption(self) -> bool:
        """Check if any interruption was detected.

        Returns:
            True if interruption detected in events.
        """
        return any("interruptionDetected" in event for event in self.events)

    def clear_events(self):
        """Clear collected events (useful for multi-turn tests)."""
        self.events.clear()
        logger.debug("Events cleared")

    # === Background threads ===

    async def _input_thread(self):
        """Continuously handle input to model.

        - Sends silence by default (background noise) if audio generator available
        - Converts queued text to audio via Polly (for "audio" type)
        - Sends text directly to model (for "text" type)
        """
        try:
            while self.active:
                try:
                    # Check for queued input (non-blocking)
                    input_item = await asyncio.wait_for(self.input_queue.get(), timeout=0.01)

                    if input_item["type"] == "audio":
                        # Generate and send audio
                        if self.audio_generator:
                            audio_data = await self.audio_generator.generate_audio(input_item["text"])

                            # Send audio in chunks
                            for i in range(0, len(audio_data), self.audio_chunk_size):
                                if not self.active:
                                    break
                                chunk = audio_data[i : i + self.audio_chunk_size]
                                chunk_event = self.audio_generator.create_audio_input_event(chunk)
                                await self.agent.send(chunk_event)
                                await asyncio.sleep(0.01)

                            logger.debug(f"Sent audio: {len(audio_data)} bytes")
                        else:
                            logger.warning("Audio requested but no generator available")

                    elif input_item["type"] == "direct":
                        # Send data directly to agent
                        await self.agent.send(input_item["data"])
                        data_repr = str(input_item["data"])[:50] if isinstance(input_item["data"], str) else type(input_item["data"]).__name__
                        logger.debug(f"Sent direct: {data_repr}")

                except asyncio.TimeoutError:
                    # No input queued - send silence if audio generator available
                    if self.audio_generator:
                        silence = self._generate_silence_chunk()
                        await self.agent.send(silence)
                        await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.debug("Input thread cancelled")
        except Exception as e:
            logger.error(f"Input thread error: {e}")

    async def _event_collection_thread(self):
        """Continuously collect events from model."""
        try:
            async for event in self.agent.receive():
                if not self.active:
                    break

                self.events.append(event)
                self.last_event_time = asyncio.get_event_loop().time()
                logger.debug(f"Event collected: {list(event.keys())}")

        except asyncio.CancelledError:
            logger.debug("Event collection thread cancelled")
        except Exception as e:
            logger.error(f"Event collection thread error: {e}")

    def _generate_silence_chunk(self) -> dict:
        """Generate silence chunk for background audio.

        Returns:
            AudioInputEvent with silence data.
        """
        silence = b"\x00" * self.silence_chunk_size
        return self.audio_generator.create_audio_input_event(silence)
