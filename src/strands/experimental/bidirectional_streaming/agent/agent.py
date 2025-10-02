"""Bidirectional Agent for real-time streaming conversations.

Provides real-time audio and text interaction through persistent streaming sessions.
Unlike traditional request-response patterns, this agent maintains long-running
conversations where users can interrupt, provide additional input, and receive
continuous responses including audio output.

Key capabilities:
- Persistent conversation sessions with concurrent processing
- Real-time audio input/output streaming
- Mid-conversation interruption and tool execution
- Event-driven communication with model providers
"""

import asyncio
import logging
from typing import AsyncIterable

from ....tools.executors import ConcurrentToolExecutor
from ....tools.registry import ToolRegistry
from ....types.content import Messages
from ..event_loop.bidirectional_event_loop import start_bidirectional_connection, stop_bidirectional_connection
from ..models.bidirectional_model import BidirectionalModel
from ..types.bidirectional_streaming import AudioInputEvent, BidirectionalStreamEvent


logger = logging.getLogger(__name__)


class BidirectionalAgent:
    """Agent for bidirectional streaming conversations.

    Enables real-time audio and text interaction with AI models through persistent
    sessions. Supports concurrent tool execution and interruption handling.
    """

    def __init__(
        self,
        model: BidirectionalModel,
        tools: list | None = None,
        system_prompt: str | None = None,
        messages: Messages | None = None,
    ):
        """Initialize bidirectional agent with required model and optional configuration.

        Args:
            model: BidirectionalModel instance supporting streaming sessions.
            tools: Optional list of tools available to the model.
            system_prompt: Optional system prompt for conversations.
            messages: Optional conversation history to initialize with.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.messages = messages or []

        # Initialize tool registry using existing Strands infrastructure
        self.tool_registry = ToolRegistry()
        if tools:
            self.tool_registry.process_tools(tools)
        self.tool_registry.initialize_tools()

        # Initialize tool executor for concurrent execution
        self.tool_executor = ConcurrentToolExecutor()

        # Session management
        self._session = None
        self._output_queue = asyncio.Queue()

    async def start(self) -> None:
        """Start a persistent bidirectional conversation session.

        Initializes the streaming session and starts background tasks for processing
        model events, tool execution, and session management.

        Raises:
            ValueError: If conversation already active.
            ConnectionError: If session creation fails.
        """
        if self._session and self._session.active:
            raise ValueError("Conversation already active. Call end() first.")

        logger.debug("Conversation start - initializing session")
        self._session = await start_bidirectional_connection(self)
        logger.debug("Conversation ready")

    async def send(self, input_data: str | AudioInputEvent) -> None:
        """Send input to the model (text or audio).

        Unified method for sending both text and audio input to the model during
        an active conversation session. User input is automatically added to
        conversation history for complete message tracking.

        Args:
            input_data: Either a string for text input or AudioInputEvent for audio input.

        Raises:
            ValueError: If no active session or invalid input type.
        """
        self._validate_active_session()

        if isinstance(input_data, str):
            # Add user text message to history
            self.messages.append({"role": "user", "content": input_data})

            logger.debug("Text sent: %d characters", len(input_data))
            await self._session.model_session.send_text_content(input_data)
        elif isinstance(input_data, dict) and "audioData" in input_data:
            # Handle audio input
            await self._session.model_session.send_audio_content(input_data)
        else:
            raise ValueError(
                "Input must be either a string (text) or AudioInputEvent "
                "(dict with audioData, format, sampleRate, channels)"
            )

    async def receive(self) -> AsyncIterable[BidirectionalStreamEvent]:
        """Receive events from the model including audio, text, and tool calls.

        Yields model output events processed by background tasks including audio output,
        text responses, tool calls, and session updates.

        Yields:
            BidirectionalStreamEvent: Events from the model session.
        """
        while self._session and self._session.active:
            try:
                event = await asyncio.wait_for(self._output_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue

    async def interrupt(self) -> None:
        """Interrupt the current model generation and clear audio buffers.

        Sends interruption signal to stop generation immediately and clears
        pending audio output for responsive conversation flow.

        Raises:
            ValueError: If no active session.
        """
        self._validate_active_session()
        await self._session.model_session.send_interrupt()

    async def end(self) -> None:
        """End the conversation session and cleanup all resources.

        Terminates the streaming session, cancels background tasks, and
        closes the connection to the model provider.
        """
        if self._session:
            await stop_bidirectional_connection(self._session)
            self._session = None

    def _validate_active_session(self) -> None:
        """Validate that an active session exists.

        Raises:
            ValueError: If no active session.
        """
        if not self._session or not self._session.active:
            raise ValueError("No active conversation. Call start() first.")
