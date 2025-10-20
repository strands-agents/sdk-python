"""Bidirectional model interface for real-time streaming conversations.

Defines the interface for models that support bidirectional streaming capabilities.
Provides abstractions for different model providers with connection-based communication
patterns that support real-time audio and text interaction.

Features:
- Session-based persistent connections
- Real-time bidirectional communication
- Provider-agnostic event normalization
- Tool execution integration
"""

import abc
import logging
from typing import AsyncIterable

from ....types.content import Messages
from ....types.tools import ToolSpec
from ..types.bidirectional_streaming import InputEvent, OutputEvent

logger = logging.getLogger(__name__)


class BidirectionalModelSession(abc.ABC):
    """Abstract interface for model-specific bidirectional communication sessions.

    Defines the contract for managing persistent streaming connections with individual
    model providers, handling audio/image input, receiving events, and managing
    tool execution results.
    """

    @abc.abstractmethod
    async def receive_events(self) -> AsyncIterable[OutputEvent]:
        """Receive events from the model in standardized format.

        Converts provider-specific events to a common TypedEvent format that can be
        processed uniformly by the event loop.

        Yields:
            OutputEvent: One of SessionStartEvent, TurnStartEvent, AudioStreamEvent,
                TranscriptStreamEvent, ToolUseStreamEvent, InterruptionEvent, TurnCompleteEvent,
                MultimodalUsage, SessionEndEvent, or ErrorEvent.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def send(self, event: InputEvent) -> None:
        """Send input event to the model using unified interface.

        This is the only method for sending input to the model. Accepts AudioInputEvent,
        ImageInputEvent, or ToolResultEvent and dispatches to the appropriate
        provider-specific implementation.

        Args:
            event: Input event to send. Must be one of:
                - AudioInputEvent: Send audio data to the model
                - ImageInputEvent: Send image data to the model
                - ToolResultEvent: Send tool execution result to the model

        Raises:
            NotImplementedError: If the provider doesn't support the event type.

        Example:
            >>> # Send audio
            >>> audio_event = AudioInputEvent(
            ...     audio=audio_bytes,
            ...     format="pcm",
            ...     sample_rate=16000,
            ...     channels=1
            ... )
            >>> await session.send(audio_event)
            >>>
            >>> # Send tool result
            >>> tool_event = ToolResultEvent(
            ...     tool_use_id="toolu_123",
            ...     result={"temperature": 72}
            ... )
            >>> await session.send(tool_event)
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the session and cleanup resources."""
        raise NotImplementedError


class BidirectionalModel(abc.ABC):
    """Interface for models that support bidirectional streaming.

    Defines the contract for creating persistent streaming sessions that support
    real-time audio and text communication with AI models.
    """

    @abc.abstractmethod
    async def create_bidirectional_connection(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
        **kwargs,
    ) -> BidirectionalModelSession:
        """Create a bidirectional session with the model.

        Establishes a persistent connection for real-time communication while
        abstracting provider-specific initialization requirements.

        Args:
            system_prompt: Optional system prompt to set context.
            tools: Optional list of tools available to the model.
            messages: Optional conversation history to initialize with.
            **kwargs: Provider-specific configuration options.

        Returns:
            BidirectionalModelSession: Active session ready for bidirectional communication.
        """
        raise NotImplementedError
