"""Bidirectional model interface for real-time streaming conversations.

Defines the interface for models that support bidirectional streaming capabilities.
Provides abstractions for different model providers with connection-based communication
patterns that support real-time audio and text interaction.

Features:
- connection-based persistent connections
- Real-time bidirectional communication
- Provider-agnostic event normalization
- Tool execution integration
"""

import abc
import logging
from typing import AsyncIterable

from ....types.content import Messages
from ....types.tools import ToolSpec
from ..types.bidirectional_streaming import AudioInputEvent, BidirectionalStreamEvent, ImageInputEvent

logger = logging.getLogger(__name__)


class BidirectionalModelSession(abc.ABC):
    """Abstract interface for model-specific bidirectional communication connections.

    Defines the contract for managing persistent streaming connections with individual
    model providers, handling audio/text input, receiving events, and managing
    tool execution results.
    """

    @abc.abstractmethod
    async def receive_events(self) -> AsyncIterable[BidirectionalStreamEvent]:
        """Receive events from the model in standardized format.

        Converts provider-specific events to a common format that can be
        processed uniformly by the event loop.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def send_audio_content(self, audio_input: AudioInputEvent) -> None:
        """Send audio content to the model during an active connection.

        Handles audio encoding and provider-specific formatting while presenting
        a simple AudioInputEvent interface.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def send_image_content(self, image_input: ImageInputEvent) -> None:
        """Send image content to the model during an active connection.
        
        Handles image encoding and provider-specific formatting while presenting
        a simple ImageInputEvent interface.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    async def send_text_content(self, text: str, **kwargs) -> None:
        """Send text content to the model during ongoing generation.

        Allows natural interruption and follow-up questions without requiring
        connection restart.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def send_interrupt(self) -> None:
        """Send interruption signal to stop generation immediately.

        Enables responsive conversational experiences where users can
        naturally interrupt during model responses.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def send_tool_result(self, tool_use_id: str, result: dict[str, any]) -> None:
        """Send tool execution result to the model.

        Formats and sends tool results according to the provider's specific protocol.
        Handles both successful results and error cases through the result dictionary.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the connection and cleanup resources."""
        raise NotImplementedError


class BidirectionalModel(abc.ABC):
    """Interface for models that support bidirectional streaming.

    Defines the contract for creating persistent streaming connections that support
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
        """Create a bidirectional connection with the model.

        Establishes a persistent connection for real-time communication while
        abstracting provider-specific initialization requirements.
        """
        raise NotImplementedError
