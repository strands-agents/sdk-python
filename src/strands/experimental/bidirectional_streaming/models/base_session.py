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
from typing import AsyncIterable, Union

from ..types.bidirectional_streaming import (
    AudioInputEvent,
    BidirectionalStreamEvent,
    ImageInputEvent,
    ToolResultInputEvent,
)

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
    async def send_events(self, content: Union[str, ImageInputEvent, AudioInputEvent, ToolResultInputEvent]) -> None:
        """Send structured content (text, images,audio tool results) to the model.

        Args:
            content: Text string, ImageInputEvent, AudioInputEvent or ToolResultInputEvent
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
    async def close(self) -> None:
        """Close the connection and cleanup resources."""
        raise NotImplementedError
