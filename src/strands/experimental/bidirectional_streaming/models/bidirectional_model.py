"""Unified bidirectional streaming interface.

Single layer combining model and session abstractions for simpler implementation.
Providers implement this directly without separate model/session classes.

Features:
- Unified model interface (no separate session class)
- Real-time bidirectional communication
- Provider-agnostic event normalization
- Tool execution integration
"""

import abc
import logging
from typing import AsyncIterable, Union

from ....types.content import Messages
from ....types.tools import ToolResult, ToolSpec
from ..types.bidirectional_streaming import (
    AudioInputEvent,
    BidirectionalStreamEvent,
    ImageInputEvent,
    TextInputEvent,
)

logger = logging.getLogger(__name__)


class BidirectionalModel(abc.ABC):
    """Unified interface for bidirectional streaming models.

    Combines model configuration and session communication in a single abstraction.
    Providers implement this directly without separate model/session classes.
    """

    @abc.abstractmethod
    async def connect(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
        **kwargs,
    ) -> None:
        """Establish bidirectional connection with the model.

        Initializes the connection state and prepares for real-time communication.
        This replaces the old create_bidirectional_connection pattern.

        Args:
            system_prompt: System instructions for the model.
            tools: List of tools available to the model.
            messages: Conversation history to initialize with.
            **kwargs: Provider-specific configuration options.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self) -> None:
        """Close connection and cleanup resources.

        Terminates the active connection and releases any held resources.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def receive(self) -> AsyncIterable[BidirectionalStreamEvent]:
        """Receive events from the model in standardized format.

        Yields provider-agnostic events that can be processed uniformly
        by the event loop. Converts provider-specific events to common format.

        Yields:
            BidirectionalStreamEvent: Standardized event dictionaries.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def send(self, content: Union[TextInputEvent, ImageInputEvent, AudioInputEvent, ToolResult]) -> None:
        """Send structured content to the model.

        Unified method for sending all types of content. Implementations should
        dispatch to appropriate internal handlers based on content type.

        Args:
            content: Typed event (TextInputEvent, ImageInputEvent, AudioInputEvent, or ToolResult).

        Example:
            await model.send(TextInputEvent(text="Hello", role="user"))
            await model.send(AudioInputEvent(audioData=bytes, format="pcm", ...))
            await model.send(ToolResult(toolUseId="123", status="success", ...))
        """
        raise NotImplementedError


# Backwards compatibility alias - will be removed in future version
BidirectionalModelSession = BidirectionalModel
