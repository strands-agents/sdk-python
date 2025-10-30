"""Bidirectional streaming model interface.

Defines the abstract interface for models that support real-time bidirectional
communication with persistent connections. Unlike traditional request-response
models, bidirectional models maintain an open connection for streaming audio,
text, and tool interactions.

Features:
- Persistent connection management with connect/close lifecycle
- Real-time bidirectional communication (send and receive simultaneously)
- Provider-agnostic event normalization
- Support for audio, text, image, and tool result streaming
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
    """Abstract base class for bidirectional streaming models.

    This interface defines the contract for models that support persistent streaming
    connections with real-time audio and text communication. Implementations handle
    provider-specific protocols while exposing a standardized event-based API.
    """

    @abc.abstractmethod
    async def connect(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
        **kwargs,
    ) -> None:
        """Establish a persistent streaming connection with the model.

        Opens a bidirectional connection that remains active for real-time communication.
        The connection supports concurrent sending and receiving of events until explicitly
        closed. Must be called before any send() or receive() operations.

        Args:
            system_prompt: System instructions to configure model behavior.
            tools: Tool specifications that the model can invoke during the conversation.
            messages: Initial conversation history to provide context.
            **kwargs: Provider-specific configuration options.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def close(self) -> None:
        """Close the streaming connection and release resources.

        Terminates the active bidirectional connection and cleans up any associated
        resources such as network connections, buffers, or background tasks. After
        calling close(), the model instance cannot be used until connect() is called again.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def receive(self) -> AsyncIterable[BidirectionalStreamEvent]:
        """Receive streaming events from the model.

        Continuously yields events from the model as they arrive over the connection.
        Events are normalized to a provider-agnostic format for uniform processing.
        This method should be called in a loop or async task to process model responses.

        The stream continues until the connection is closed or an error occurs.

        Yields:
            BidirectionalStreamEvent: Standardized event dictionaries containing
                audio output, text responses, tool calls, or control signals.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def send(self, content: Union[TextInputEvent, ImageInputEvent, AudioInputEvent, ToolResult]) -> None:
        """Send content to the model over the active connection.

        Transmits user input or tool results to the model during an active streaming
        session. Supports multiple content types including text, audio, images, and
        tool execution results. Can be called multiple times during a conversation.

        Args:
            content: The content to send. Must be one of:
                - TextInputEvent: Text message from the user
                - ImageInputEvent: Image data for visual understanding
                - AudioInputEvent: Audio data for speech input
                - ToolResult: Result from a tool execution

        Example:
            await model.send(TextInputEvent(text="Hello", role="user"))
            await model.send(AudioInputEvent(audioData=bytes, format="pcm", ...))
            await model.send(ToolResult(toolUseId="123", status="success", ...))
        """
        raise NotImplementedError


# Backwards compatibility alias - will be removed in future version
BidirectionalModelSession = BidirectionalModel
