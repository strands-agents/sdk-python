"""Protocol interface for real-time bidirectional streaming AI models.

This module defines the BaseModel protocol that standardizes how AI models handle
real-time, two-way communication with audio, text, images, and tool interactions.
It abstracts provider-specific implementations (Gemini Live, Nova Sonic, OpenAI Realtime)
into a unified interface for seamless integration.

The protocol enables:
- Persistent streaming connections with automatic reconnection
- Real-time audio input/output with interruption support
- Multi-modal content (text, audio, images) in both directions
- Function calling and tool execution during conversations
- Standardized event formats across different AI providers
- Async/await patterns for non-blocking operations
"""

from typing import AsyncIterable, Protocol, Union

from ....types.content import Messages
from ....types.tools import ToolResult, ToolSpec
from ..types.bidirectional_streaming import (
    AudioInputEvent,
    BidirectionalStreamEvent,
    ImageInputEvent,
    TextInputEvent,
)


class BaseModel(Protocol):
    """Protocol defining the interface for real-time bidirectional AI models.

    This protocol standardizes how AI models handle persistent streaming connections
    for real-time conversations with audio, text, images, and tool interactions.
    Implementations handle provider-specific connection management, event processing,
    and content serialization while exposing a consistent async interface.

    Models implementing this protocol support:
    - WebSocket or streaming API connections
    - Real-time audio input/output with voice activity detection
    - Multi-modal content streaming (text, audio, images)
    - Function calling and tool execution
    - Interruption handling and conversation state management
    """

    async def connect(
            self,
            system_prompt: str | None = None,
            tools: list[ToolSpec] | None = None,
            messages: Messages | None = None,
            **kwargs,
    ) -> None:
        """Establish bidirectional connection with the model."""
        ...

    async def close(self) -> None:
        """Close connection and cleanup resources."""
        ...

    async def receive(self) -> AsyncIterable[BidirectionalStreamEvent]:
        """Receive events from the model in standardized format."""
        ...

    async def send(self, content: Union[TextInputEvent, ImageInputEvent, AudioInputEvent, ToolResult]) -> None:
        """Send structured content to the model."""
        ...