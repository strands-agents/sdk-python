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
    """Unified interface for bidirectional streaming models.

    Combines model configuration and session communication in a single abstraction.
    Providers implement this directly without separate model/session classes.
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