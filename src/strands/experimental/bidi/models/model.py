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

import logging
from typing import Any, AsyncIterable, Protocol, runtime_checkable

from ....types._events import ToolResultEvent
from ....types.content import Messages
from ....types.tools import ToolSpec
from ..types.events import (
    BidiInputEvent,
    BidiOutputEvent,
)

logger = logging.getLogger(__name__)

# Nova Sonic model identifiers
NOVA_SONIC_V1_MODEL_ID = "amazon.nova-sonic-v1:0"
NOVA_SONIC_V2_MODEL_ID = "amazon.nova-2-sonic-v1:0"


@runtime_checkable
class BidiModel(Protocol):
    """Protocol for bidirectional streaming models.

    This interface defines the contract for models that support persistent streaming
    connections with real-time audio and text communication. Implementations handle
    provider-specific protocols while exposing a standardized event-based API.

    Attributes:
        config: Configuration dictionary with provider-specific settings.
    """

    config: dict[str, Any]

    async def start(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
        **kwargs: Any,
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
        ...

    async def stop(self) -> None:
        """Close the streaming connection and release resources.

        Terminates the active bidirectional connection and cleans up any associated
        resources such as network connections, buffers, or background tasks. After
        calling close(), the model instance cannot be used until start() is called again.
        """
        ...

    def receive(self) -> AsyncIterable[BidiOutputEvent]:
        """Receive streaming events from the model.

        Continuously yields events from the model as they arrive over the connection.
        Events are normalized to a provider-agnostic format for uniform processing.
        This method should be called in a loop or async task to process model responses.

        The stream continues until the connection is closed or an error occurs.

        Yields:
            BidiOutputEvent: Standardized event objects containing audio output,
                transcripts, tool calls, or control signals.
        """
        ...

    async def send(
        self,
        content: BidiInputEvent | ToolResultEvent,
    ) -> None:
        """Send content to the model over the active connection.

        Transmits user input or tool results to the model during an active streaming
        session. Supports multiple content types including text, audio, images, and
        tool execution results. Can be called multiple times during a conversation.

        Args:
            content: The content to send. Must be one of:

                - BidiTextInputEvent: Text message from the user
                - BidiAudioInputEvent: Audio data for speech input
                - BidiImageInputEvent: Image data for visual understanding
                - ToolResultEvent: Result from a tool execution

        Example:
            ```
            await model.send(BidiTextInputEvent(text="Hello", role="user"))
            await model.send(BidiAudioInputEvent(audio=bytes, format="pcm", sample_rate=16000, channels=1))
            await model.send(BidiImageInputEvent(image=bytes, mime_type="image/jpeg", encoding="raw"))
            await model.send(ToolResultEvent(tool_result))
            ```
        """
        ...


class BidiModelTimeoutError(Exception):
    """Model timeout error.

    Bidirectional models are often configured with a connection time limit. Nova sonic for example keeps the connection
    open for 8 minutes max. Upon receiving a timeout, the agent loop is configured to restart the model connection so as
    to create a seamless, uninterrupted experience for the user.
    """

    def __init__(self, message: str, **restart_config: Any) -> None:
        """Initialize error.

        Args:
            message: Timeout message from model.
            **restart_config: Configure restart specific behaviors in the call to model start.
        """
        super().__init__(self, message)

        self.restart_config = restart_config


def create_nova_sonic_v1(
    provider_config: dict[str, Any] | None = None,
    client_config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> "BidiModel":
    """Create a Nova Sonic v1 bidirectional model instance.

    Convenience function to create a BidiNovaSonicModel configured for Nova Sonic v1.

    Args:
        provider_config: Model behavior configuration (audio, inference settings).
        client_config: AWS authentication configuration (boto_session OR region, not both).
        **kwargs: Additional configuration options.

    Returns:
        BidiNovaSonicModel instance configured for Nova Sonic v1.

    Example:
        ```python
        model = create_nova_sonic_v1()
        # or with custom config
        model = create_nova_sonic_v1(
            provider_config={"audio": {"voice": "joanna"}},
            client_config={"region": "us-west-2"}
        )
        ```
    """
    from .nova_sonic import BidiNovaSonicModel

    return BidiNovaSonicModel(
        model_id=NOVA_SONIC_V1_MODEL_ID,
        provider_config=provider_config,
        client_config=client_config,
        **kwargs,
    )


def create_nova_sonic_v2(
    provider_config: dict[str, Any] | None = None,
    client_config: dict[str, Any] | None = None,
    **kwargs: Any,
) -> "BidiModel":
    """Create a Nova Sonic v2 bidirectional model instance.

    Convenience function to create a BidiNovaSonicModel configured for Nova Sonic v2.

    Args:
        provider_config: Model behavior configuration (audio, inference settings).
        client_config: AWS authentication configuration (boto_session OR region, not both).
        **kwargs: Additional configuration options.

    Returns:
        BidiNovaSonicModel instance configured for Nova Sonic v2.

    Example:
        ```python
        model = create_nova_sonic_v2()
        # or with custom config
        model = create_nova_sonic_v2(
            provider_config={"audio": {"voice": "joanna"}},
            client_config={"region": "us-west-2"}
        )
        ```
    """
    from .nova_sonic import BidiNovaSonicModel

    return BidiNovaSonicModel(
        model_id=NOVA_SONIC_V2_MODEL_ID,
        provider_config=provider_config,
        client_config=client_config,
        **kwargs,
    )
