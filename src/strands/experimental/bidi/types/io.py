"""Protocol for bidirectional streaming IO channels.

Defines callable protocols for input and output channels that can be used
with BidiAgent. This approach provides better typing and flexibility
by separating input and output concerns into independent callables.
"""

from typing import TYPE_CHECKING, Awaitable, Literal, Protocol, TypedDict

from ..types.events import BidiInputEvent, BidiOutputEvent

if TYPE_CHECKING:
    from ..agent.agent import BidiAgent


class AudioConfig(TypedDict, total=False):
    """Audio configuration for bidirectional streaming.

    Defines standard audio parameters shared between model providers
    and audio I/O implementations. All fields are optional to support
    models that may not use audio or only need specific parameters.

    Attributes:
        input_rate: Input sample rate in Hz (e.g., 16000, 24000, 48000)
        output_rate: Output sample rate in Hz (e.g., 16000, 24000, 48000)
        channels: Number of audio channels (1=mono, 2=stereo)
        format: Audio encoding format
        voice: Voice identifier for text-to-speech (e.g., "alloy", "matthew")
    """

    input_rate: int
    output_rate: int
    channels: int
    format: Literal["pcm", "wav", "opus", "mp3"]
    voice: str


class BidiInput(Protocol):
    """Protocol for bidirectional input callables.

    Input callables read data from a source (microphone, camera, websocket, etc.)
    and return events to be sent to the agent.
    """

    async def start(self, agent: "BidiAgent") -> None:
        """Start input.
        
        Args:
            agent: The BidiAgent instance, providing access to model configuration.
        """
        ...

    async def stop(self) -> None:
        """Stop input."""
        ...

    def __call__(self) -> Awaitable[BidiInputEvent]:
        """Read input data from the source.

        Returns:
            Awaitable that resolves to an input event (audio, text, image, etc.)
        """
        ...


class BidiOutput(Protocol):
    """Protocol for bidirectional output callables.

    Output callables receive events from the agent and handle them appropriately
    (play audio, display text, send over websocket, etc.).
    """

    async def start(self, agent: "BidiAgent") -> None:
        """Start output.
        
        Args:
            agent: The BidiAgent instance, providing access to model configuration.
        """
        ...

    async def stop(self) -> None:
        """Stop output."""
        ...

    def __call__(self, event: BidiOutputEvent) -> Awaitable[None]:
        """Process output events from the agent.

        Args:
            event: Output event from the agent (audio, text, tool calls, etc.)
        """
        ...
