"""Protocol for bidirectional streaming IO channels.

Defines callable protocols for input and output channels that can be used
with BidiAgent. This approach provides better typing and flexibility
by separating input and output concerns into independent callables.
"""

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..types.events import BidiInputEvent, BidiOutputEvent

if TYPE_CHECKING:
    from ..agent.agent import BidiAgent

_MAX_STREAM_DELAY_MS = 1000
"""Upper bound for the stream delay hint; values beyond ~1s are meaningless for AEC alignment."""


@dataclass
class AudioProcessingConfig:
    """Configuration for microphone audio processing.

    Passing an instance of this config to ``BidiAudioIO`` enables microphone processing via WebRTC. Requires
    pywebrtc-audio (pip install strands-agents[bidi-aec]).

    Processing always applies noise suppression and automatic gain control to the microphone signal. Echo
    cancellation additionally removes the agent's own speaker audio from the mic input; it needs the speaker
    signal as a reference and therefore only works when the same ``BidiAudioIO`` produces both ``input()`` and
    ``output()``. Disable it for setups with no acoustic echo (for example a headset) while still benefiting
    from noise suppression and gain control.

    Attributes:
        echo_cancellation: Cancel the agent's own speaker audio from the mic input.
        stream_delay_ms: Playback-to-capture delay hint in ms for AEC. 0 lets AEC3 auto-estimate the delay.
            Advanced tuning knob; only set a non-zero value if echo cancellation is measurably failing on
            hardware with large or fixed playback-to-capture latency (e.g. Bluetooth).
    """

    echo_cancellation: bool = True
    stream_delay_ms: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If stream_delay_ms is out of the [0, 1000] range.
        """
        if not 0 <= self.stream_delay_ms <= _MAX_STREAM_DELAY_MS:
            raise ValueError(f"stream_delay_ms=<{self.stream_delay_ms}> | must be between 0 and {_MAX_STREAM_DELAY_MS}")


@runtime_checkable
class BidiInput(Protocol):
    """Protocol for bidirectional input callables.

    Input callables read data from a source (microphone, camera, websocket, etc.)
    and return events to be sent to the agent.
    """

    async def start(self, agent: "BidiAgent") -> None:
        """Start input."""
        return

    async def stop(self) -> None:
        """Stop input."""
        return

    def __call__(self) -> Awaitable[BidiInputEvent]:
        """Read input data from the source.

        Returns:
            Awaitable that resolves to an input event (audio, text, image, etc.)
        """
        ...


@runtime_checkable
class BidiOutput(Protocol):
    """Protocol for bidirectional output callables.

    Output callables receive events from the agent and handle them appropriately
    (play audio, display text, send over websocket, etc.).
    """

    async def start(self, agent: "BidiAgent") -> None:
        """Start output."""
        return

    async def stop(self) -> None:
        """Stop output."""
        return

    def __call__(self, event: BidiOutputEvent) -> Awaitable[None]:
        """Process output events from the agent.

        Args:
            event: Output event from the agent (audio, text, tool calls, etc.)
        """
        ...
