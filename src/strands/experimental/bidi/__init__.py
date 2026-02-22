"""Bidirectional streaming package."""

from typing import Any

# Main components - Primary user interface
# Re-export standard agent events for tool handling
from ...types._events import (
    ToolResultEvent,
    ToolStreamEvent,
    ToolUseStreamEvent,
)
from .agent.agent import BidiAgent

# Model interface (for custom implementations)
from .models.model import BidiModel

# Built-in tools
from .tools import stop_conversation

# Event types - For type hints and event handling
from .types.events import (
    BidiAudioInputEvent,
    BidiAudioStreamEvent,
    BidiConnectionCloseEvent,
    BidiConnectionStartEvent,
    BidiErrorEvent,
    BidiImageInputEvent,
    BidiInputEvent,
    BidiInterruptionEvent,
    BidiOutputEvent,
    BidiResponseCompleteEvent,
    BidiResponseStartEvent,
    BidiTextInputEvent,
    BidiTranscriptStreamEvent,
    BidiUsageEvent,
    ModalityUsage,
)

__all__ = [
    # Main interface
    "BidiAgent",
    # Built-in tools
    "stop_conversation",
    # Input Event types
    "BidiTextInputEvent",
    "BidiAudioInputEvent",
    "BidiImageInputEvent",
    "BidiInputEvent",
    # Output Event types
    "BidiConnectionStartEvent",
    "BidiConnectionCloseEvent",
    "BidiResponseStartEvent",
    "BidiResponseCompleteEvent",
    "BidiAudioStreamEvent",
    "BidiTranscriptStreamEvent",
    "BidiInterruptionEvent",
    "BidiUsageEvent",
    "ModalityUsage",
    "BidiErrorEvent",
    "BidiOutputEvent",
    # Tool Event types (reused from standard agent)
    "ToolUseStreamEvent",
    "ToolResultEvent",
    "ToolStreamEvent",
    # Model interface
    "BidiModel",
]


def __getattr__(name: str) -> Any:
    """Lazy load IO implementations only when accessed.

    This defers the import of optional dependencies until actually needed.
    """
    if name == "BidiAudioIO":
        from .io.audio import BidiAudioIO

        return BidiAudioIO
    if name == "BidiTextIO":
        from .io.text import BidiTextIO

        return BidiTextIO
    raise AttributeError(f"cannot import name '{name}' from '{__name__}' ({__file__})")
