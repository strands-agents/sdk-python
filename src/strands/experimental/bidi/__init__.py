"""Bidirectional streaming package."""

import sys
from typing import TYPE_CHECKING, Any

if sys.version_info < (3, 12):
    raise ImportError("bidi only supported for >= Python 3.12")

# Re-export standard agent events for tool handling (these are safe to import eagerly)
from ...types._events import (
    ToolResultEvent,
    ToolStreamEvent,
    ToolUseStreamEvent,
)

# Built-in tools
from .tools import stop_conversation

# Event types - For type hints and event handling (these are safe to import eagerly)
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

# Type checking imports - not executed at runtime
if TYPE_CHECKING:
    from .agent.agent import BidiAgent
    from .io.audio import BidiAudioIO
    from .models.model import BidiModel
    from .models.nova_sonic import BidiNovaSonicModel


def __getattr__(name: str) -> Any:
    """Lazy import classes to avoid requiring optional dependencies."""
    if name == "BidiAgent":
        try:
            from .agent.agent import BidiAgent

            return BidiAgent
        except ImportError as e:
            raise ImportError(
                "BidiAgent requires aws_sdk_bedrock_runtime. Install it with: pip install strands-agents[bidi]"
            ) from e
    elif name == "BidiAudioIO":
        from .io import BidiAudioIO

        return BidiAudioIO
    elif name == "BidiModel":
        try:
            from .models.model import BidiModel

            return BidiModel
        except ImportError as e:
            raise ImportError(
                "BidiModel requires aws_sdk_bedrock_runtime. Install it with: pip install strands-agents[bidi]"
            ) from e
    elif name == "BidiNovaSonicModel":
        try:
            from .models.nova_sonic import BidiNovaSonicModel

            return BidiNovaSonicModel
        except ImportError as e:
            raise ImportError(
                "BidiNovaSonicModel requires aws_sdk_bedrock_runtime. Install it with: pip install strands-agents[bidi]"
            ) from e
    elif name == "stop_conversation":
        try:
            from .tools import stop_conversation

            return stop_conversation
        except ImportError as e:
            raise ImportError(
                "stop_conversation requires aws_sdk_bedrock_runtime. Install it with: pip install strands-agents[bidi]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Main interface
    "BidiAgent",
    # IO channels
    "BidiAudioIO",
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
