"""Bidirectional streaming package."""

import sys
from typing import Any

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

from .agent.agent import BidiAgent
from .models.model import BidiModel
from .models.nova_sonic import BidiNovaSonicModel
from .tools import stop_conversation
from .io import BidiAudioIO


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
