"""Bidirectional streaming package."""

import sys

if sys.version_info < (3, 12):
    raise ImportError("bidi only supported for >= Python 3.12")

# Main components - Primary user interface
# Re-export standard agent events for tool handling
from ...types._events import (
    ToolResultEvent,
    ToolStreamEvent,
    ToolUseStreamEvent,
)
from .agent.agent import BidiAgent

# IO channels - Hardware abstraction
from .io.audio import BidiAudioIO

# Model providers - What users need to create models
from .models.gemini_live import BidiGeminiLiveModel

# Model interface (for custom implementations)
from .models.model import BidiModel
from .models.nova_sonic import BidiNovaSonicModel
from .models.openai_realtime import BidiOpenAIRealtimeModel

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
    # IO channels
    "BidiAudioIO",
    # Model providers
    "BidiGeminiLiveModel",
    "BidiNovaSonicModel",
    "BidiOpenAIRealtimeModel",
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
