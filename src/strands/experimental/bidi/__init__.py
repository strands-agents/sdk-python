"""Bidirectional streaming package."""

# Main components - Primary user interface
from .agent.agent import BidiAgent

# IO channels - Hardware abstraction
from .io.audio import BidiAudioIO

# Model interface (for custom implementations)
from .models.bidi_model import BidiModel

# Model providers - What users need to create models
from .models.gemini_live import BidiGeminiLiveModel
from .models.novasonic import BidiNovaSonicModel
from .models.openai import BidiOpenAIRealtimeModel

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
    ModalityUsage,
    BidiUsageEvent,
    BidiOutputEvent,
    BidiResponseCompleteEvent,
    BidiResponseStartEvent,
    BidiTextInputEvent,
    BidiTranscriptStreamEvent,
)

# Re-export standard agent events for tool handling
from ...types._events import (
    ToolResultEvent,
    ToolStreamEvent,
    ToolUseStreamEvent,
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
