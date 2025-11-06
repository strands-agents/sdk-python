"""Bidirectional streaming package."""

# Main components - Primary user interface
from .agent.agent import BidirectionalAgent

# Model interface (for custom implementations)
from .models.bidirectional_model import BidirectionalModel

# Model providers - What users need to create models
from .models.gemini_live import GeminiLiveModel
from .models.novasonic import NovaSonicModel
from .models.openai import OpenAIRealtimeModel

# Event types - For type hints and event handling
from .types.bidirectional_streaming import (
    AudioInputEvent,
    AudioStreamEvent,
    ConnectionCloseEvent,
    ConnectionStartEvent,
    ErrorEvent,
    ImageInputEvent,
    InputEvent,
    InterruptionEvent,
    ModalityUsage,
    UsageEvent,
    OutputEvent,
    ResponseCompleteEvent,
    ResponseStartEvent,
    TextInputEvent,
    TranscriptStreamEvent,
)

# Re-export standard agent events for tool handling
from ...types._events import (
    ToolResultEvent,
    ToolStreamEvent,
    ToolUseStreamEvent,
)

__all__ = [
    # Main interface
    "BidirectionalAgent",
    
    # Model providers
    "GeminiLiveModel",
    "NovaSonicModel",
    "OpenAIRealtimeModel",
    
    # Input Event types
    "TextInputEvent",
    "AudioInputEvent",
    "ImageInputEvent",
    "InputEvent",
    
    # Output Event types
    "ConnectionStartEvent",
    "ConnectionCloseEvent",
    "ResponseStartEvent",
    "ResponseCompleteEvent",
    "AudioStreamEvent",
    "TranscriptStreamEvent",
    "InterruptionEvent",
    "UsageEvent",
    "ModalityUsage",
    "ErrorEvent",
    "OutputEvent",
    
    # Tool Event types (reused from standard agent)
    "ToolUseStreamEvent",
    "ToolResultEvent",
    "ToolStreamEvent",
    
    # Model interface
    "BidirectionalModel",
]
