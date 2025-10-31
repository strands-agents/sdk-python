"""Bidirectional streaming package."""

# Main components - Primary user interface
from .agent.agent import BidirectionalAgent

# Model interface (for custom implementations)
from .models.bidirectional_model import BidirectionalModel

# Model providers - What users need to create models
from .models.gemini_live import GeminiLiveBidirectionalModel
from .models.novasonic import NovaSonicBidirectionalModel
from .models.openai import OpenAIRealtimeBidirectionalModel

# Event types - For type hints and event handling
from .types.bidirectional_streaming import (
    AudioInputEvent,
    AudioStreamEvent,
    ErrorEvent,
    ImageInputEvent,
    InputEvent,
    InterruptionEvent,
    ModalityUsage,
    MultimodalUsage,
    OutputEvent,
    SessionEndEvent,
    SessionStartEvent,
    TextInputEvent,
    TranscriptStreamEvent,
    TurnCompleteEvent,
    TurnStartEvent,
)

__all__ = [
    # Main interface
    "BidirectionalAgent",
    # Model providers
    "GeminiLiveBidirectionalModel",
    "NovaSonicBidirectionalModel",
    "OpenAIRealtimeBidirectionalModel",
    # Input Event types
    "TextInputEvent",
    "AudioInputEvent",
    "ImageInputEvent",
    "InputEvent",
    # Output Event types
    "SessionStartEvent",
    "TurnStartEvent",
    "AudioStreamEvent",
    "TranscriptStreamEvent",
    "InterruptionEvent",
    "TurnCompleteEvent",
    "MultimodalUsage",
    "ModalityUsage",
    "SessionEndEvent",
    "ErrorEvent",
    "OutputEvent",
    # Model interface
    "BidirectionalModel",
]
