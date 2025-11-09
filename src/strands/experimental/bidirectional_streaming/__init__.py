"""Bidirectional streaming package."""

# Main components - Primary user interface
from .agent.agent import BidirectionalAgent

# IO channels - Hardware abstraction
from .io.audio import AudioIO

# Model interface (for custom implementations)
from .models.bidirectional_model import BidirectionalModel

# Model providers - What users need to create models
from .models.gemini_live import GeminiLiveModel
from .models.novasonic import NovaSonicModel
from .models.openai import OpenAIRealtimeModel

# Event types - For type hints and event handling
from .types.bidirectional_streaming import (
    AudioInputEvent,
    AudioOutputEvent,
    BidirectionalStreamEvent,
    ImageInputEvent,
    InterruptionDetectedEvent,
    TextInputEvent,
    TextOutputEvent,
    UsageMetricsEvent,
    VoiceActivityEvent,
)

__all__ = [
    # Main interface
    "BidirectionalAgent",
    # IO channels
    "AudioIO",
    # Model providers
    "GeminiLiveModel",
    "NovaSonicModel",
    "OpenAIRealtimeModel",
    
    # Event types
    "AudioInputEvent",
    "AudioOutputEvent",
    "ImageInputEvent",
    "TextInputEvent",
    "TextOutputEvent",
    "InterruptionDetectedEvent",
    "BidirectionalStreamEvent",
    "VoiceActivityEvent",
    "UsageMetricsEvent",
    # Model interface
    "BidirectionalModel",
]
