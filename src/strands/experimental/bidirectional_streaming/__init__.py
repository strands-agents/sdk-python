"""Bidirectional streaming package."""

# Main components - Primary user interface
from .agent.agent import BidiAgent

# IO channels - Hardware abstraction
from .io.audio import AudioIO

# Model interface (for custom implementations)
from .models.bidirectional_model import BidiModel

# Model providers - What users need to create models
from .models.gemini_live import BidiGeminiLiveModel
from .models.novasonic import BidiNovaSonicModel
from .models.openai import BidiOpenAIRealtimeModel

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
    "BidiAgent",
    # IO channels
    "AudioIO",
    # Model providers
    "BidiGeminiLiveModel",
    "BidiNovaSonicModel",
    "BidiOpenAIRealtimeModel",
    
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
    "BidiModel",
]
