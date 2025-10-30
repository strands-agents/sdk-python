"""Bidirectional streaming package."""

# Main components - Primary user interface
from .agent.agent import BidirectionalAgent

# Unified model interface (for custom implementations)
from .models.bidirectional_model import BidirectionalModel, BidirectionalModelSession

# Model providers - What users need to create models
from .models.gemini_live import GeminiLiveBidirectionalModel
from .models.novasonic import NovaSonicBidirectionalModel
from .models.openai import OpenAIRealtimeBidirectionalModel

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
    
    # Model providers
    "GeminiLiveBidirectionalModel",
    "NovaSonicBidirectionalModel",
    "OpenAIRealtimeBidirectionalModel",
    
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
    
    # Unified model interface
    "BidirectionalModel",
    "BidirectionalModelSession",  # Backwards compatibility alias
]
