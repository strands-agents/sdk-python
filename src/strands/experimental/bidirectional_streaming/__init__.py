"""
Bidirectional streaming package.
"""

# Main components - Primary user interface
from .agent.agent import BidirectionalAgent

# Model providers - What users need to create models
from .models.novasonic import NovaSonicBidirectionalModel
from .models.openai import OpenAIRealtimeBidirectionalModel

# Event types - For type hints and event handling
from .types.bidirectional_streaming import (
    AudioInputEvent,
    AudioOutputEvent,
    TextOutputEvent,
    InterruptionDetectedEvent,
    BidirectionalStreamEvent,
    VoiceActivityEvent,
    UsageMetricsEvent,
)

# Advanced interfaces (for custom implementations)
from .models.bidirectional_model import BidirectionalModel, BidirectionalModelSession

__all__ = [
    # Main interface
    "BidirectionalAgent",
    
    # Model providers
    "NovaSonicBidirectionalModel",
    "OpenAIRealtimeBidirectionalModel",
    
    # Event types
    "AudioInputEvent",
    "AudioOutputEvent", 
    "TextOutputEvent",
    "InterruptionDetectedEvent",
    "BidirectionalStreamEvent",
    "VoiceActivityEvent",
    "UsageMetricsEvent",
    
    # Model interface
    "BidirectionalModel",
    "BidirectionalModelSession",
]