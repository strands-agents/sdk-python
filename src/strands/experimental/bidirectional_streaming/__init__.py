"""Bidirectional streaming package."""

# Main components - Primary user interface
from .agent.agent import BidirectionalAgent
from .models.base_model import BidirectionalModel
from .models.base_session import BidirectionalModelSession

# Model providers - What users need to create models
from .models.novasonic import NovaSonicBidirectionalModel

# Event types - For type hints and event handling
from .types.bidirectional_streaming import (
    AudioInputEvent,
    AudioOutputEvent,
    BidirectionalStreamEvent,
    InterruptionDetectedEvent,
    TextOutputEvent,
    UsageMetricsEvent,
)

__all__ = [
    # Main interface
    "BidirectionalAgent",
    # Model providers
    "NovaSonicBidirectionalModel",
    # Event types
    "AudioInputEvent",
    "AudioOutputEvent",
    "TextOutputEvent",
    "InterruptionDetectedEvent",
    "BidirectionalStreamEvent",
    "UsageMetricsEvent",
    # Model interface
    "BidirectionalModel",
    "BidirectionalModelSession",
]
