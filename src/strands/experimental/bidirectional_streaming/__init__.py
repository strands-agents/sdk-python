"""Bidirectional streaming package."""

# Main components - Primary user interface
from .agent.agent import BidirectionalAgent

# Advanced interfaces (for custom implementations)
from .models.bidirectional_model import BidirectionalModel, BidirectionalModelSession

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
