"""Type definitions for bidirectional streaming."""

# Import from core Strands
from ....types._events import ToolResultEvent as CoreToolResultEvent
from ....types._events import ToolUseStreamEvent as CoreToolUseStreamEvent

# Import from bidirectional_streaming module
from .bidirectional_streaming import (
    DEFAULT_CHANNELS,
    DEFAULT_FORMAT,
    DEFAULT_SAMPLE_RATE,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_CHANNELS,
    SUPPORTED_SAMPLE_RATES,
    AudioInputEvent,
    AudioOutputEvent,
    AudioStreamEvent,
    BidirectionalConnectionEndEvent,
    BidirectionalConnectionStartEvent,
    BidirectionalStreamEvent,
    ErrorEvent,
    ImageInputEvent,
    InputEvent,
    InterruptionDetectedEvent,
    InterruptionEvent,
    LegacyAudioInputEvent,
    LegacyAudioOutputEvent,
    LegacyImageInputEvent,
    LegacyInterruptionDetectedEvent,
    LegacyTextOutputEvent,
    LegacyTranscriptEvent,
    LegacyUsageMetricsEvent,
    LegacyVoiceActivityEvent,
    ModalityUsage,
    MultimodalUsage,
    OutputEvent,
    SessionEndEvent,
    SessionStartEvent,
    TextOutputEvent,
    TranscriptEvent,
    TranscriptStreamEvent,
    TurnCompleteEvent,
    TurnStartEvent,
    UsageEvent,
    UsageMetricsEvent,
    VoiceActivityEvent,
)

# Re-export core events for convenience
ToolResultEvent = CoreToolResultEvent
ToolUseStreamEvent = CoreToolUseStreamEvent
# Backward compatibility alias
ToolUseEvent = CoreToolUseStreamEvent

__all__ = [
    # New TypedEvent-based events (preferred)
    "AudioInputEvent",
    "AudioStreamEvent",
    "ErrorEvent",
    "ImageInputEvent",
    "InputEvent",
    "InterruptionEvent",
    "ModalityUsage",
    "MultimodalUsage",
    "OutputEvent",
    "SessionEndEvent",
    "SessionStartEvent",
    "ToolResultEvent",
    "ToolUseEvent",
    "ToolUseStreamEvent",
    "TranscriptStreamEvent",
    "TurnCompleteEvent",
    "TurnStartEvent",
    "UsageEvent",
    # Backward compatibility aliases (point to legacy TypedDict versions)
    "AudioOutputEvent",
    "TextOutputEvent",
    "TranscriptEvent",
    "InterruptionDetectedEvent",
    "VoiceActivityEvent",
    "UsageMetricsEvent",
    # Legacy TypedDict events (backward compatibility)
    "LegacyAudioInputEvent",
    "LegacyAudioOutputEvent",
    "LegacyImageInputEvent",
    "LegacyInterruptionDetectedEvent",
    "LegacyTextOutputEvent",
    "LegacyTranscriptEvent",
    "LegacyUsageMetricsEvent",
    "LegacyVoiceActivityEvent",
    "BidirectionalConnectionEndEvent",
    "BidirectionalConnectionStartEvent",
    "BidirectionalStreamEvent",
    # Constants
    "SUPPORTED_AUDIO_FORMATS",
    "SUPPORTED_SAMPLE_RATES",
    "SUPPORTED_CHANNELS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHANNELS",
    "DEFAULT_FORMAT",
]
