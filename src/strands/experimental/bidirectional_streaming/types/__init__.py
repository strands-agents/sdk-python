"""Type definitions for bidirectional streaming."""

from .agent import BidiAgentInput
from .io import BidiInput, BidiOutput
from .events import (
    DEFAULT_CHANNELS,
    DEFAULT_FORMAT,
    DEFAULT_SAMPLE_RATE,
    SUPPORTED_AUDIO_FORMATS,
    SUPPORTED_CHANNELS,
    SUPPORTED_SAMPLE_RATES,
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

__all__ = [
    "BidiInput",
    "BidiOutput",
    "BidiAgentInput",
    # Input Events
    "BidiTextInputEvent",
    "BidiAudioInputEvent",
    "BidiImageInputEvent",
    "BidiInputEvent",
    # Output Events
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
    # Constants
    "SUPPORTED_AUDIO_FORMATS",
    "SUPPORTED_SAMPLE_RATES",
    "SUPPORTED_CHANNELS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_CHANNELS",
    "DEFAULT_FORMAT",
]
