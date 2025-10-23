"""Bidirectional streaming types for real-time audio/text conversations.

Type definitions for bidirectional streaming that extends Strands' existing streaming
capabilities with real-time audio and persistent connection support.

Key features:
- Audio input/output events with standardized formats
- Interruption detection and handling
- connection lifecycle management
- Provider-agnostic event types
- Backwards compatibility with existing StreamEvent types

Audio format normalization:
- Supports PCM, WAV, Opus, and MP3 formats
- Standardizes sample rates (16kHz, 24kHz, 48kHz)
- Normalizes channel configurations (mono/stereo)
- Abstracts provider-specific encodings
"""

from typing import Any, Dict, Literal, Optional

from typing_extensions import TypedDict

from ....types.content import Role
from ....types.streaming import StreamEvent

# Audio format constants
SUPPORTED_AUDIO_FORMATS = ["pcm", "wav", "opus", "mp3"]
SUPPORTED_SAMPLE_RATES = [16000, 24000, 48000]
SUPPORTED_CHANNELS = [1, 2]  # 1=mono, 2=stereo
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1


class AudioOutputEvent(TypedDict):
    """Audio output event from the model.

    Provides standardized audio output format across different providers using
    raw bytes instead of provider-specific encodings.

    Attributes:
        audioData: Raw audio bytes (not base64 or hex encoded).
        format: Audio format from SUPPORTED_AUDIO_FORMATS.
        sampleRate: Sample rate from SUPPORTED_SAMPLE_RATES.
        channels: Channel count from SUPPORTED_CHANNELS.
        encoding: Original provider encoding for debugging purposes.
    """

    audioData: bytes
    format: Literal["pcm", "wav", "opus", "mp3"]
    sampleRate: Literal[16000, 24000, 48000]
    channels: Literal[1, 2]
    encoding: Optional[str]


class AudioInputEvent(TypedDict):
    """Audio input event for sending audio to the model.

    Used for sending audio data through the send() method.

    Attributes:
        audioData: Raw audio bytes to send to model.
        format: Audio format from SUPPORTED_AUDIO_FORMATS.
        sampleRate: Sample rate from SUPPORTED_SAMPLE_RATES.
        channels: Channel count from SUPPORTED_CHANNELS.
    """

    audioData: bytes
    format: Literal["pcm", "wav", "opus", "mp3"]
    sampleRate: Literal[16000, 24000, 48000]
    channels: Literal[1, 2]


class TextOutputEvent(TypedDict):
    """Text output event from the model during bidirectional streaming.

    Attributes:
        text: The text content from the model.
        role: The role of the message sender.
    """

    text: str
    role: Role


class ImageInputEvent(TypedDict):
    """Image input event for sending images/video frames to the model.

    Supports multiple input methods following OpenAI realtime API patterns:
    - Base64 data URLs (data:image/png;base64,...)
    - Hosted URLs (https://...)
    - OpenAI file IDs (file-...)
    - Raw bytes with MIME type

    Attributes:
        image_url: Data URL, hosted URL, or OpenAI file ID.
        imageData: Raw image bytes (alternative to image_url).
        mimeType: MIME type when using imageData.
    """

    image_url: Optional[str]  # Primary: data URL, hosted URL, or file ID
    imageData: Optional[bytes]  # Alternative: raw bytes
    mimeType: Optional[str]  # Required when using imageData


class InterruptionDetectedEvent(TypedDict):
    """Interruption detection event.

    Signals when user interruption is detected during model generation.

    Attributes:
        reason: Interruption reason from predefined set.
    """

    reason: Literal["user_input", "vad_detected", "manual"]


class BidirectionalConnectionStartEvent(TypedDict, total=False):
    """connection start event for bidirectional streaming.

    Attributes:
        connectionId: Unique connection identifier.
        metadata: Provider-specific connection metadata.
    """

    connectionId: Optional[str]
    metadata: Optional[Dict[str, Any]]


class BidirectionalConnectionEndEvent(TypedDict):
    """connection end event for bidirectional streaming.

    Attributes:
        reason: Reason for connection end from predefined set.
        connectionId: Unique connection identifier.
        metadata: Provider-specific connection metadata.
    """

    reason: Literal["user_request", "timeout", "error", "connection_complete"]
    connectionId: Optional[str]
    metadata: Optional[Dict[str, Any]]


class ToolResultInputEvent(TypedDict):
    """Tool result input event for sending tool execution results.

    Attributes:
        tool_use_id: Identifier for the tool use being responded to.
        result: Tool execution result data.
    """

    tool_use_id: str
    result: Dict[str, Any]


class UsageMetricsEvent(TypedDict):
    """Token usage and performance tracking.

    Provides standardized usage metrics across providers for cost monitoring
    and performance optimization.

    Attributes:
        totalTokens: Total tokens used in the interaction.
        inputTokens: Tokens used for input processing.
        outputTokens: Tokens used for output generation.
        audioTokens: Tokens used specifically for audio processing.
    """

    totalTokens: Optional[int]
    inputTokens: Optional[int]
    outputTokens: Optional[int]
    audioTokens: Optional[int]


class BidirectionalStreamEvent(StreamEvent, total=False):
    """Bidirectional stream event extending existing StreamEvent.

    Extends the existing StreamEvent type with bidirectional-specific events
    while maintaining full backward compatibility with existing Strands streaming.

    Attributes:
        audioOutput: Audio output from the model.
        audioInput: Audio input sent to the model.
        textOutput: Text output from the model.
        interruptionDetected: User interruption detection.
        BidirectionalConnectionStart: connection start event.
        BidirectionalConnectionEnd: connection end event.
        usageMetrics: Token usage and performance metrics.
    """

    audioOutput: Optional[AudioOutputEvent]
    audioInput: Optional[AudioInputEvent]
    textOutput: Optional[TextOutputEvent]
    interruptionDetected: Optional[InterruptionDetectedEvent]
    BidirectionalConnectionStart: Optional[BidirectionalConnectionStartEvent]
    BidirectionalConnectionEnd: Optional[BidirectionalConnectionEndEvent]
    usageMetrics: Optional[UsageMetricsEvent]
