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


class ImageInputEvent(TypedDict):
    """Image input event for sending images/video frames to the model.
    
    Used for sending image data through the send() method. Supports both
    raw image bytes and base64-encoded data.
    
    Attributes:
        imageData: Image bytes (raw or base64-encoded string).
        mimeType: MIME type (e.g., "image/jpeg", "image/png").
        encoding: How the imageData is encoded.
    """
    
    imageData: bytes | str
    mimeType: str
    encoding: Literal["base64", "raw"]

class TextInputEvent(TypedDict):
    """Text input event for sending text messages to the model.

    Used for sending text messages through the send() method.

    Attributes:
        text: The text content to send to the model.
    """

    text: str

class TextOutputEvent(TypedDict):
    """Text output event from the model during bidirectional streaming.

    Attributes:
        text: The text content from the model.
        role: The role of the message sender.
    """

    text: str
    role: Role


class TranscriptEvent(TypedDict):
    """Transcript event for audio transcriptions.
    
    Used for both input transcriptions (user speech) and output transcriptions
    (model audio). These are informational and separate from actual text responses.
    
    Attributes:
        text: The transcribed text.
        role: The role of the speaker ("user" or "assistant").
        type: Type of transcription ("input" or "output").
    """
    
    text: str
    role: Role
    type: Literal["input", "output"]


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


class VoiceActivityEvent(TypedDict):
    """Voice activity detection event for speech monitoring.

    Provides standardized voice activity detection events across providers
    to enable speech-aware applications and better conversation flow.

    Attributes:
        activityType: Type of voice activity detected.
    """

    activityType: Literal["speech_started", "speech_stopped", "timeout"]


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
        imageInput: Image input sent to the model.
        textOutput: Text output from the model.
        transcript: Audio transcription (input or output).
        interruptionDetected: User interruption detection.
        BidirectionalConnectionStart: connection start event.
        BidirectionalConnectionEnd: connection end event.
        voiceActivity: Voice activity detection events.
        usageMetrics: Token usage and performance metrics.
    """

    audioOutput: Optional[AudioOutputEvent]
    audioInput: Optional[AudioInputEvent]
    imageInput: Optional[ImageInputEvent]
    textOutput: Optional[TextOutputEvent]
    transcript: Optional[TranscriptEvent]
    interruptionDetected: Optional[InterruptionDetectedEvent]
    BidirectionalConnectionStart: Optional[BidirectionalConnectionStartEvent]
    BidirectionalConnectionEnd: Optional[BidirectionalConnectionEndEvent]
    voiceActivity: Optional[VoiceActivityEvent]
    usageMetrics: Optional[UsageMetricsEvent]
