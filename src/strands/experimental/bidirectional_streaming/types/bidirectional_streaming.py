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

from strands.types.content import Role
from strands.types.streaming import StreamEvent
from typing_extensions import TypedDict

# Audio format constants
SUPPORTED_AUDIO_FORMATS = ['pcm', 'wav', 'opus', 'mp3']
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
    format: Literal['pcm', 'wav', 'opus', 'mp3']
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
    format: Literal['pcm', 'wav', 'opus', 'mp3']
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


class InterruptionDetectedEvent(TypedDict):
    """Interruption detection event.
    
    Signals when user interruption is detected during model generation.
    
    Attributes:
        reason: Interruption reason from predefined set.
    """
    
    reason: Literal['user_input', 'vad_detected', 'manual']


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
    
    reason: Literal['user_request', 'timeout', 'error']
    connectionId: Optional[str]
    metadata: Optional[Dict[str, Any]]


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
    """
    
    audioOutput: AudioOutputEvent
    audioInput: AudioInputEvent
    textOutput: TextOutputEvent
    interruptionDetected: InterruptionDetectedEvent
    BidirectionalConnectionStart: BidirectionalConnectionStartEvent
    BidirectionalConnectionEnd: BidirectionalConnectionEndEvent

