"""Bidirectional streaming types for real-time audio/text conversations.

PROBLEM ADDRESSED:
-----------------
Strands currently uses a request-response architecture without bidirectional streaming 
support. Users cannot interrupt ongoing responses, provide additional context during 
processing, or engage in real-time conversations. Each interaction requires a complete 
request-response cycle.

ARCHITECTURAL TRANSFORMATION:
----------------------------
Current Limitations: Strands' unidirectional architecture follows sequential 
request-response cycles that prevent real-time interaction. This represents a 
pull-based architecture where the model receives the request, processes it, and 
sends a response back.

Bidirectional Solution: Uses persistent session-based connections with continuous 
input and output flow. This implements a push-based architecture where the model 
sends updates to the client as soon as response becomes available, without explicit 
client requests.

KEY CHARACTERISTICS:
-------------------
- Persistent Sessions: Connections remain open for extended periods (Nova Sonic: 8 minutes, 
  Google Live API: 15 minutes, OpenAI Realtime: 30 minutes) maintaining conversation context
- Bidirectional Communication: Users can send input while models generate responses
- Interruption Handling: Users can interrupt ongoing model responses in real-time without 
  terminating the session
- Tool Execution: Tools execute concurrently within the conversation flow rather than 
  requiring requests rebuilding

PROVIDER NORMALIZATION:
----------------------
Must normalize incompatible audio formats: Nova Sonic's hex-encoded base64, Google's 
LINEAR16 PCM, OpenAI's Base64-encoded PCM16. Requires unified interruption event types 
to handle Nova Sonic's stopReason = INTERRUPTED events, Google's VAD cancellation, and 
OpenAI's conversation.item.truncate.

This module extends existing StreamEvent types while maintaining backward compatibility 
with existing Strands streaming patterns.
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
    
    Standardizes audio output across different providers using raw bytes
    instead of provider-specific encodings (base64, hex, etc.).
    
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
    
    Used when sending audio data through send_audio() method.
    
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
    """Session start event for bidirectional streaming.
    
    Attributes:
        sessionId: Unique session identifier.
        metadata: Provider-specific session metadata.
    """
    
    sessionId: Optional[str]
    metadata: Optional[Dict[str, Any]]


class BidirectionalConnectionEndEvent(TypedDict):
    """Session end event for bidirectional streaming.
    
    Attributes:
        reason: Reason for session end from predefined set.
        sessionId: Unique session identifier.
        metadata: Provider-specific session metadata.
    """
    
    reason: Literal['user_request', 'timeout', 'error']
    sessionId: Optional[str]
    metadata: Optional[Dict[str, Any]]


class BidirectionalStreamEvent(StreamEvent, total=False):
    """Bidirectional stream event extending existing StreamEvent.
    
    Inherits all existing StreamEvent fields (contentBlockDelta, toolUse, 
    messageStart, etc.) while adding bidirectional-specific events.
    Maintains full backward compatibility with existing Strands streaming.
    
    Attributes:
        audioOutput: Audio output from the model.
        audioInput: Audio input sent to the model.
        textOutput: Text output from the model.
        interruptionDetected: User interruption detection.
        BidirectionalConnectionStart: Session start event.
        BidirectionalConnectionEnd: Session end event.
    """
    
    audioOutput: AudioOutputEvent
    audioInput: AudioInputEvent
    textOutput: TextOutputEvent
    interruptionDetected: InterruptionDetectedEvent
    BidirectionalConnectionStart: BidirectionalConnectionStartEvent
    BidirectionalConnectionEnd: BidirectionalConnectionEndEvent

