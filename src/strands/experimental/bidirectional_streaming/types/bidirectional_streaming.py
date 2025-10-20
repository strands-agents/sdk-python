"""Bidirectional streaming types for real-time audio/text conversations.

Type definitions for bidirectional streaming that extends Strands' existing streaming
capabilities with real-time audio and persistent connection support.

Key features:
- Audio input/output events with standardized formats
- Interruption detection and handling
- Session lifecycle management
- Provider-agnostic event types
- TypedEvent-based discriminated unions for type safety
- Reuses core Strands events (ToolResultEvent, ToolUseStreamEvent)

Audio format normalization:
- Supports PCM, WAV, Opus, and MP3 formats
- Standardizes sample rates (16kHz, 24kHz, 48kHz)
- Normalizes channel configurations (mono/stereo)
- Abstracts provider-specific encodings
"""

from typing import Any, Dict, List, Literal, Optional, Union, cast

from typing_extensions import TypedDict

from ....types._events import ToolResultEvent as CoreToolResultEvent
from ....types._events import ToolUseStreamEvent as CoreToolUseStreamEvent
from ....types._events import TypedEvent
from ....types.content import Role
from ....types.streaming import StreamEvent

# Audio format constants
SUPPORTED_AUDIO_FORMATS = ["pcm", "wav", "opus", "mp3"]
SUPPORTED_SAMPLE_RATES = [16000, 24000, 48000]
SUPPORTED_CHANNELS = [1, 2]  # 1=mono, 2=stereo
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_FORMAT = "pcm"


# =============================================================================
# Input Events (sent via session.send())
# =============================================================================


class AudioInputEvent(TypedEvent):
    """Audio input event for sending audio to the model.

    Used for sending audio data through the unified send() method.

    Args:
        audio: Raw audio bytes to send to model (not base64 encoded).
        format: Audio format from SUPPORTED_AUDIO_FORMATS.
        sample_rate: Sample rate from SUPPORTED_SAMPLE_RATES.
        channels: Channel count from SUPPORTED_CHANNELS (1=mono, 2=stereo).
    """

    def __init__(
        self,
        audio: bytes,
        format: Literal["pcm", "wav", "opus", "mp3"],
        sample_rate: Literal[16000, 24000, 48000],
        channels: Literal[1, 2],
    ) -> None:
        """Initialize audio input event."""
        super().__init__(
            {
                "audio_input": True,
                "audio": audio,
                "format": format,
                "sample_rate": sample_rate,
                "channels": channels,
            }
        )

    @property
    def audio(self) -> bytes:
        """Raw audio data as bytes."""
        return cast(bytes, self.get("audio"))

    @property
    def format(self) -> str:
        """Audio encoding format (pcm, wav, opus, mp3)."""
        return cast(str, self.get("format"))

    @property
    def sample_rate(self) -> int:
        """Number of audio samples per second in Hz."""
        return cast(int, self.get("sample_rate"))

    @property
    def channels(self) -> int:
        """Number of audio channels (1=mono, 2=stereo)."""
        return cast(int, self.get("channels"))


class ImageInputEvent(TypedEvent):
    """Image input event for sending images/video frames to the model.

    Used for sending image data through the unified send() method.
    Supports both raw image bytes and base64-encoded data.

    Args:
        image: Image bytes (raw or base64-encoded string).
        mime_type: MIME type (e.g., "image/jpeg", "image/png").
        encoding: How the imageData is encoded ("base64" or "raw").
    """

    def __init__(
        self,
        image: Union[bytes, str],
        mime_type: str,
        encoding: Literal["base64", "raw"],
    ) -> None:
        """Initialize image input event."""
        super().__init__(
            {
                "image_input": True,
                "image": image,
                "mime_type": mime_type,
                "encoding": encoding,
            }
        )

    @property
    def image(self) -> Union[bytes, str]:
        """Image data, either raw bytes or base64-encoded string."""
        return cast(Union[bytes, str], self.get("image"))

    @property
    def mime_type(self) -> str:
        """MIME type of the image (e.g., image/jpeg, image/png)."""
        return cast(str, self.get("mime_type"))

    @property
    def encoding(self) -> str:
        """How the image data is encoded (base64 or raw)."""
        return cast(str, self.get("encoding"))


# Union type for all input events
InputEvent = Union[AudioInputEvent, ImageInputEvent, CoreToolResultEvent]


# =============================================================================
# Output Events (received via session.receive_events())
# =============================================================================


class SessionStartEvent(TypedEvent):
    """Session established and ready for interaction.

    Args:
        session_id: Unique identifier for this session.
        model: Model identifier (e.g., "gpt-realtime", "gemini-2.0-flash-live").
        capabilities: List of supported features (e.g., ["audio", "tools", "images"]).
    """

    def __init__(self, session_id: str, model: str, capabilities: List[str]) -> None:
        """Initialize session start event."""
        super().__init__(
            {
                "session_start": True,
                "session_id": session_id,
                "model": model,
                "capabilities": capabilities,
            }
        )

    @property
    def session_id(self) -> str:
        """Unique identifier for this session."""
        return cast(str, self.get("session_id"))

    @property
    def model(self) -> str:
        """Model identifier."""
        return cast(str, self.get("model"))

    @property
    def capabilities(self) -> List[str]:
        """List of supported features."""
        return cast(List[str], self.get("capabilities"))


class TurnStartEvent(TypedEvent):
    """Model starts generating a response.

    Args:
        turn_id: Unique identifier for this turn (used in TurnCompleteEvent).
    """

    def __init__(self, turn_id: str) -> None:
        """Initialize turn start event."""
        super().__init__(
            {
                "turn_start": True,
                "turn_id": turn_id,
            }
        )

    @property
    def turn_id(self) -> str:
        """Unique identifier for this turn."""
        return cast(str, self.get("turn_id"))


class AudioStreamEvent(TypedEvent):
    """Streaming audio output from the model.

    Args:
        audio: Raw audio data as bytes (not base64 encoded).
        format: Audio encoding format.
        sample_rate: Number of audio samples per second in Hz.
        channels: Number of audio channels (1=mono, 2=stereo).
    """

    def __init__(
        self,
        audio: bytes,
        format: Literal["pcm", "wav", "opus", "mp3"],
        sample_rate: Literal[16000, 24000, 48000],
        channels: Literal[1, 2],
    ) -> None:
        """Initialize audio stream event."""
        super().__init__(
            {
                "audio_stream": True,
                "audio": audio,
                "format": format,
                "sample_rate": sample_rate,
                "channels": channels,
            }
        )

    @property
    def audio(self) -> bytes:
        """Raw audio data as bytes."""
        return cast(bytes, self.get("audio"))

    @property
    def format(self) -> str:
        """Audio encoding format (pcm, wav, opus, mp3)."""
        return cast(str, self.get("format"))

    @property
    def sample_rate(self) -> int:
        """Number of audio samples per second in Hz."""
        return cast(int, self.get("sample_rate"))

    @property
    def channels(self) -> int:
        """Number of audio channels (1=mono, 2=stereo)."""
        return cast(int, self.get("channels"))


class TranscriptStreamEvent(TypedEvent):
    """Audio transcription of speech (user or assistant).

    Args:
        text: Transcribed text from audio.
        source: Who is speaking ("user" or "assistant").
        is_final: Whether this is the final/complete transcript.
    """

    def __init__(
        self,
        text: str,
        source: Literal["user", "assistant"],
        is_final: bool,
    ) -> None:
        """Initialize transcript stream event."""
        super().__init__(
            {
                "transcript_stream": True,
                "text": text,
                "source": source,
                "is_final": is_final,
            }
        )

    @property
    def text(self) -> str:
        """Transcribed text from audio."""
        return cast(str, self.get("text"))

    @property
    def source(self) -> str:
        """Who is speaking (user or assistant)."""
        return cast(str, self.get("source"))

    @property
    def is_final(self) -> bool:
        """Whether this is the final/complete transcript."""
        return cast(bool, self.get("is_final"))





class InterruptionEvent(TypedEvent):
    """Model generation was interrupted.

    Args:
        reason: Why the interruption occurred ("user_speech" or "error").
        turn_id: ID of the turn that was interrupted (may be None).
    """

    def __init__(
        self,
        reason: Literal["user_speech", "error"],
        turn_id: Optional[str] = None,
    ) -> None:
        """Initialize interruption event."""
        super().__init__(
            {
                "interruption": True,
                "reason": reason,
                "turn_id": turn_id,
            }
        )

    @property
    def reason(self) -> str:
        """Why the interruption occurred."""
        return cast(str, self.get("reason"))

    @property
    def turn_id(self) -> Optional[str]:
        """ID of the turn that was interrupted."""
        return cast(Optional[str], self.get("turn_id"))


class TurnCompleteEvent(TypedEvent):
    """Model finished generating response.

    Args:
        turn_id: ID of the turn that completed (matches TurnStartEvent).
        stop_reason: Why the turn ended.
    """

    def __init__(
        self,
        turn_id: str,
        stop_reason: Literal["complete", "interrupted", "tool_use", "error"],
    ) -> None:
        """Initialize turn complete event."""
        super().__init__(
            {
                "turn_complete": True,
                "turn_id": turn_id,
                "stop_reason": stop_reason,
            }
        )

    @property
    def turn_id(self) -> str:
        """ID of the turn that completed."""
        return cast(str, self.get("turn_id"))

    @property
    def stop_reason(self) -> str:
        """Why the turn ended."""
        return cast(str, self.get("stop_reason"))


class ModalityUsage(TypedDict):
    """Token usage for a specific modality.

    Attributes:
        modality: Type of content (text, audio, image, cached).
        input_tokens: Tokens used for this modality's input.
        output_tokens: Tokens used for this modality's output.
    """

    modality: Literal["text", "audio", "image", "cached"]
    input_tokens: int
    output_tokens: int


class MultimodalUsage(TypedEvent):
    """Token usage event with modality breakdown for multimodal streaming.

    Combines TypedEvent behavior with Usage fields for a unified event type.
    Compatible with strands.types.event_loop.Usage field names.

    Args:
        input_tokens: Total tokens used for all input modalities.
        output_tokens: Total tokens used for all output modalities.
        total_tokens: Sum of input and output tokens.
        modality_details: Optional list of token usage per modality.
        cache_read_input_tokens: Optional tokens read from cache.
        cache_write_input_tokens: Optional tokens written to cache.
    """

    def __init__(
        self,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        modality_details: Optional[List[ModalityUsage]] = None,
        cache_read_input_tokens: Optional[int] = None,
        cache_write_input_tokens: Optional[int] = None,
    ) -> None:
        """Initialize multimodal usage event."""
        data: Dict[str, Any] = {
            "type": "multimodal_usage",
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": total_tokens,
        }
        if modality_details is not None:
            data["modality_details"] = modality_details
        if cache_read_input_tokens is not None:
            data["cacheReadInputTokens"] = cache_read_input_tokens
        if cache_write_input_tokens is not None:
            data["cacheWriteInputTokens"] = cache_write_input_tokens
        super().__init__(data)

    @property
    def input_tokens(self) -> int:
        """Total tokens used for all input modalities."""
        return cast(int, self.get("inputTokens"))

    @property
    def output_tokens(self) -> int:
        """Total tokens used for all output modalities."""
        return cast(int, self.get("outputTokens"))

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens."""
        return cast(int, self.get("totalTokens"))

    @property
    def modality_details(self) -> List[ModalityUsage]:
        """List of token usage per modality."""
        return cast(List[ModalityUsage], self.get("modality_details", []))

    @property
    def cache_read_input_tokens(self) -> Optional[int]:
        """Tokens read from cache."""
        return cast(Optional[int], self.get("cacheReadInputTokens"))

    @property
    def cache_write_input_tokens(self) -> Optional[int]:
        """Tokens written to cache."""
        return cast(Optional[int], self.get("cacheWriteInputTokens"))


class SessionEndEvent(TypedEvent):
    """Session terminated.

    Args:
        reason: Why the session ended.
    """

    def __init__(
        self,
        reason: Literal["client_disconnect", "timeout", "error", "complete"],
    ) -> None:
        """Initialize session end event."""
        super().__init__(
            {
                "session_end": True,
                "reason": reason,
            }
        )

    @property
    def reason(self) -> str:
        """Why the session ended."""
        return cast(str, self.get("reason"))


class ErrorEvent(TypedEvent):
    """Error occurred during the session.

    Follows the pattern of strands.types._events.ForceStopEvent which accepts
    exceptions for consistent error handling.

    Args:
        error: The exception that occurred.
        code: Optional error code for programmatic handling (defaults to exception class name).
        details: Optional additional error information.
    """

    def __init__(
        self,
        error: Exception,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize error event."""
        super().__init__(
            {
                "bidirectional_error": True,
                "error": error,
                "error_message": str(error),
                "error_code": code or type(error).__name__,
                "error_details": details,
            }
        )

    @property
    def error(self) -> Exception:
        """The exception that occurred."""
        return cast(Exception, self.get("error"))

    @property
    def code(self) -> str:
        """Error code for programmatic handling."""
        return cast(str, self.get("error_code"))

    @property
    def message(self) -> str:
        """Human-readable error message."""
        return cast(str, self.get("error_message"))

    @property
    def details(self) -> Optional[Dict[str, Any]]:
        """Optional additional error information."""
        return cast(Optional[Dict[str, Any]], self.get("error_details"))


# Union type for all output events
OutputEvent = Union[
    SessionStartEvent,
    TurnStartEvent,
    AudioStreamEvent,
    TranscriptStreamEvent,
    CoreToolUseStreamEvent,
    InterruptionEvent,
    TurnCompleteEvent,
    MultimodalUsage,
    SessionEndEvent,
    ErrorEvent,
]


# =============================================================================
# Legacy TypedDict Events (for backward compatibility)
# =============================================================================


class LegacyAudioOutputEvent(TypedDict):
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


class LegacyAudioInputEvent(TypedDict):
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


class LegacyImageInputEvent(TypedDict):
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


class LegacyTextOutputEvent(TypedDict):
    """Text output event from the model during bidirectional streaming.

    Attributes:
        text: The text content from the model.
        role: The role of the message sender.
    """

    text: str
    role: Role


class LegacyTranscriptEvent(TypedDict):
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


class LegacyInterruptionDetectedEvent(TypedDict):
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


class LegacyVoiceActivityEvent(TypedDict):
    """Voice activity detection event for speech monitoring.

    Provides standardized voice activity detection events across providers
    to enable speech-aware applications and better conversation flow.

    Attributes:
        activityType: Type of voice activity detected.
    """

    activityType: Literal["speech_started", "speech_stopped", "timeout"]


class LegacyUsageMetricsEvent(TypedDict):
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

    DEPRECATED: This TypedDict-based event system is maintained for backward
    compatibility. New code should use the TypedEvent-based events above
    (AudioInputEvent, AudioStreamEvent, etc.) which provide better type safety
    and property-based access.

    Extends the existing StreamEvent type with bidirectional-specific events
    while maintaining full backward compatibility with existing Strands streaming.

    Attributes:
        audioOutput: Audio output from the model (legacy).
        audioInput: Audio input sent to the model (legacy).
        imageInput: Image input sent to the model (legacy).
        textOutput: Text output from the model (legacy).
        transcript: Audio transcription (input or output) (legacy).
        interruptionDetected: User interruption detection (legacy).
        BidirectionalConnectionStart: connection start event (legacy).
        BidirectionalConnectionEnd: connection end event (legacy).
        voiceActivity: Voice activity detection events (legacy).
        usageMetrics: Token usage and performance metrics (legacy).
    """

    audioOutput: Optional[LegacyAudioOutputEvent]
    audioInput: Optional[LegacyAudioInputEvent]
    imageInput: Optional[LegacyImageInputEvent]
    textOutput: Optional[LegacyTextOutputEvent]
    transcript: Optional[LegacyTranscriptEvent]
    interruptionDetected: Optional[LegacyInterruptionDetectedEvent]
    BidirectionalConnectionStart: Optional[BidirectionalConnectionStartEvent]
    BidirectionalConnectionEnd: Optional[BidirectionalConnectionEndEvent]
    voiceActivity: Optional[LegacyVoiceActivityEvent]
    usageMetrics: Optional[LegacyUsageMetricsEvent]


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# Alias old names to legacy versions for backward compatibility
AudioOutputEvent = LegacyAudioOutputEvent
TextOutputEvent = LegacyTextOutputEvent
TranscriptEvent = LegacyTranscriptEvent
InterruptionDetectedEvent = LegacyInterruptionDetectedEvent
VoiceActivityEvent = LegacyVoiceActivityEvent
UsageMetricsEvent = LegacyUsageMetricsEvent

# Backward compatibility: old UsageEvent name
UsageEvent = MultimodalUsage
