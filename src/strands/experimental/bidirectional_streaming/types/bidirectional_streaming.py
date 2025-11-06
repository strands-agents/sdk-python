"""Bidirectional streaming types for real-time audio/text conversations.

Type definitions for bidirectional streaming that extends Strands' existing streaming
capabilities with real-time audio and persistent connection support.

Key features:
- Audio input/output events with standardized formats
- Interruption detection and handling
- Session lifecycle management
- Provider-agnostic event types
- Type-safe discriminated unions with TypedEvent
- JSON-serializable events (audio/images stored as base64 strings)

Audio format normalization:
- Supports PCM, WAV, Opus, and MP3 formats
- Standardizes sample rates (16kHz, 24kHz, 48kHz)
- Normalizes channel configurations (mono/stereo)
- Abstracts provider-specific encodings
- Audio data stored as base64-encoded strings for JSON compatibility
"""

from typing import Any, Dict, List, Literal, Optional, Union, cast

from ....types._events import TypedEvent

# Audio format constants
SUPPORTED_AUDIO_FORMATS = ["pcm", "wav", "opus", "mp3"]
SUPPORTED_SAMPLE_RATES = [16000, 24000, 48000]
SUPPORTED_CHANNELS = [1, 2]  # 1=mono, 2=stereo
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_FORMAT = "pcm"


# ============================================================================
# Input Events (sent via session.send())
# ============================================================================


class TextInputEvent(TypedEvent):
    """Text input event for sending text to the model.

    Used for sending text content through the send() method.

    Parameters:
        text: The text content to send to the model.
        role: The role of the message sender (typically "user").
    """

    def __init__(self, text: str, role: str):
        super().__init__(
            {
                "type": "bidirectional_text_input",
                "text": text,
                "role": role,
            }
        )

    @property
    def text(self) -> str:
        return cast(str, self.get("text"))

    @property
    def role(self) -> str:
        return cast(str, self.get("role"))


class AudioInputEvent(TypedEvent):
    """Audio input event for sending audio to the model.

    Used for sending audio data through the send() method.

    Parameters:
        audio: Base64-encoded audio string to send to model.
        format: Audio format from SUPPORTED_AUDIO_FORMATS.
        sample_rate: Sample rate from SUPPORTED_SAMPLE_RATES.
        channels: Channel count from SUPPORTED_CHANNELS.
    """

    def __init__(
        self,
        audio: str,
        format: Literal["pcm", "wav", "opus", "mp3"],
        sample_rate: Literal[16000, 24000, 48000],
        channels: Literal[1, 2],
    ):
        super().__init__(
            {
                "type": "bidirectional_audio_input",
                "audio": audio,
                "format": format,
                "sample_rate": sample_rate,
                "channels": channels,
            }
        )

    @property
    def audio(self) -> str:
        return cast(str, self.get("audio"))

    @property
    def format(self) -> str:
        return cast(str, self.get("format"))

    @property
    def sample_rate(self) -> int:
        return cast(int, self.get("sample_rate"))

    @property
    def channels(self) -> int:
        return cast(int, self.get("channels"))


class ImageInputEvent(TypedEvent):
    """Image input event for sending images/video frames to the model.

    Used for sending image data through the send() method.

    Parameters:
        image: Base64-encoded image string.
        mime_type: MIME type (e.g., "image/jpeg", "image/png").
    """

    def __init__(
        self,
        image: str,
        mime_type: str,
    ):
        super().__init__(
            {
                "type": "bidirectional_image_input",
                "image": image,
                "mime_type": mime_type,
            }
        )

    @property
    def image(self) -> str:
        return cast(str, self.get("image"))

    @property
    def mime_type(self) -> str:
        return cast(str, self.get("mime_type"))


# ============================================================================
# Output Events (received via session.receive_events())
# ============================================================================


class ConnectionStartEvent(TypedEvent):
    """Streaming connection established and ready for interaction.

    Parameters:
        connection_id: Unique identifier for this streaming connection.
        model: Model identifier (e.g., "gpt-realtime", "gemini-2.0-flash-live").
        capabilities: List of supported features (e.g., ["audio", "tools", "images"]).
    """

    def __init__(self, connection_id: str, model: str, capabilities: List[str]):
        super().__init__(
            {
                "type": "bidirectional_connection_start",
                "connection_id": connection_id,
                "model": model,
                "capabilities": capabilities,
            }
        )

    @property
    def connection_id(self) -> str:
        return cast(str, self.get("connection_id"))

    @property
    def model(self) -> str:
        return cast(str, self.get("model"))

    @property
    def capabilities(self) -> List[str]:
        return cast(List[str], self.get("capabilities"))


class TurnStartEvent(TypedEvent):
    """Model starts generating a response.

    Parameters:
        turn_id: Unique identifier for this turn (used in turn.complete).
    """

    def __init__(self, turn_id: str):
        super().__init__({"type": "bidirectional_turn_start", "turn_id": turn_id})

    @property
    def turn_id(self) -> str:
        return cast(str, self.get("turn_id"))


class AudioStreamEvent(TypedEvent):
    """Streaming audio output from the model.

    Parameters:
        audio: Base64-encoded audio string.
        format: Audio encoding format.
        sample_rate: Number of audio samples per second in Hz.
        channels: Number of audio channels (1=mono, 2=stereo).
    """

    def __init__(
        self,
        audio: str,
        format: Literal["pcm", "wav", "opus", "mp3"],
        sample_rate: Literal[16000, 24000, 48000],
        channels: Literal[1, 2],
    ):
        super().__init__(
            {
                "type": "bidirectional_audio_stream",
                "audio": audio,
                "format": format,
                "sample_rate": sample_rate,
                "channels": channels,
            }
        )

    @property
    def audio(self) -> str:
        return cast(str, self.get("audio"))

    @property
    def format(self) -> str:
        return cast(str, self.get("format"))

    @property
    def sample_rate(self) -> int:
        return cast(int, self.get("sample_rate"))

    @property
    def channels(self) -> int:
        return cast(int, self.get("channels"))


class TranscriptStreamEvent(TypedEvent):
    """Audio transcription of speech (user or assistant).

    Parameters:
        text: Transcribed text from audio.
        source: Who is speaking ("user" or "assistant").
        is_final: Whether this is the final/complete transcript.
    """

    def __init__(
        self, text: str, source: Literal["user", "assistant"], is_final: bool
    ):
        super().__init__(
            {
                "type": "bidirectional_transcript_stream",
                "text": text,
                "source": source,
                "is_final": is_final,
            }
        )

    @property
    def text(self) -> str:
        return cast(str, self.get("text"))

    @property
    def source(self) -> str:
        return cast(str, self.get("source"))

    @property
    def is_final(self) -> bool:
        return cast(bool, self.get("is_final"))


class InterruptionEvent(TypedEvent):
    """Model generation was interrupted.

    Parameters:
        reason: Why the interruption occurred.
        turn_id: ID of the turn that was interrupted (may be None).
    """

    def __init__(
        self, reason: Literal["user_speech", "error"], turn_id: Optional[str] = None
    ):
        super().__init__(
            {
                "type": "bidirectional_interruption",
                "reason": reason,
                "turn_id": turn_id,
            }
        )

    @property
    def reason(self) -> str:
        return cast(str, self.get("reason"))

    @property
    def turn_id(self) -> Optional[str]:
        return cast(Optional[str], self.get("turn_id"))


class TurnCompleteEvent(TypedEvent):
    """Model finished generating response.

    Parameters:
        turn_id: ID of the turn that completed (matches turn.start).
        stop_reason: Why the turn ended.
    """

    def __init__(
        self,
        turn_id: str,
        stop_reason: Literal["complete", "interrupted", "tool_use", "error"],
    ):
        super().__init__(
            {
                "type": "bidirectional_turn_complete",
                "turn_id": turn_id,
                "stop_reason": stop_reason,
            }
        )

    @property
    def turn_id(self) -> str:
        return cast(str, self.get("turn_id"))

    @property
    def stop_reason(self) -> str:
        return cast(str, self.get("stop_reason"))


class ModalityUsage(dict):
    """Token usage for a specific modality.

    Attributes:
        modality: Type of content.
        input_tokens: Tokens used for this modality's input.
        output_tokens: Tokens used for this modality's output.
    """

    modality: Literal["text", "audio", "image", "cached"]
    input_tokens: int
    output_tokens: int


class UsageEvent(TypedEvent):
    """Token usage event with modality breakdown for bidirectional streaming.

    Tracks token consumption across different modalities (audio, text, images)
    during bidirectional streaming sessions.

    Parameters:
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
    ):
        data: Dict[str, Any] = {
            "type": "bidirectional_usage",
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
        return cast(int, self.get("inputTokens"))

    @property
    def output_tokens(self) -> int:
        return cast(int, self.get("outputTokens"))

    @property
    def total_tokens(self) -> int:
        return cast(int, self.get("totalTokens"))

    @property
    def modality_details(self) -> List[ModalityUsage]:
        return cast(List[ModalityUsage], self.get("modality_details", []))

    @property
    def cache_read_input_tokens(self) -> Optional[int]:
        return cast(Optional[int], self.get("cacheReadInputTokens"))

    @property
    def cache_write_input_tokens(self) -> Optional[int]:
        return cast(Optional[int], self.get("cacheWriteInputTokens"))


class ConnectionCloseEvent(TypedEvent):
    """Streaming connection closed.

    Parameters:
        connection_id: Unique identifier for this streaming connection (matches ConnectionStartEvent).
        reason: Why the connection was closed.
    """

    def __init__(
        self,
        connection_id: str,
        reason: Literal["client_disconnect", "timeout", "error", "complete"],
    ):
        super().__init__(
            {
                "type": "bidirectional_connection_close",
                "connection_id": connection_id,
                "reason": reason,
            }
        )

    @property
    def connection_id(self) -> str:
        return cast(str, self.get("connection_id"))

    @property
    def reason(self) -> str:
        return cast(str, self.get("reason"))


class ErrorEvent(TypedEvent):
    """Error occurred during the session.

    Stores the full Exception object as an instance attribute for debugging while
    keeping the event dict JSON-serializable. The exception can be accessed via
    the `error` property for re-raising or type-based error handling.

    Parameters:
        error: The exception that occurred.
        details: Optional additional error information.
    """

    def __init__(
        self,
        error: Exception,
        details: Optional[Dict[str, Any]] = None,
    ):
        # Store serializable data in dict (for JSON serialization)
        super().__init__(
            {
                "type": "bidirectional_error",
                "message": str(error),
                "code": type(error).__name__,
                "details": details,
            }
        )
        # Store exception as instance attribute (not serialized)
        self._error = error

    @property
    def error(self) -> Exception:
        """The original exception that occurred.
        
        Can be used for re-raising or type-based error handling.
        """
        return self._error

    @property
    def code(self) -> str:
        """Error code derived from exception class name."""
        return cast(str, self.get("code"))

    @property
    def message(self) -> str:
        """Human-readable error message from the exception."""
        return cast(str, self.get("message"))

    @property
    def details(self) -> Optional[Dict[str, Any]]:
        """Additional error context beyond the exception itself."""
        return cast(Optional[Dict[str, Any]], self.get("details"))


# ============================================================================
# Type Unions
# ============================================================================

# Note: ToolResultEvent and ToolUseStreamEvent are reused from strands.types._events

InputEvent = Union[TextInputEvent, AudioInputEvent, ImageInputEvent]

OutputEvent = Union[
    ConnectionStartEvent,
    TurnStartEvent,
    AudioStreamEvent,
    TranscriptStreamEvent,
    InterruptionEvent,
    TurnCompleteEvent,
    UsageEvent,
    ConnectionCloseEvent,
    ErrorEvent,
]
