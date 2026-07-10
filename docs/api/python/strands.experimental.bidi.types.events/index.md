Bidirectional streaming types for real-time audio/text conversations.

Type definitions for bidirectional streaming that extends Strands’ existing streaming capabilities with real-time audio and persistent connection support.

Key features:

-   Audio input/output events with standardized formats
-   Interruption detection and handling
-   Connection lifecycle management
-   Provider-agnostic event types
-   Type-safe discriminated unions with TypedEvent
-   JSON-serializable events (audio/images stored as base64 strings)

Audio format normalization:

-   Supports PCM, WAV, Opus, and MP3 formats
-   Standardizes sample rates (16kHz, 24kHz, 48kHz)
-   Normalizes channel configurations (mono/stereo)
-   Abstracts provider-specific encodings
-   Audio data stored as base64-encoded strings for JSON compatibility

#### AudioChannel

Number of audio channels.

-   Mono: 1
-   Stereo: 2

#### AudioFormat

Audio encoding format.

#### AudioSampleRate

Audio sample rate in Hz.

#### Role

Role of a message sender.

-   “user”: Messages from the user to the assistant.
-   “assistant”: Messages from the assistant to the user.

#### StopReason

Reason for the model ending its response generation.

-   “complete”: Model completed its response.
-   “error”: Model encountered an error.
-   “interrupted”: Model was interrupted by the user.
-   “tool\_use”: Model is requesting a tool use.

## BidiTextInputEvent

```python
class BidiTextInputEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:97](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L97)

Text input event for sending text to the model.

Used for sending text content through the send() method.

**Arguments**:

-   `text` - The text content to send to the model.
-   `role` - The role of the message sender (default: “user”).

#### \_\_init\_\_

```python
def __init__(text: str, role: Role = "user")
```

Defined in: [src/strands/experimental/bidi/types/events.py:107](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L107)

Initialize text input event.

#### text

```python
@property
def text() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:118](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L118)

The text content to send to the model.

#### role

```python
@property
def role() -> Role
```

Defined in: [src/strands/experimental/bidi/types/events.py:123](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L123)

The role of the message sender.

## BidiAudioInputEvent

```python
class BidiAudioInputEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:128](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L128)

Audio input event for sending audio to the model.

Used for sending audio data through the send() method.

**Arguments**:

-   `audio` - Base64-encoded audio string to send to model.
-   `format` - Audio format from SUPPORTED\_AUDIO\_FORMATS.
-   `sample_rate` - Sample rate from SUPPORTED\_SAMPLE\_RATES.
-   `channels` - Channel count from SUPPORTED\_CHANNELS.

#### \_\_init\_\_

```python
def __init__(audio: str, format: AudioFormat | str,
             sample_rate: AudioSampleRate, channels: AudioChannel)
```

Defined in: [src/strands/experimental/bidi/types/events.py:140](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L140)

Initialize audio input event.

#### audio

```python
@property
def audio() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:159](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L159)

Base64-encoded audio string.

#### format

```python
@property
def format() -> AudioFormat
```

Defined in: [src/strands/experimental/bidi/types/events.py:164](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L164)

Audio encoding format.

#### sample\_rate

```python
@property
def sample_rate() -> AudioSampleRate
```

Defined in: [src/strands/experimental/bidi/types/events.py:169](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L169)

Number of audio samples per second in Hz.

#### channels

```python
@property
def channels() -> AudioChannel
```

Defined in: [src/strands/experimental/bidi/types/events.py:174](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L174)

Number of audio channels (1=mono, 2=stereo).

## BidiImageInputEvent

```python
class BidiImageInputEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:179](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L179)

Image input event for sending images/video frames to the model.

Used for sending image data through the send() method.

**Arguments**:

-   `image` - Base64-encoded image string.
-   `mime_type` - MIME type (e.g., “image/jpeg”, “image/png”).

#### \_\_init\_\_

```python
def __init__(image: str, mime_type: str)
```

Defined in: [src/strands/experimental/bidi/types/events.py:189](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L189)

Initialize image input event.

#### image

```python
@property
def image() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:204](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L204)

Base64-encoded image string.

#### mime\_type

```python
@property
def mime_type() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:209](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L209)

MIME type of the image (e.g., “image/jpeg”, “image/png”).

## BidiConnectionStartEvent

```python
class BidiConnectionStartEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:219](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L219)

Streaming connection established and ready for interaction.

**Arguments**:

-   `connection_id` - Unique identifier for this streaming connection.
-   `model` - Model identifier (e.g., “gpt-realtime”, “gemini-2.0-flash-live”).

#### \_\_init\_\_

```python
def __init__(connection_id: str, model: str)
```

Defined in: [src/strands/experimental/bidi/types/events.py:227](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L227)

Initialize connection start event.

#### connection\_id

```python
@property
def connection_id() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:238](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L238)

Unique identifier for this streaming connection.

#### model

```python
@property
def model() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:243](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L243)

Model identifier (e.g., ‘gpt-realtime’, ‘gemini-2.0-flash-live’).

## BidiConnectionRestartEvent

```python
class BidiConnectionRestartEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:248](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L248)

Agent is restarting the model connection after timeout.

#### \_\_init\_\_

```python
def __init__(timeout_error: "BidiModelTimeoutError")
```

Defined in: [src/strands/experimental/bidi/types/events.py:251](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L251)

Initialize.

**Arguments**:

-   `timeout_error` - Timeout error reported by the model.

#### timeout\_error

```python
@property
def timeout_error() -> "BidiModelTimeoutError"
```

Defined in: [src/strands/experimental/bidi/types/events.py:265](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L265)

Model timeout error.

## BidiResponseStartEvent

```python
class BidiResponseStartEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:270](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L270)

Model starts generating a response.

**Arguments**:

-   `response_id` - Unique identifier for this response (used in response.complete).

#### \_\_init\_\_

```python
def __init__(response_id: str)
```

Defined in: [src/strands/experimental/bidi/types/events.py:277](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L277)

Initialize response start event.

#### response\_id

```python
@property
def response_id() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:282](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L282)

Unique identifier for this response.

## BidiAudioStreamEvent

```python
class BidiAudioStreamEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:287](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L287)

Streaming audio output from the model.

**Arguments**:

-   `audio` - Base64-encoded audio string.
-   `format` - Audio encoding format.
-   `sample_rate` - Number of audio samples per second in Hz.
-   `channels` - Number of audio channels (1=mono, 2=stereo).

#### \_\_init\_\_

```python
def __init__(audio: str, format: AudioFormat, sample_rate: AudioSampleRate,
             channels: AudioChannel)
```

Defined in: [src/strands/experimental/bidi/types/events.py:297](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L297)

Initialize audio stream event.

#### audio

```python
@property
def audio() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:316](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L316)

Base64-encoded audio string.

#### format

```python
@property
def format() -> AudioFormat
```

Defined in: [src/strands/experimental/bidi/types/events.py:321](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L321)

Audio encoding format.

#### sample\_rate

```python
@property
def sample_rate() -> AudioSampleRate
```

Defined in: [src/strands/experimental/bidi/types/events.py:326](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L326)

Number of audio samples per second in Hz.

#### channels

```python
@property
def channels() -> AudioChannel
```

Defined in: [src/strands/experimental/bidi/types/events.py:331](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L331)

Number of audio channels (1=mono, 2=stereo).

## BidiTranscriptStreamEvent

```python
class BidiTranscriptStreamEvent(ModelStreamEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:336](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L336)

Audio transcription streaming (user or assistant speech).

Supports incremental transcript updates for providers that send partial transcripts before the final version.

**Arguments**:

-   `delta` - The incremental transcript change (ContentBlockDelta).
-   `text` - The delta text (same as delta content for convenience).
-   `role` - Who is speaking (“user” or “assistant”).
-   `is_final` - Whether this is the final/complete transcript.
-   `current_transcript` - The accumulated transcript text so far (None for first delta).

#### \_\_init\_\_

```python
def __init__(delta: ContentBlockDelta,
             text: str,
             role: Role,
             is_final: bool,
             current_transcript: str | None = None)
```

Defined in: [src/strands/experimental/bidi/types/events.py:350](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L350)

Initialize transcript stream event.

#### delta

```python
@property
def delta() -> ContentBlockDelta
```

Defined in: [src/strands/experimental/bidi/types/events.py:371](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L371)

The incremental transcript change.

#### text

```python
@property
def text() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:376](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L376)

The text content to send to the model.

#### role

```python
@property
def role() -> Role
```

Defined in: [src/strands/experimental/bidi/types/events.py:381](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L381)

The role of the message sender.

#### is\_final

```python
@property
def is_final() -> bool
```

Defined in: [src/strands/experimental/bidi/types/events.py:386](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L386)

Whether this is the final/complete transcript.

#### current\_transcript

```python
@property
def current_transcript() -> str | None
```

Defined in: [src/strands/experimental/bidi/types/events.py:391](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L391)

The accumulated transcript text so far.

## BidiInterruptionEvent

```python
class BidiInterruptionEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:396](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L396)

Model generation was interrupted.

**Arguments**:

-   `reason` - Why the interruption occurred.

#### \_\_init\_\_

```python
def __init__(reason: Literal["user_speech", "error"])
```

Defined in: [src/strands/experimental/bidi/types/events.py:403](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L403)

Initialize interruption event.

#### reason

```python
@property
def reason() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:413](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L413)

Why the interruption occurred.

## BidiResponseCompleteEvent

```python
class BidiResponseCompleteEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:418](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L418)

Model finished generating response.

**Arguments**:

-   `response_id` - ID of the response that completed (matches response.start).
-   `stop_reason` - Why the response ended.

#### \_\_init\_\_

```python
def __init__(response_id: str, stop_reason: StopReason)
```

Defined in: [src/strands/experimental/bidi/types/events.py:426](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L426)

Initialize response complete event.

#### response\_id

```python
@property
def response_id() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:441](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L441)

Unique identifier for this response.

#### stop\_reason

```python
@property
def stop_reason() -> StopReason
```

Defined in: [src/strands/experimental/bidi/types/events.py:446](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L446)

Why the response ended.

## ModalityUsage

```python
class ModalityUsage(dict)
```

Defined in: [src/strands/experimental/bidi/types/events.py:451](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L451)

Token usage for a specific modality.

**Attributes**:

-   `modality` - Type of content.
-   `input_tokens` - Tokens used for this modality’s input.
-   `output_tokens` - Tokens used for this modality’s output.

## BidiUsageEvent

```python
class BidiUsageEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:465](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L465)

Token usage event with modality breakdown for bidirectional streaming.

Tracks token consumption across different modalities (audio, text, images) during bidirectional streaming sessions.

**Arguments**:

-   `input_tokens` - Total tokens used for all input modalities.
-   `output_tokens` - Total tokens used for all output modalities.
-   `total_tokens` - Sum of input and output tokens.
-   `modality_details` - Optional list of token usage per modality.
-   `cache_read_input_tokens` - Optional tokens read from cache.
-   `cache_write_input_tokens` - Optional tokens written to cache.

#### \_\_init\_\_

```python
def __init__(input_tokens: int,
             output_tokens: int,
             total_tokens: int,
             modality_details: list[ModalityUsage] | None = None,
             cache_read_input_tokens: int | None = None,
             cache_write_input_tokens: int | None = None)
```

Defined in: [src/strands/experimental/bidi/types/events.py:480](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L480)

Initialize usage event.

#### input\_tokens

```python
@property
def input_tokens() -> int
```

Defined in: [src/strands/experimental/bidi/types/events.py:505](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L505)

Total tokens used for all input modalities.

#### output\_tokens

```python
@property
def output_tokens() -> int
```

Defined in: [src/strands/experimental/bidi/types/events.py:510](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L510)

Total tokens used for all output modalities.

#### total\_tokens

```python
@property
def total_tokens() -> int
```

Defined in: [src/strands/experimental/bidi/types/events.py:515](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L515)

Sum of input and output tokens.

#### modality\_details

```python
@property
def modality_details() -> list[ModalityUsage]
```

Defined in: [src/strands/experimental/bidi/types/events.py:520](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L520)

Optional list of token usage per modality.

#### cache\_read\_input\_tokens

```python
@property
def cache_read_input_tokens() -> int | None
```

Defined in: [src/strands/experimental/bidi/types/events.py:525](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L525)

Optional tokens read from cache.

#### cache\_write\_input\_tokens

```python
@property
def cache_write_input_tokens() -> int | None
```

Defined in: [src/strands/experimental/bidi/types/events.py:530](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L530)

Optional tokens written to cache.

## BidiConnectionCloseEvent

```python
class BidiConnectionCloseEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:535](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L535)

Streaming connection closed.

**Arguments**:

-   `connection_id` - Unique identifier for this streaming connection (matches BidiConnectionStartEvent).
-   `reason` - Why the connection was closed.

#### \_\_init\_\_

```python
def __init__(connection_id: str,
             reason: Literal["client_disconnect", "timeout", "error",
                             "complete", "user_request"])
```

Defined in: [src/strands/experimental/bidi/types/events.py:543](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L543)

Initialize connection close event.

#### connection\_id

```python
@property
def connection_id() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:558](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L558)

Unique identifier for this streaming connection.

#### reason

```python
@property
def reason() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:563](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L563)

Why the interruption occurred.

## BidiErrorEvent

```python
class BidiErrorEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:568](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L568)

Error occurred during the session.

Stores the full Exception object as an instance attribute for debugging while keeping the event dict JSON-serializable. The exception can be accessed via the `error` property for re-raising or type-based error handling.

**Arguments**:

-   `error` - The exception that occurred.
-   `details` - Optional additional error information.

#### \_\_init\_\_

```python
def __init__(error: Exception, details: dict[str, Any] | None = None)
```

Defined in: [src/strands/experimental/bidi/types/events.py:580](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L580)

Initialize error event.

#### error

```python
@property
def error() -> Exception
```

Defined in: [src/strands/experimental/bidi/types/events.py:599](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L599)

The original exception that occurred.

Can be used for re-raising or type-based error handling.

#### code

```python
@property
def code() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:607](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L607)

Error code derived from exception class name.

#### message

```python
@property
def message() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:612](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L612)

Human-readable error message from the exception.

#### details

```python
@property
def details() -> dict[str, Any] | None
```

Defined in: [src/strands/experimental/bidi/types/events.py:617](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L617)

Additional error context beyond the exception itself.

#### BidiInputEvent

Union of different bidi input event types.

#### BidiOutputEvent

Union of different bidi output event types.