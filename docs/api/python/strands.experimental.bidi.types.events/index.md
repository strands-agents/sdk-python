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

Agent is restarting the model connection.

Emitted on both reconnect paths: reactively after the model reports a timeout, and proactively when the reconnect timer fires ahead of the provider’s limit.

**Arguments**:

-   `reason` - What triggered the restart (“timeout” reactively, “scheduled” proactively).
-   `timeout_error` - The model’s timeout error on the reactive path; None when scheduled.
-   `turn_interrupted` - True if the restart cut an in-progress or owed turn (the alignment wait could not complete it before the deadline, or a timeout struck mid-turn). The provider replays history as context, so that turn will not be answered on its own — an app can re-prompt or notify the user when this is set.

#### \_\_init\_\_

```python
def __init__(reason: Literal["timeout", "scheduled"],
             timeout_error: "BidiModelTimeoutError | None" = None,
             turn_interrupted: bool = False)
```

Defined in: [src/strands/experimental/bidi/types/events.py:263](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L263)

Initialize connection restart event.

#### reason

```python
@property
def reason() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:280](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L280)

What triggered the restart (“timeout” or “scheduled”).

#### timeout\_error

```python
@property
def timeout_error() -> "BidiModelTimeoutError | None"
```

Defined in: [src/strands/experimental/bidi/types/events.py:285](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L285)

Model timeout error on the reactive path; None when scheduled.

#### turn\_interrupted

```python
@property
def turn_interrupted() -> bool
```

Defined in: [src/strands/experimental/bidi/types/events.py:290](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L290)

True if the restart cut an in-progress or owed turn that will not be answered.

## BidiConnectionWarningEvent

```python
class BidiConnectionWarningEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:295](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L295)

Agent is approaching a proactive reconnect.

Emitted by the proactive reconnect timer before a reconnect; informational only.

**Arguments**:

-   `time_left_s` - Approximate seconds until the scheduled reconnect.

#### \_\_init\_\_

```python
def __init__(time_left_s: float)
```

Defined in: [src/strands/experimental/bidi/types/events.py:304](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L304)

Initialize connection warning event.

#### time\_left\_s

```python
@property
def time_left_s() -> float
```

Defined in: [src/strands/experimental/bidi/types/events.py:314](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L314)

Approximate seconds until the scheduled reconnect.

## BidiResponseStartEvent

```python
class BidiResponseStartEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:319](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L319)

Model starts generating a response.

**Arguments**:

-   `response_id` - Unique identifier for this response (used in response.complete).

#### \_\_init\_\_

```python
def __init__(response_id: str)
```

Defined in: [src/strands/experimental/bidi/types/events.py:326](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L326)

Initialize response start event.

#### response\_id

```python
@property
def response_id() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:331](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L331)

Unique identifier for this response.

## BidiAudioStreamEvent

```python
class BidiAudioStreamEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:336](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L336)

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

Defined in: [src/strands/experimental/bidi/types/events.py:346](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L346)

Initialize audio stream event.

#### audio

```python
@property
def audio() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:365](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L365)

Base64-encoded audio string.

#### format

```python
@property
def format() -> AudioFormat
```

Defined in: [src/strands/experimental/bidi/types/events.py:370](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L370)

Audio encoding format.

#### sample\_rate

```python
@property
def sample_rate() -> AudioSampleRate
```

Defined in: [src/strands/experimental/bidi/types/events.py:375](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L375)

Number of audio samples per second in Hz.

#### channels

```python
@property
def channels() -> AudioChannel
```

Defined in: [src/strands/experimental/bidi/types/events.py:380](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L380)

Number of audio channels (1=mono, 2=stereo).

## BidiTranscriptStreamEvent

```python
class BidiTranscriptStreamEvent(ModelStreamEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:385](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L385)

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

Defined in: [src/strands/experimental/bidi/types/events.py:399](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L399)

Initialize transcript stream event.

#### delta

```python
@property
def delta() -> ContentBlockDelta
```

Defined in: [src/strands/experimental/bidi/types/events.py:420](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L420)

The incremental transcript change.

#### text

```python
@property
def text() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:425](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L425)

The text content to send to the model.

#### role

```python
@property
def role() -> Role
```

Defined in: [src/strands/experimental/bidi/types/events.py:430](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L430)

The role of the message sender.

#### is\_final

```python
@property
def is_final() -> bool
```

Defined in: [src/strands/experimental/bidi/types/events.py:435](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L435)

Whether this is the final/complete transcript.

#### current\_transcript

```python
@property
def current_transcript() -> str | None
```

Defined in: [src/strands/experimental/bidi/types/events.py:440](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L440)

The accumulated transcript text so far.

## BidiInterruptionEvent

```python
class BidiInterruptionEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:445](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L445)

Model generation was interrupted.

**Arguments**:

-   `reason` - Why the interruption occurred.

#### \_\_init\_\_

```python
def __init__(reason: Literal["user_speech", "error"])
```

Defined in: [src/strands/experimental/bidi/types/events.py:452](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L452)

Initialize interruption event.

#### reason

```python
@property
def reason() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:462](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L462)

Why the interruption occurred.

## BidiResponseCompleteEvent

```python
class BidiResponseCompleteEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:467](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L467)

Model finished generating response.

**Arguments**:

-   `response_id` - ID of the response that completed (matches response.start).
-   `stop_reason` - Why the response ended.

#### \_\_init\_\_

```python
def __init__(response_id: str, stop_reason: StopReason)
```

Defined in: [src/strands/experimental/bidi/types/events.py:475](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L475)

Initialize response complete event.

#### response\_id

```python
@property
def response_id() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:490](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L490)

Unique identifier for this response.

#### stop\_reason

```python
@property
def stop_reason() -> StopReason
```

Defined in: [src/strands/experimental/bidi/types/events.py:495](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L495)

Why the response ended.

## ModalityUsage

```python
class ModalityUsage(dict)
```

Defined in: [src/strands/experimental/bidi/types/events.py:500](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L500)

Token usage for a specific modality.

**Attributes**:

-   `modality` - Type of content.
-   `input_tokens` - Tokens used for this modality’s input.
-   `output_tokens` - Tokens used for this modality’s output.

## BidiUsageEvent

```python
class BidiUsageEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:514](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L514)

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

Defined in: [src/strands/experimental/bidi/types/events.py:529](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L529)

Initialize usage event.

#### input\_tokens

```python
@property
def input_tokens() -> int
```

Defined in: [src/strands/experimental/bidi/types/events.py:554](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L554)

Total tokens used for all input modalities.

#### output\_tokens

```python
@property
def output_tokens() -> int
```

Defined in: [src/strands/experimental/bidi/types/events.py:559](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L559)

Total tokens used for all output modalities.

#### total\_tokens

```python
@property
def total_tokens() -> int
```

Defined in: [src/strands/experimental/bidi/types/events.py:564](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L564)

Sum of input and output tokens.

#### modality\_details

```python
@property
def modality_details() -> list[ModalityUsage]
```

Defined in: [src/strands/experimental/bidi/types/events.py:569](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L569)

Optional list of token usage per modality.

#### cache\_read\_input\_tokens

```python
@property
def cache_read_input_tokens() -> int | None
```

Defined in: [src/strands/experimental/bidi/types/events.py:574](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L574)

Optional tokens read from cache.

#### cache\_write\_input\_tokens

```python
@property
def cache_write_input_tokens() -> int | None
```

Defined in: [src/strands/experimental/bidi/types/events.py:579](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L579)

Optional tokens written to cache.

## BidiConnectionCloseEvent

```python
class BidiConnectionCloseEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:584](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L584)

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

Defined in: [src/strands/experimental/bidi/types/events.py:592](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L592)

Initialize connection close event.

#### connection\_id

```python
@property
def connection_id() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:607](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L607)

Unique identifier for this streaming connection.

#### reason

```python
@property
def reason() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:612](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L612)

Why the interruption occurred.

## BidiErrorEvent

```python
class BidiErrorEvent(TypedEvent)
```

Defined in: [src/strands/experimental/bidi/types/events.py:617](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L617)

Error occurred during the session.

Stores the full Exception object as an instance attribute for debugging while keeping the event dict JSON-serializable. The exception can be accessed via the `error` property for re-raising or type-based error handling.

**Arguments**:

-   `error` - The exception that occurred.
-   `details` - Optional additional error information.

#### \_\_init\_\_

```python
def __init__(error: Exception, details: dict[str, Any] | None = None)
```

Defined in: [src/strands/experimental/bidi/types/events.py:629](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L629)

Initialize error event.

#### error

```python
@property
def error() -> Exception
```

Defined in: [src/strands/experimental/bidi/types/events.py:648](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L648)

The original exception that occurred.

Can be used for re-raising or type-based error handling.

#### code

```python
@property
def code() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:656](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L656)

Error code derived from exception class name.

#### message

```python
@property
def message() -> str
```

Defined in: [src/strands/experimental/bidi/types/events.py:661](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L661)

Human-readable error message from the exception.

#### details

```python
@property
def details() -> dict[str, Any] | None
```

Defined in: [src/strands/experimental/bidi/types/events.py:666](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/events.py#L666)

Additional error context beyond the exception itself.

#### BidiInputEvent

Union of different bidi input event types.

#### BidiOutputEvent

Union of different bidi output event types.