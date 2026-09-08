Bidirectional streaming model interface.

Defines the abstract interface for models that support real-time bidirectional communication with persistent connections. Unlike traditional request-response models, bidirectional models maintain an open connection for streaming audio, text, and tool interactions.

Features:

-   Persistent connection management with connect/close lifecycle
-   Real-time bidirectional communication (send and receive simultaneously)
-   Provider-agnostic event normalization
-   Support for audio, text, image, and tool result streaming

## BidiModelConfig

```python
class BidiModelConfig(TypedDict)
```

Defined in: [src/strands/experimental/bidi/models/model.py:31](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L31)

Configuration shared by bidirectional model providers.

**Attributes**:

-   `model_id` - Provider model identifier.
-   `params` - Provider-specific keyword arguments passed to the model request or session.
-   `connection` - Reconnect timing overrides.

## Restartable

```python
@runtime_checkable
class Restartable(Protocol)
```

Defined in: [src/strands/experimental/bidi/models/model.py:46](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L46)

A bidirectional model that can replace its active connection while preserving context.

#### restart

```python
async def restart(system_prompt: str | None = None,
                  tools: list[ToolSpec] | None = None,
                  messages: Messages | None = None,
                  **restart_kwargs: Any) -> None
```

Defined in: [src/strands/experimental/bidi/models/model.py:49](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L49)

Replace the active connection while preserving conversation context.

**Arguments**:

-   `system_prompt` - System instructions for the new connection.
-   `tools` - Tool specifications for the new connection.
-   `messages` - Conversation history to replay when required by the provider.
-   `**restart_kwargs` - Provider-specific restart options.

## BidiModel

```python
class BidiModel(Model, abc.ABC)
```

Defined in: [src/strands/experimental/bidi/models/model.py:67](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L67)

Abstract base class for bidirectional streaming models.

This interface defines the contract for models that support persistent streaming connections with real-time audio and text communication. Implementations handle provider-specific protocols while exposing a standardized event-based API.

**Attributes**:

-   `config` - Configuration dictionary with provider-specific settings.
-   `model_id` - Provider model identifier.
-   `connection_config` - Declared connection limit and reconnect timing. Providers that support proactive reconnect populate this; an empty config means reactive-only behavior.
-   `usage_is_cumulative` - Whether the provider reports cumulative connection token totals (True) rather than per-response deltas (False, the default when absent). Providers reporting deltas may omit it.

#### update\_config

```python
def update_config(**model_config: Any) -> None
```

Defined in: [src/strands/experimental/bidi/models/model.py:90](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L90)

Update the model configuration with the provided arguments.

**Arguments**:

-   `**model_config` - Configuration overrides.

#### get\_config

```python
def get_config() -> dict[str, Any]
```

Defined in: [src/strands/experimental/bidi/models/model.py:98](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L98)

Return a copy of the model configuration.

#### structured\_output

```python
def structured_output(*args: Any, **kwargs: Any) -> NoReturn
```

Defined in: [src/strands/experimental/bidi/models/model.py:102](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L102)

Raise because bidirectional models do not support structured output.

#### stream

```python
def stream(*args: Any, **kwargs: Any) -> NoReturn
```

Defined in: [src/strands/experimental/bidi/models/model.py:106](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L106)

Raise because bidirectional models use their persistent streaming API.

#### start

```python
@abc.abstractmethod
async def start(system_prompt: str | None = None,
                tools: list[ToolSpec] | None = None,
                messages: Messages | None = None,
                **kwargs: Any) -> None
```

Defined in: [src/strands/experimental/bidi/models/model.py:112](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L112)

Establish a persistent streaming connection with the model.

Opens a bidirectional connection that remains active for real-time communication. The connection supports concurrent sending and receiving of events until explicitly closed. Must be called before any send() or receive() operations.

**Arguments**:

-   `system_prompt` - System instructions to configure model behavior.
-   `tools` - Tool specifications that the model can invoke during the conversation.
-   `messages` - Initial conversation history to provide context.
-   `**kwargs` - Provider-specific configuration options.

#### stop

```python
@abc.abstractmethod
async def stop() -> None
```

Defined in: [src/strands/experimental/bidi/models/model.py:135](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L135)

Close the streaming connection and release resources.

Terminates the active bidirectional connection and cleans up any associated resources such as network connections, buffers, or background tasks. After calling close(), the model instance cannot be used until start() is called again.

#### receive

```python
@abc.abstractmethod
def receive() -> AsyncIterable[BidiOutputEvent]
```

Defined in: [src/strands/experimental/bidi/models/model.py:146](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L146)

Receive streaming events from the model.

Continuously yields events from the model as they arrive over the connection. Events are normalized to a provider-agnostic format for uniform processing. This method should be called in a loop or async task to process model responses.

The stream continues until the connection is closed or an error occurs.

**Yields**:

-   `BidiOutputEvent` - Standardized event objects containing audio output, transcripts, tool calls, or control signals.

#### send

```python
@abc.abstractmethod
async def send(content: BidiInputEvent | ToolResultEvent) -> None
```

Defined in: [src/strands/experimental/bidi/models/model.py:163](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L163)

Send content to the model over the active connection.

Transmits user input or tool results to the model during an active streaming session. Supports multiple content types including text, audio, images, and tool execution results. Can be called multiple times during a conversation.

**Arguments**:

-   `content` - The content to send. Must be one of:
    
    -   BidiTextInputEvent: Text message from the user
    -   BidiAudioInputEvent: Audio data for speech input
    -   BidiImageInputEvent: Image data for visual understanding
    -   ToolResultEvent: Result from a tool execution

**Example**:

```plaintext
await model.send(BidiTextInputEvent(text="Hello", role="user"))
await model.send(BidiAudioInputEvent(audio=bytes, format="pcm", sample_rate=16000, channels=1))
await model.send(BidiImageInputEvent(image=bytes, mime_type="image/jpeg", encoding="raw"))
await model.send(ToolResultEvent(tool_result))
```

## BidiModelTimeoutError

```python
class BidiModelTimeoutError(Exception)
```

Defined in: [src/strands/experimental/bidi/models/model.py:192](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L192)

Model timeout error.

Bidirectional models are often configured with a connection time limit. Bedrock Nova Sonic, for example, keeps the connection open for 8 minutes max. Upon receiving a timeout, the agent loop is configured to restart the model connection so as to create a seamless, uninterrupted experience for the user.

#### \_\_init\_\_

```python
def __init__(message: str, **restart_config: Any) -> None
```

Defined in: [src/strands/experimental/bidi/models/model.py:200](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L200)

Initialize error.

**Arguments**:

-   `message` - Timeout message from model.
-   `**restart_config` - Configure restart specific behaviors in the call to model start.

## AudioCapable

```python
@runtime_checkable
class AudioCapable(Protocol)
```

Defined in: [src/strands/experimental/bidi/models/model.py:213](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L213)

Protocol for models that support audio input and output.

#### audio\_config

```python
@property
def audio_config() -> AudioConfig
```

Defined in: [src/strands/experimental/bidi/models/model.py:217](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/model.py#L217)

Get the resolved audio configuration.