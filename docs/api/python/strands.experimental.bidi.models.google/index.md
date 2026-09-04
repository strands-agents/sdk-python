Google Gemini Live model provider using the Gemini Live API and official Google GenAI SDK.

Implements the BidiModel interface for Google’s Gemini Live API using the official Google GenAI SDK for simplified and robust WebSocket communication.

Key improvements over custom WebSocket implementation:

-   Uses official google-genai SDK with native Live API support
-   Simplified session management with client.aio.live.connect()
-   Built-in tool integration and event handling
-   Automatic WebSocket connection management and error handling
-   Native support for audio/text streaming and interruption

## GoogleGeminiLiveModel

```python
class GoogleGeminiLiveModel(BidiModel)
```

Defined in: [src/strands/experimental/bidi/models/google.py:72](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/google.py#L72)

Google Gemini Live implementation using the official Google GenAI SDK.

Combines model configuration and connection state in a single class. Provides a clean interface to Gemini Live API using the official SDK, eliminating custom WebSocket handling and providing robust error handling.

#### \_\_init\_\_

```python
def __init__(model_id: str = "gemini-2.5-flash-native-audio-preview-09-2025",
             provider_config: dict[str, Any] | None = None,
             client_config: dict[str, Any] | None = None,
             **kwargs: Any)
```

Defined in: [src/strands/experimental/bidi/models/google.py:80](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/google.py#L80)

Initialize the Google Gemini Live bidirectional model.

**Arguments**:

-   `model_id` - Model identifier (default: gemini-2.5-flash-native-audio-preview-09-2025)
-   `provider_config` - Model behavior (audio, inference)
-   `client_config` - Authentication (api\_key, http\_options)
-   `**kwargs` - Reserved for future parameters.

#### start

```python
async def start(system_prompt: str | None = None,
                tools: list[ToolSpec] | None = None,
                messages: Messages | None = None,
                **kwargs: Any) -> None
```

Defined in: [src/strands/experimental/bidi/models/google.py:168](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/google.py#L168)

Establish bidirectional connection with Gemini Live API.

**Arguments**:

-   `system_prompt` - System instructions for the model.
-   `tools` - List of tools available to the model.
-   `messages` - Conversation history to initialize with.
-   `**kwargs` - Additional configuration options.

#### receive

```python
async def receive() -> AsyncGenerator[BidiOutputEvent, None]
```

Defined in: [src/strands/experimental/bidi/models/google.py:239](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/google.py#L239)

Receive Gemini Live API events and convert to provider-agnostic format.

#### send

```python
async def send(content: BidiInputEvent | ToolResultEvent) -> None
```

Defined in: [src/strands/experimental/bidi/models/google.py:491](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/google.py#L491)

Unified send method for all content types. Sends the given inputs to the Gemini Live API.

Dispatches to appropriate internal handler based on content type.

**Arguments**:

-   `content` - Typed event (BidiTextInputEvent, BidiAudioInputEvent, BidiImageInputEvent, or ToolResultEvent).

**Raises**:

-   `ValueError` - If content type not supported (e.g., image content).

#### stop

```python
async def stop() -> None
```

Defined in: [src/strands/experimental/bidi/models/google.py:590](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/google.py#L590)

Close Gemini Live API connection.

#### restart

```python
async def restart(system_prompt: str | None = None,
                  tools: list[ToolSpec] | None = None,
                  messages: Messages | None = None,
                  **restart_kwargs: Any) -> None
```

Defined in: [src/strands/experimental/bidi/models/google.py:610](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/google.py#L610)

Restart by closing the connection and resuming the same session via its handle.

Resumes the Gemini session using the last resumption handle so server-side context carries across the swap without replaying history. The handle is supplied by the reactive (GoAway) path via `restart_kwargs` or read from the tracked handle on the proactive path. When no handle is available yet, falls back to a fresh connection with history replay.

**Arguments**:

-   `system_prompt` - System instructions for the resumed connection.
-   `tools` - Tool specifications for the resumed connection.
-   `messages` - Conversation history, replayed only when resuming without a handle.
-   `**restart_kwargs` - Provider restart options; `live_session_handle` resumes the session.