Gemini Live API bidirectional model provider using official Google GenAI SDK.

Implements the BidiModel interface for Google’s Gemini Live API using the official Google GenAI SDK for simplified and robust WebSocket communication.

Key improvements over custom WebSocket implementation:

-   Uses official google-genai SDK with native Live API support
-   Simplified session management with client.aio.live.connect()
-   Built-in tool integration and event handling
-   Automatic WebSocket connection management and error handling
-   Native support for audio/text streaming and interruption

## BidiGeminiLiveModel

```python
class BidiGeminiLiveModel(BidiModel)
```

Defined in: [src/strands/experimental/bidi/models/gemini\_live.py:54](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/gemini_live.py#L54)

Gemini Live API implementation using official Google GenAI SDK.

Combines model configuration and connection state in a single class. Provides a clean interface to Gemini Live API using the official SDK, eliminating custom WebSocket handling and providing robust error handling.

#### \_\_init\_\_

```python
def __init__(model_id: str = "gemini-2.5-flash-native-audio-preview-09-2025",
             provider_config: dict[str, Any] | None = None,
             client_config: dict[str, Any] | None = None,
             **kwargs: Any)
```

Defined in: [src/strands/experimental/bidi/models/gemini\_live.py:62](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/gemini_live.py#L62)

Initialize Gemini Live API bidirectional model.

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

Defined in: [src/strands/experimental/bidi/models/gemini\_live.py:134](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/gemini_live.py#L134)

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

Defined in: [src/strands/experimental/bidi/models/gemini\_live.py:199](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/gemini_live.py#L199)

Receive Gemini Live API events and convert to provider-agnostic format.

#### send

```python
async def send(content: BidiInputEvent | ToolResultEvent) -> None
```

Defined in: [src/strands/experimental/bidi/models/gemini\_live.py:384](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/gemini_live.py#L384)

Unified send method for all content types. Sends the given inputs to Google Live API.

Dispatches to appropriate internal handler based on content type.

**Arguments**:

-   `content` - Typed event (BidiTextInputEvent, BidiAudioInputEvent, BidiImageInputEvent, or ToolResultEvent).

**Raises**:

-   `ValueError` - If content type not supported (e.g., image content).

#### stop

```python
async def stop() -> None
```

Defined in: [src/strands/experimental/bidi/models/gemini\_live.py:483](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/gemini_live.py#L483)

Close Gemini Live API connection.