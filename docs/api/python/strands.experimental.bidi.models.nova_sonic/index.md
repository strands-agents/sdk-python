Nova Sonic bidirectional model provider for real-time streaming conversations.

Implements the BidiModel interface for Amazon’s Nova Sonic, handling the complex event sequencing and audio processing required by Nova Sonic’s InvokeModelWithBidirectionalStream protocol.

Nova Sonic specifics:

-   Hierarchical event sequences: connectionStart → promptStart → content streaming
-   Base64-encoded audio format with hex encoding
-   Tool execution with content containers and identifier tracking
-   8-minute connection limits with proper cleanup sequences
-   Interruption detection through stopReason events

Note, BidiNovaSonicModel is only supported for Python 3.12+

## BidiNovaSonicModel

```python
class BidiNovaSonicModel(BidiModel)
```

Defined in: [src/strands/experimental/bidi/models/nova\_sonic.py:107](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/nova_sonic.py#L107)

Nova Sonic implementation for bidirectional streaming.

Combines model configuration and connection state in a single class. Manages Nova Sonic’s complex event sequencing, audio format conversion, and tool execution patterns while providing the standard BidiModel interface.

Note, BidiNovaSonicModel is only supported for Python 3.12+.

**Attributes**:

-   `_stream` - open bedrock stream to nova sonic.

#### \_\_init\_\_

```python
def __init__(model_id: str = NOVA_SONIC_V2_MODEL_ID,
             provider_config: dict[str, Any] | None = None,
             client_config: dict[str, Any] | None = None,
             **kwargs: Any) -> None
```

Defined in: [src/strands/experimental/bidi/models/nova\_sonic.py:122](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/nova_sonic.py#L122)

Initialize Nova Sonic bidirectional model.

**Arguments**:

-   `model_id` - Model identifier (default: amazon.nova-2-sonic-v1:0)
-   `provider_config` - Model behavior configuration including:
    -   audio: Audio input/output settings (sample rate, voice, etc.)
    -   inference: Model inference settings (max\_tokens, temperature, top\_p)
    -   turn\_detection: Turn detection configuration (v2 only feature)
    -   endpointingSensitivity: “HIGH” | “MEDIUM” | “LOW” (optional)
-   `client_config` - AWS authentication (boto\_session OR region, not both)
-   `**kwargs` - Reserved for future parameters.

**Raises**:

-   `ValueError` - If turn\_detection is used with v1 model.
-   `ValueError` - If endpointingSensitivity is not HIGH, MEDIUM, or LOW.
-   `ValueError` - If the resolved AWS region is not a valid region identifier.

#### start

```python
async def start(system_prompt: str | None = None,
                tools: list[ToolSpec] | None = None,
                messages: Messages | None = None,
                **kwargs: Any) -> None
```

Defined in: [src/strands/experimental/bidi/models/nova\_sonic.py:228](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/nova_sonic.py#L228)

Establish bidirectional connection to Nova Sonic.

**Arguments**:

-   `system_prompt` - System instructions for the model.
-   `tools` - List of tools available to the model.
-   `messages` - Conversation history to initialize with.
-   `**kwargs` - Additional configuration options.

**Raises**:

-   `RuntimeError` - If user calls start again without first stopping.

#### receive

```python
async def receive() -> AsyncGenerator[BidiOutputEvent, None]
```

Defined in: [src/strands/experimental/bidi/models/nova\_sonic.py:362](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/nova_sonic.py#L362)

Receive Nova Sonic events and convert to provider-agnostic format.

**Raises**:

-   `RuntimeError` - If start has not been called.

#### send

```python
async def send(content: BidiInputEvent | ToolResultEvent) -> None
```

Defined in: [src/strands/experimental/bidi/models/nova\_sonic.py:409](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/nova_sonic.py#L409)

Unified send method for all content types. Sends the given content to Nova Sonic.

Dispatches to appropriate internal handler based on content type.

**Arguments**:

-   `content` - Input event.

**Raises**:

-   `ValueError` - If content type not supported (e.g., image content).

#### stop

```python
async def stop() -> None
```

Defined in: [src/strands/experimental/bidi/models/nova\_sonic.py:555](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/nova_sonic.py#L555)

Close Nova Sonic connection with proper cleanup sequence.