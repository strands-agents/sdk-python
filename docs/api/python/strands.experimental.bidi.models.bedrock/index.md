Amazon Bedrock Nova Sonic provider for real-time streaming conversations.

Implements the BidiModel interface for Amazon’s Nova Sonic, handling the complex event sequencing and audio processing required by Nova Sonic’s InvokeModelWithBidirectionalStream protocol.

Nova Sonic specifics:

-   Hierarchical event sequences: connectionStart → promptStart → content streaming
-   Base64-encoded audio format with hex encoding
-   Tool execution with content containers and identifier tracking
-   8-minute connection limits with proper cleanup sequences
-   Interruption detection through stopReason events

Note, BedrockNovaSonicModel is only supported for Python 3.12+

## BedrockNovaSonicModel

```python
class BedrockNovaSonicModel(BidiModel, AudioCapable)
```

Defined in: [src/strands/experimental/bidi/models/bedrock.py:89](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/bedrock.py#L89)

Amazon Bedrock Nova Sonic implementation for bidirectional streaming.

Combines model configuration and connection state in a single class. Manages Nova Sonic’s complex event sequencing, audio format conversion, and tool execution patterns while providing the standard BidiModel interface.

Note, BedrockNovaSonicModel is only supported for Python 3.12+.

**Attributes**:

-   `_stream` - open bedrock stream to nova sonic.

#### \_\_init\_\_

```python
def __init__(*,
             boto_session: Session | None = None,
             region: str | None = None,
             audio: AudioConfig | None = None,
             **model_config: Unpack[BidiModelConfig]) -> None
```

Defined in: [src/strands/experimental/bidi/models/bedrock.py:104](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/bedrock.py#L104)

Initialize Nova Sonic bidirectional model.

**Arguments**:

-   `boto_session` - Boto3 session used to resolve credentials and region.
-   `region` - AWS region. Cannot be combined with `boto_session`.
-   `audio` - Audio configuration.
-   `**model_config` - Model configuration.

**Raises**:

-   `ValueError` - If both `boto_session` and `region` are provided or the resolved region is invalid.

#### audio\_config

```python
@property
def audio_config() -> AudioConfig
```

Defined in: [src/strands/experimental/bidi/models/bedrock.py:163](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/bedrock.py#L163)

Get the resolved audio configuration.

#### start

```python
async def start(system_prompt: str | None = None,
                tools: list[ToolSpec] | None = None,
                messages: Messages | None = None,
                **kwargs: Any) -> None
```

Defined in: [src/strands/experimental/bidi/models/bedrock.py:167](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/bedrock.py#L167)

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

Defined in: [src/strands/experimental/bidi/models/bedrock.py:304](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/bedrock.py#L304)

Receive Nova Sonic events and convert to provider-agnostic format.

**Raises**:

-   `RuntimeError` - If start has not been called.

#### send

```python
async def send(content: BidiInputEvent | ToolResultEvent) -> None
```

Defined in: [src/strands/experimental/bidi/models/bedrock.py:355](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/bedrock.py#L355)

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

Defined in: [src/strands/experimental/bidi/models/bedrock.py:501](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/bedrock.py#L501)

Close Nova Sonic connection with proper cleanup sequence.

#### restart

```python
async def restart(system_prompt: str | None = None,
                  tools: list[ToolSpec] | None = None,
                  messages: Messages | None = None,
                  **restart_kwargs: Any) -> None
```

Defined in: [src/strands/experimental/bidi/models/bedrock.py:538](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/models/bedrock.py#L538)

Restart by closing the connection and starting a new one, replaying messages.

**Arguments**:

-   `system_prompt` - System instructions for the new connection.
-   `tools` - Tool specifications for the new connection.
-   `messages` - Conversation history to replay into the new connection.
-   `**restart_kwargs` - Reserved for provider-specific restart options.