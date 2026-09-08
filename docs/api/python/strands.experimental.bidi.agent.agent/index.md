Bidirectional Agent for real-time streaming conversations.

Provides real-time audio and text interaction through persistent streaming connections. Unlike traditional request-response patterns, this agent maintains long-running conversations where users can interrupt, provide additional input, and receive continuous responses including audio output.

Key capabilities:

-   Persistent conversation connections with concurrent processing
-   Real-time audio input/output streaming
-   Automatic interruption detection and tool execution
-   Event-driven communication with model providers

## BidiAgent

```python
class BidiAgent(LocalAgent)
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:60](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L60)

Agent for bidirectional streaming conversations.

Enables real-time audio and text interaction with AI models through persistent connections. Supports concurrent tool execution and interruption handling.

#### \_\_init\_\_

```python
def __init__(model: BidiModel | str | None = None,
             tools: list[str | AgentTool | ToolProvider] | None = None,
             system_prompt: str | list[SystemContentBlock] | None = None,
             messages: Messages | None = None,
             record_direct_tool_call: bool = True,
             load_tools_from_directory: bool = False,
             agent_id: str | None = None,
             name: str | None = None,
             description: str | None = None,
             hooks: list[HookProvider] | None = None,
             state: AgentState | dict | None = None,
             session_manager: "SessionManager | None" = None,
             tool_executor: ToolExecutor | None = None,
             **kwargs: Any)
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:69](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L69)

Initialize bidirectional agent.

**Arguments**:

-   `model` - BidiModel instance, string model\_id, or None for default detection.
-   `tools` - Optional list of tools with flexible format support.
-   `system_prompt` - System prompt for conversations as a string or structured content blocks. Structured blocks are retained, while their text is passed to Bidi models as a string.
-   `messages` - Optional conversation history to initialize with.
-   `record_direct_tool_call` - Whether to record direct tool calls in message history.
-   `load_tools_from_directory` - Whether to load and automatically reload tools in the `./tools/` directory.
-   `agent_id` - Optional ID for the agent, useful for connection management and multi-agent scenarios.
-   `name` - Name of the Agent.
-   `description` - Description of what the Agent does.
-   `hooks` - Optional list of hook providers to register for lifecycle events.
-   `state` - Stateful information for the agent. Can be either an AgentState object, or a json serializable dict.
-   `session_manager` - Manager for handling agent sessions including conversation history and state. If provided, enables session-based persistence and state management.
-   `tool_executor` - Definition of tool execution strategy (e.g., sequential, concurrent, etc.).
-   `**kwargs` - Additional configuration for future extensibility.

**Raises**:

-   `ValueError` - If model configuration is invalid or state is invalid type.
-   `TypeError` - If model type is unsupported.

#### tool

```python
@property
def tool() -> _ToolCaller
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:197](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L197)

Call tool as a function.

**Returns**:

ToolCaller for method-style tool execution.

**Example**:

```plaintext
agent = BidiAgent(model=model, tools=[calculator])
agent.tool.calculator(expression="2+2")
```

#### tool\_names

```python
@property
def tool_names() -> list[str]
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:212](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L212)

Get a list of all registered tool names.

**Returns**:

Names of all tools available to this agent.

#### system\_prompt

```python
@property
def system_prompt() -> str | None
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:222](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L222)

Get the system prompt as a string.

#### system\_prompt

```python
@system_prompt.setter
def system_prompt(value: str | list[SystemContentBlock] | None) -> None
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:227](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L227)

Set the system prompt and retain its structured content representation.

#### system\_prompt\_content

```python
@property
def system_prompt_content() -> list[SystemContentBlock] | None
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:232](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L232)

Get the system prompt as structured content blocks.

#### session\_id

```python
@property
def session_id() -> str
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:237](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L237)

Get the conversation session identifier.

#### add\_hook

```python
def add_hook(callback: HookCallback[TEvent],
             event_type: type[TEvent] | list[type[TEvent]] | None = None,
             *,
             order: float = HookOrder.DEFAULT) -> None
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:241](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L241)

Register a callback function for a specific event type.

This method supports multiple call patterns:

1.  `add_hook(callback)` - Event type inferred from callback’s type hint
2.  `add_hook(callback, event_type)` - Event type specified explicitly
3.  `add_hook(callback, [TypeA, TypeB])` - Register for multiple event types

When the callback’s type hint is a union type (`A | B` or `Union[A, B]`), the callback is automatically registered for each event type in the union.

Callbacks can be either synchronous or asynchronous functions.

**Arguments**:

-   `callback` - The callback function to invoke when events of this type occur.
-   `event_type` - The class type(s) of events this callback should handle. Can be a single type, a list of types, or None to infer from the callback’s first parameter type hint. If a list is provided, the callback is registered for each type in the list.
-   `order` - Execution priority. Lower values execute first. Use a HookOrder constant such as SDK\_FIRST (-100), DEFAULT (0), MODEL\_ROUTING (50), or SDK\_LAST (100).

**Raises**:

-   `ValueError` - If event\_type is not provided and cannot be inferred from the callback’s type hints, or if the event\_type list is empty.

#### start

```python
async def start(invocation_state: dict[str, Any] | None = None) -> None
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:276](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L276)

Start a persistent bidirectional conversation connection.

Initializes the streaming connection and starts background tasks for processing model events, tool execution, and connection management.

**Arguments**:

-   `invocation_state` - Optional context to pass to tools during execution. This allows passing custom data (user\_id, session\_id, database connections, etc.) that tools can access via their invocation\_state parameter.

**Raises**:

RuntimeError: If agent already started.

**Example**:

```python
await agent.start(invocation_state=\{
    "user_id": "user_123",
    "session_id": "session_456",
    "database": db_connection,
})
```

#### send

```python
async def send(input_data: BidiAgentInput | dict[str, Any]) -> None
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:307](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L307)

Send input to the model (text, audio, image, or event dict).

Unified method for sending text, audio, and image input to the model during an active conversation session. Accepts TypedEvent instances or plain dicts (e.g., from WebSocket clients) which are automatically reconstructed.

**Arguments**:

-   `input_data` - Can be:
    
    -   str: Text message from user
    -   BidiInputEvent: TypedEvent
    -   dict: Event dictionary (will be reconstructed to TypedEvent)

**Raises**:

-   `RuntimeError` - If start has not been called.
-   `ValueError` - If invalid input type.

**Example**:

await agent.send(“Hello”) await agent.send(BidiAudioInputEvent(audio=“base64…”, format=“pcm”, …)) await agent.send({“type”: “bidirectional\_text\_input”, “text”: “Hello”, “role”: “user”})

#### receive

```python
async def receive() -> AsyncGenerator[BidiOutputEvent, None]
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:358](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L358)

Receive events from the model including audio, text, and tool calls.

**Yields**:

Model output events processed by background tasks including audio output, text responses, tool calls, and connection updates.

**Raises**:

-   `RuntimeError` - If start has not been called.

#### stop

```python
async def stop() -> None
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:374](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L374)

End the conversation connection and cleanup all resources.

Terminates the streaming connection, cancels background tasks, and closes the connection to the model provider.

#### \_\_aenter\_\_

```python
async def __aenter__(
        invocation_state: dict[str, Any] | None = None) -> "BidiAgent"
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:383](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L383)

Async context manager entry point.

Automatically starts the bidirectional connection when entering the context.

**Arguments**:

-   `invocation_state` - Optional context to pass to tools during execution. This allows passing custom data (user\_id, session\_id, database connections, etc.) that tools can access via their invocation\_state parameter.

**Returns**:

Self for use in the context.

#### \_\_aexit\_\_

```python
async def __aexit__(*_: Any) -> None
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:400](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L400)

Async context manager exit point.

Automatically ends the connection and cleans up resources including when exiting the context, regardless of whether an exception occurred.

#### run

```python
async def run(inputs: list[BidiInput],
              outputs: list[BidiOutput],
              invocation_state: dict[str, Any] | None = None) -> None
```

Defined in: [src/strands/experimental/bidi/agent/agent.py:409](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/agent.py#L409)

Run the agent using provided IO channels for bidirectional communication.

**Arguments**:

-   `inputs` - Input callables to read data from a source
-   `outputs` - Output callables to receive events from the agent
-   `invocation_state` - Optional context to pass to tools during execution. This allows passing custom data (user\_id, session\_id, database connections, etc.) that tools can access via their invocation\_state parameter.

**Example**:

```python
# Using model defaults:
model = BedrockNovaSonicModel()
audio_io = BidiAudioIO()
text_io = BidiTextIO()
agent = BidiAgent(model=model, tools=[calculator])
await agent.run(
    inputs=[audio_io.input()],
    outputs=[audio_io.output(), text_io.output()],
    invocation_state=\{"user_id": "user_123"}
)

# Using custom audio config:
model = BedrockNovaSonicModel(
    audio=\{
        "input_rate": 48000,
        "output_rate": 24000,
    }
)
audio_io = BidiAudioIO()
agent = BidiAgent(model=model, tools=[calculator])
await agent.run(
    inputs=[audio_io.input()],
    outputs=[audio_io.output()],
)
```