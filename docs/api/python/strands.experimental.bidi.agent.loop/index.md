Agent loop.

The agent loop handles the events received from the model and executes tools when given a tool use request.

## \_BidiAgentLoop

```python
class _BidiAgentLoop()
```

Defined in: [src/strands/experimental/bidi/agent/loop.py:50](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/loop.py#L50)

Agent loop.

**Attributes**:

-   `_agent` - BidiAgent instance to loop.
-   `_started` - Flag if agent loop has started.
-   `_task_pool` - Track active async tasks created in loop.
-   `_event_queue` - Queue model and tool call events for receiver.
-   `_invocation_state` - Optional context to pass to tools during execution. This allows passing custom data (user\_id, session\_id, database connections, etc.) that tools can access via their invocation\_state parameter.
-   `_send_gate` - Gate the sending of events to the model. Blocks when agent is resetting the model connection after timeout.

#### \_\_init\_\_

```python
def __init__(agent: "BidiAgent") -> None
```

Defined in: [src/strands/experimental/bidi/agent/loop.py:65](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/loop.py#L65)

Initialize members of the agent loop.

Note, before receiving events from the loop, the user must call `start`.

**Arguments**:

-   `agent` - Bidirectional agent to loop over.

#### start

```python
async def start(invocation_state: dict[str, Any] | None = None) -> None
```

Defined in: [src/strands/experimental/bidi/agent/loop.py:88](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/loop.py#L88)

Start the agent loop.

The agent model is started as part of this call.

**Arguments**:

-   `invocation_state` - Optional context to pass to tools during execution. This allows passing custom data (user\_id, session\_id, database connections, etc.) that tools can access via their invocation\_state parameter.

**Raises**:

-   `RuntimeError` - If loop already started.

#### stop

```python
async def stop() -> None
```

Defined in: [src/strands/experimental/bidi/agent/loop.py:146](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/loop.py#L146)

Stop the agent loop.

#### send

```python
async def send(event: BidiInputEvent | ToolResultEvent) -> None
```

Defined in: [src/strands/experimental/bidi/agent/loop.py:176](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/loop.py#L176)

Send model event.

Additionally, add text input to messages array.

**Arguments**:

-   `event` - User input event or tool result.

**Raises**:

-   `RuntimeError` - If start has not been called.

#### receive

```python
async def receive() -> AsyncGenerator[BidiOutputEvent, None]
```

Defined in: [src/strands/experimental/bidi/agent/loop.py:200](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/agent/loop.py#L200)

Receive model and tool call events.

**Returns**:

Model and tool call events.

**Raises**:

-   `RuntimeError` - If start has not been called.