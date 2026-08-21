Protocol for bidirectional streaming IO channels.

Defines callable protocols for input and output channels that can be used with BidiAgent. This approach provides better typing and flexibility by separating input and output concerns into independent callables.

## BidiInput

```python
@runtime_checkable
class BidiInput(Protocol)
```

Defined in: [src/strands/experimental/bidi/types/io.py:18](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/io.py#L18)

Protocol for bidirectional input callables.

Input callables read data from a source (microphone, camera, websocket, etc.) and return events to be sent to the agent.

#### start

```python
async def start(agent: "BidiAgent") -> None
```

Defined in: [src/strands/experimental/bidi/types/io.py:25](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/io.py#L25)

Start input.

#### stop

```python
async def stop() -> None
```

Defined in: [src/strands/experimental/bidi/types/io.py:29](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/io.py#L29)

Stop input.

#### \_\_call\_\_

```python
def __call__() -> Awaitable[BidiInputEvent]
```

Defined in: [src/strands/experimental/bidi/types/io.py:33](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/io.py#L33)

Read input data from the source.

**Returns**:

Awaitable that resolves to an input event (audio, text, image, etc.)

## BidiOutput

```python
@runtime_checkable
class BidiOutput(Protocol)
```

Defined in: [src/strands/experimental/bidi/types/io.py:43](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/io.py#L43)

Protocol for bidirectional output callables.

Output callables receive events from the agent and handle them appropriately (play audio, display text, send over websocket, etc.).

#### start

```python
async def start(agent: "BidiAgent") -> None
```

Defined in: [src/strands/experimental/bidi/types/io.py:50](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/io.py#L50)

Start output.

#### stop

```python
async def stop() -> None
```

Defined in: [src/strands/experimental/bidi/types/io.py:54](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/io.py#L54)

Stop output.

#### \_\_call\_\_

```python
def __call__(event: BidiOutputEvent) -> Awaitable[None]
```

Defined in: [src/strands/experimental/bidi/types/io.py:58](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/types/io.py#L58)

Process output events from the agent.

**Arguments**:

-   `event` - Output event from the agent (audio, text, tool calls, etc.)