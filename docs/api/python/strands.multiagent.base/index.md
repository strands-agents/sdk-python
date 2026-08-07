Multi-Agent Base Class.

Provides minimal foundation for multi-agent patterns (Swarm, Graph).

## Status

```python
class Status(Enum)
```

Defined in: [src/strands/multiagent/base.py:26](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L26)

Execution status for both graphs and nodes.

**Attributes**:

-   `PENDING` - Task has not started execution yet.
-   `EXECUTING` - Task is currently running.
-   `COMPLETED` - Task finished successfully.
-   `FAILED` - Task encountered an error and could not complete.
-   `INTERRUPTED` - Task was interrupted by user.

## NodeResult

```python
@dataclass
class NodeResult()
```

Defined in: [src/strands/multiagent/base.py:45](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L45)

Unified result from node execution - handles both Agent and nested MultiAgentBase results.

#### \_\_str\_\_

```python
def __str__() -> str
```

Defined in: [src/strands/multiagent/base.py:61](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L61)

Return a human-readable string representation of the node result.

Delegates to the inner result’s string representation:

-   AgentResult: Uses AgentResult.**str** (extracts text content)
-   MultiAgentResult: Uses MultiAgentResult.**str** (recursive)
-   Exception: Uses the exception’s string representation

**Returns**:

String representation of the node’s result.

#### get\_agent\_results

```python
def get_agent_results() -> list[AgentResult]
```

Defined in: [src/strands/multiagent/base.py:74](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L74)

Get all AgentResult objects from this node, flattened if nested.

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Defined in: [src/strands/multiagent/base.py:87](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L87)

Convert NodeResult to JSON-serializable dict, ignoring state field.

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "NodeResult"
```

Defined in: [src/strands/multiagent/base.py:108](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L108)

Rehydrate a NodeResult from persisted JSON.

## MultiAgentResult

```python
@dataclass
class MultiAgentResult()
```

Defined in: [src/strands/multiagent/base.py:143](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L143)

Result from multi-agent execution with accumulated metrics.

#### \_\_str\_\_

```python
def __str__() -> str
```

Defined in: [src/strands/multiagent/base.py:154](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L154)

Return a human-readable string representation of the multi-agent result.

Priority order:

1.  Interrupts (if present) → stringified list of interrupt dicts
2.  Node results → each node’s string output, prefixed with node name

**Returns**:

String representation based on the priority order above.

#### from\_dict

```python
@classmethod
def from_dict(cls, data: dict[str, Any]) -> "MultiAgentResult"
```

Defined in: [src/strands/multiagent/base.py:179](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L179)

Rehydrate a MultiAgentResult from persisted JSON.

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Defined in: [src/strands/multiagent/base.py:203](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L203)

Convert MultiAgentResult to JSON-serializable dict.

## MultiAgentBase

```python
class MultiAgentBase(ABC)
```

Defined in: [src/strands/multiagent/base.py:217](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L217)

Base class for multi-agent helpers.

This class integrates with existing Strands Agent instances and provides multi-agent orchestration capabilities.

**Attributes**:

-   `id` - Unique MultiAgent id for session management,etc.

#### \_\_init\_\_

```python
def __init__() -> None
```

Defined in: [src/strands/multiagent/base.py:233](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L233)

Initialize base multi-agent state.

#### invoke\_async

```python
@abstractmethod
async def invoke_async(task: MultiAgentInput,
                       invocation_state: dict[str, Any] | None = None,
                       **kwargs: Any) -> MultiAgentResult
```

Defined in: [src/strands/multiagent/base.py:271](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L271)

Invoke asynchronously.

**Arguments**:

-   `task` - The task to execute
-   `invocation_state` - Additional state/context passed to underlying agents. Defaults to None to avoid mutable default argument issues.
-   `**kwargs` - Additional keyword arguments passed to underlying agents.

#### stream\_async

```python
async def stream_async(task: MultiAgentInput,
                       invocation_state: dict[str, Any] | None = None,
                       **kwargs: Any) -> AsyncIterator[dict[str, Any]]
```

Defined in: [src/strands/multiagent/base.py:284](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L284)

Stream events during multi-agent execution.

Default implementation executes invoke\_async and yields the result as a single event. Subclasses can override this method to provide true streaming capabilities.

**Arguments**:

-   `task` - The task to execute
-   `invocation_state` - Additional state/context passed to underlying agents. Defaults to None to avoid mutable default argument issues.
-   `**kwargs` - Additional keyword arguments passed to underlying agents.

**Yields**:

Dictionary events containing multi-agent execution information including:

-   Multi-agent coordination events (node start/complete, handoffs)
-   Forwarded single-agent events with node context
-   Final result event

#### \_\_call\_\_

```python
def __call__(task: MultiAgentInput,
             invocation_state: dict[str, Any] | None = None,
             **kwargs: Any) -> MultiAgentResult
```

Defined in: [src/strands/multiagent/base.py:309](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L309)

Invoke synchronously.

**Arguments**:

-   `task` - The task to execute
-   `invocation_state` - Additional state/context passed to underlying agents. Defaults to None to avoid mutable default argument issues.
-   `**kwargs` - Additional keyword arguments passed to underlying agents.

#### serialize\_state

```python
def serialize_state() -> dict[str, Any]
```

Defined in: [src/strands/multiagent/base.py:329](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L329)

Return a JSON-serializable snapshot of the orchestrator state.

#### deserialize\_state

```python
def deserialize_state(payload: dict[str, Any]) -> None
```

Defined in: [src/strands/multiagent/base.py:333](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L333)

Restore orchestrator state from a session dict.

#### add\_hook

```python
def add_hook(callback: HookCallback,
             event_type: type | list[type] | None = None,
             *,
             order: float = HookOrder.DEFAULT) -> None
```

Defined in: [src/strands/multiagent/base.py:337](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/base.py#L337)

Register a hook callback with the orchestrator.

Subclasses that support hooks should override this method to register the callback with their hook registry.

**Arguments**:

-   `callback` - The callback function to invoke when events of this type occur.
-   `event_type` - The class type(s) of events this callback should handle. Can be a single type, a list of types, or None to infer from the callback’s first parameter type hint.
-   `order` - Execution priority. Lower values execute first.