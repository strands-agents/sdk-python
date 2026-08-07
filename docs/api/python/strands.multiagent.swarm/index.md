Swarm Multi-Agent Pattern Implementation.

This module provides a collaborative agent orchestration system where agents work together as a team to solve complex tasks, with shared context and autonomous coordination.

Key Features:

-   Self-organizing agent teams with shared working memory
-   Tool-based coordination
-   Autonomous agent collaboration without central control
-   Dynamic task distribution based on agent capabilities
-   Collective intelligence through shared context
-   Human input via user interrupts raised in BeforeNodeCallEvent hooks and agent nodes

## SwarmNode

```python
@dataclass
class SwarmNode()
```

Defined in: [src/strands/multiagent/swarm.py:67](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L67)

Represents a node (e.g. Agent) in the swarm.

#### \_\_post\_init\_\_

```python
def __post_init__() -> None
```

Defined in: [src/strands/multiagent/swarm.py:77](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L77)

Capture initial executor state after initialization.

#### \_\_hash\_\_

```python
def __hash__() -> int
```

Defined in: [src/strands/multiagent/swarm.py:84](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L84)

Return hash for SwarmNode based on node\_id.

#### \_\_eq\_\_

```python
def __eq__(other: Any) -> bool
```

Defined in: [src/strands/multiagent/swarm.py:88](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L88)

Return equality for SwarmNode based on node\_id.

#### \_\_str\_\_

```python
def __str__() -> str
```

Defined in: [src/strands/multiagent/swarm.py:94](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L94)

Return string representation of SwarmNode.

#### \_\_repr\_\_

```python
def __repr__() -> str
```

Defined in: [src/strands/multiagent/swarm.py:98](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L98)

Return detailed representation of SwarmNode.

#### reset\_executor\_state

```python
def reset_executor_state() -> None
```

Defined in: [src/strands/multiagent/swarm.py:102](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L102)

Reset SwarmNode executor state to initial state when swarm was created.

If Swarm is resuming from an interrupt, we reset the executor state from the interrupt context.

## SharedContext

```python
@dataclass
class SharedContext()
```

Defined in: [src/strands/multiagent/swarm.py:121](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L121)

Shared context between swarm nodes.

#### add\_context

```python
def add_context(node: SwarmNode, key: str, value: Any) -> None
```

Defined in: [src/strands/multiagent/swarm.py:126](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L126)

Add context.

## SwarmState

```python
@dataclass
class SwarmState()
```

Defined in: [src/strands/multiagent/swarm.py:170](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L170)

Current state of swarm execution.

#### current\_node

The agent currently executing

#### task

The original task from the user that is being executed

#### completion\_status

Current swarm execution status

#### shared\_context

Context shared between agents

#### node\_history

Complete history of agents that have executed

#### start\_time

When swarm execution began

#### results

Results from each agent execution

#### execution\_time

Total execution time in milliseconds

#### handoff\_node

The agent to execute next

#### handoff\_message

Message passed during agent handoff

#### should\_continue

```python
def should_continue(
        *, max_handoffs: int, max_iterations: int, execution_timeout: float,
        repetitive_handoff_detection_window: int,
        repetitive_handoff_min_unique_agents: int) -> tuple[bool, str]
```

Defined in: [src/strands/multiagent/swarm.py:188](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L188)

Check if the swarm should continue.

Returns: (should\_continue, reason)

## SwarmResult

```python
@dataclass
class SwarmResult(MultiAgentResult)
```

Defined in: [src/strands/multiagent/swarm.py:231](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L231)

Result from swarm execution - extends MultiAgentResult with swarm-specific details.

## Swarm

```python
class Swarm(MultiAgentBase)
```

Defined in: [src/strands/multiagent/swarm.py:258](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L258)

Self-organizing collaborative agent teams with shared working memory.

#### \_\_init\_\_

```python
def __init__(nodes: list[Agent],
             *,
             entry_point: Agent | None = None,
             max_handoffs: int = 20,
             max_iterations: int = 20,
             execution_timeout: float = 900.0,
             node_timeout: float = 300.0,
             repetitive_handoff_detection_window: int = 0,
             repetitive_handoff_min_unique_agents: int = 0,
             session_manager: SessionManager | None = None,
             hooks: list[HookProvider] | None = None,
             id: str = _DEFAULT_SWARM_ID,
             trace_attributes: Mapping[str, AttributeValue] | None = None,
             plugins: list[MultiAgentPlugin] | None = None) -> None
```

Defined in: [src/strands/multiagent/swarm.py:261](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L261)

Initialize Swarm with agents and configuration.

**Arguments**:

-   `id` - Unique swarm id (default: “default\_swarm”)
-   `nodes` - List of nodes (e.g. Agent) to include in the swarm
-   `entry_point` - Agent to start with. If None, uses the first agent (default: None)
-   `max_handoffs` - Maximum handoffs to agents and users (default: 20)
-   `max_iterations` - Maximum node executions within the swarm (default: 20)
-   `execution_timeout` - Total execution timeout in seconds (default: 900.0)
-   `node_timeout` - Individual node timeout in seconds (default: 300.0)
-   `repetitive_handoff_detection_window` - Number of recent nodes to check for repetitive handoffs Disabled by default (default: 0)
-   `repetitive_handoff_min_unique_agents` - Minimum unique agents required in recent sequence Disabled by default (default: 0)
-   `session_manager` - Session manager for persisting graph state and execution history (default: None)
-   `hooks` - List of hook providers for monitoring and extending graph execution behavior (default: None)
-   `trace_attributes` - Custom trace attributes to apply to the agent’s trace span (default: None)
-   `plugins` - List of multi-agent plugins for extending swarm behavior (default: None)

#### add\_hook

```python
def add_hook(callback: HookCallback,
             event_type: type | list[type] | None = None,
             *,
             order: float = HookOrder.DEFAULT) -> None
```

Defined in: [src/strands/multiagent/swarm.py:341](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L341)

Register a hook callback with the swarm.

**Arguments**:

-   `callback` - The callback function to invoke when events of this type occur.
-   `event_type` - The class type(s) of events this callback should handle. Can be a single type, a list of types, or None to infer from the callback’s first parameter type hint.
-   `order` - Execution priority. Lower values execute first.

#### \_\_call\_\_

```python
def __call__(task: MultiAgentInput,
             invocation_state: dict[str, Any] | None = None,
             **kwargs: Any) -> SwarmResult
```

Defined in: [src/strands/multiagent/swarm.py:355](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L355)

Invoke the swarm synchronously.

**Arguments**:

-   `task` - The task to execute
-   `invocation_state` - Additional state/context passed to underlying agents. Defaults to None to avoid mutable default argument issues.
-   `**kwargs` - Keyword arguments allowing backward compatible future changes.

#### invoke\_async

```python
async def invoke_async(task: MultiAgentInput,
                       invocation_state: dict[str, Any] | None = None,
                       **kwargs: Any) -> SwarmResult
```

Defined in: [src/strands/multiagent/swarm.py:370](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L370)

Invoke the swarm asynchronously.

This method uses stream\_async internally and consumes all events until completion, following the same pattern as the Agent class.

**Arguments**:

-   `task` - The task to execute
-   `invocation_state` - Additional state/context passed to underlying agents. Defaults to None to avoid mutable default argument issues.
-   `**kwargs` - Keyword arguments allowing backward compatible future changes.

#### stream\_async

```python
async def stream_async(task: MultiAgentInput,
                       invocation_state: dict[str, Any] | None = None,
                       **kwargs: Any) -> AsyncIterator[dict[str, Any]]
```

Defined in: [src/strands/multiagent/swarm.py:394](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L394)

Stream events during swarm execution.

**Arguments**:

-   `task` - The task to execute
-   `invocation_state` - Additional state/context passed to underlying agents. Defaults to None to avoid mutable default argument issues.
-   `**kwargs` - Keyword arguments allowing backward compatible future changes.

**Yields**:

Dictionary events during swarm execution, such as:

-   multi\_agent\_node\_start: When a node begins execution
-   multi\_agent\_node\_stream: Forwarded agent events with node context
-   multi\_agent\_handoff: When control is handed off between agents
-   multi\_agent\_node\_stop: When a node stops execution
-   result: Final swarm result

#### serialize\_state

```python
def serialize_state() -> dict[str, Any]
```

Defined in: [src/strands/multiagent/swarm.py:1043](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L1043)

Serialize the current swarm state to a dictionary.

#### deserialize\_state

```python
def deserialize_state(payload: dict[str, Any]) -> None
```

Defined in: [src/strands/multiagent/swarm.py:1095](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/swarm.py#L1095)

Restore swarm state from a session dict and prepare for execution.

This method handles two scenarios:

1.  If the swarm has no resume frontier (next\_nodes\_to\_execute is empty — e.g. completed, or failed at a dead end), resets all nodes and swarm state to allow re-execution from the beginning.
2.  Otherwise, restores the persisted state and prepares to resume execution from the next node.

**Notes**:

Resume is at-least-once: a crash between a node finishing and the next checkpoint replays that node on restore (e.g. a terminal node whose completion checkpoint never landed), and a resume frontier naming a node this swarm no longer defines re-runs the task from the beginning. Swarm nodes should therefore be idempotent.

**Arguments**:

-   `payload` - Dictionary containing persisted state data including status, node history, results, and next nodes to execute.