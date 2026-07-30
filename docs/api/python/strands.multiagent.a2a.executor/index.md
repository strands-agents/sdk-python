Strands Agent executor for the A2A protocol.

This module provides the StrandsA2AExecutor class, which adapts a Strands Agent to be used as an executor in the A2A protocol. It handles the execution of agent requests and the conversion of Strands Agent streamed responses to A2A events.

The A2A AgentExecutor ensures clients receive responses for synchronous and streamed requests to the A2AServer.

## StrandsA2AExecutor

```python
class StrandsA2AExecutor(AgentExecutor)
```

Defined in: [src/strands/multiagent/a2a/executor.py:109](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/a2a/executor.py#L109)

Executor that adapts a Strands Agent to the A2A protocol.

Handles agent execution in streaming mode and converts Strands Agent responses to A2A protocol events, supporting the full task lifecycle (failed state, cancellation, and interrupt-based input\_required flows).

Conversation state is isolated per A2A `context_id` so callers in different contexts cannot read or influence each other’s history. See `__init__` for the two isolation modes (`agent_factory` and the deprecated single `agent`).

#### \_\_init\_\_

```python
def __init__(agent: SAAgent | None = None,
             *,
             agent_factory: AgentFactory | None = None,
             enable_a2a_compliant_streaming: bool = False,
             max_contexts: int = DEFAULT_MAX_CONTEXTS)
```

Defined in: [src/strands/multiagent/a2a/executor.py:131](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/a2a/executor.py#L131)

Initialize a StrandsA2AExecutor.

Provide exactly one of `agent` or `agent_factory`:

-   `agent_factory` (recommended): a callable `(context_id) -> Agent` invoked once per context to build a dedicated `Agent`. Each context owns an independent agent and runs under its own lock, so different contexts execute concurrently and never share state. The factory is also where per-context concerns such as a `session_manager` are wired.
-   `agent` (deprecated): a single `Agent` reused across contexts. Each context’s conversation state is swapped on/off this instance under a lock, so requests are serialized. A `session_manager` is not supported here, since every context would persist into one interleaved session — use `agent_factory` instead.

**Notes**:

Contexts are keyed on the client-supplied `context_id`, which is not an authentication boundary. A caller that knows another caller’s `context_id` can attach to that conversation. Multi-tenant deployments must enforce authenticated identity at the transport/gateway layer.

At most `max_contexts` contexts are retained; beyond that the least-recently-used is evicted (A2A spec §3.4.1 context cleanup policy) and a later request reusing that `context_id` starts fresh.

**Arguments**:

-   `agent` - A single Strands Agent. Deprecated; prefer `agent_factory`.
-   `agent_factory` - Callable `(context_id) -> Agent` building a fresh agent per context.
-   `enable_a2a_compliant_streaming` - If True, uses A2A-compliant streaming with artifact updates. If False, uses legacy status updates streaming behavior for backwards compatibility. Defaults to False.
-   `max_contexts` - Maximum number of contexts to retain concurrently; the least-recently- used is evicted beyond this. Must be >= 1. Defaults to `DEFAULT_MAX_CONTEXTS`.

**Raises**:

-   `ValueError` - If neither or both of `agent`/`agent_factory` are provided, if `max_contexts` is less than 1, or if a single `agent` has a `session_manager`.

#### execute

```python
async def execute(context: RequestContext, event_queue: EventQueue) -> None
```

Defined in: [src/strands/multiagent/a2a/executor.py:315](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/a2a/executor.py#L315)

Execute a request using the Strands Agent and send the response as A2A events.

This method executes the user’s input using the Strands Agent in streaming mode and converts the agent’s response to A2A events. If the agent raises an exception, the task transitions to the `failed` state. If the agent returns with interrupts, the task transitions to the `input_required` state.

**Arguments**:

-   `context` - The A2A request context, containing the user’s input and task metadata.
-   `event_queue` - The A2A event queue used to send response events back to the client.

**Raises**:

-   `ServerError` - If an unrecoverable error occurs during agent execution setup (e.g., missing input). Agent execution errors are handled gracefully by transitioning the task to the failed state.

#### cancel

```python
async def cancel(context: RequestContext, event_queue: EventQueue) -> None
```

Defined in: [src/strands/multiagent/a2a/executor.py:547](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/multiagent/a2a/executor.py#L547)

Cancel an ongoing execution.

Transitions the task to the canceled state and attempts to stop the agent. The agent’s cancel() method is called to signal cooperative cancellation of in-flight execution.

Note: This transitions the A2A task state. The underlying agent execution may still complete its current model call before stopping.

**Arguments**:

-   `context` - The A2A request context.
-   `event_queue` - The A2A event queue.

**Raises**:

-   `ServerError` - If no current task exists or the task is already in a terminal state.