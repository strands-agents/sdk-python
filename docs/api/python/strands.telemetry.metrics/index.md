Utilities for collecting and reporting performance metrics in the SDK.

## Trace

```python
class Trace()
```

Defined in: [src/strands/telemetry/metrics.py:22](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L22)

A trace representing a single operation or step in the execution flow.

#### \_\_init\_\_

```python
def __init__(name: str,
             parent_id: str | None = None,
             start_time: float | None = None,
             raw_name: str | None = None,
             metadata: dict[str, Any] | None = None,
             message: Message | None = None) -> None
```

Defined in: [src/strands/telemetry/metrics.py:25](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L25)

Initialize a new trace.

**Arguments**:

-   `name` - Human-readable name of the operation being traced.
-   `parent_id` - ID of the parent trace, if this is a child operation.
-   `start_time` - Timestamp when the trace started. If not provided, the current time will be used.
-   `raw_name` - System level name.
-   `metadata` - Additional contextual information about the trace.
-   `message` - Message associated with the trace.

#### end

```python
def end(end_time: float | None = None) -> None
```

Defined in: [src/strands/telemetry/metrics.py:55](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L55)

Mark the trace as complete with the given or current timestamp.

**Arguments**:

-   `end_time` - Timestamp to use as the end time. If not provided, the current time will be used.

#### add\_child

```python
def add_child(child: "Trace") -> None
```

Defined in: [src/strands/telemetry/metrics.py:64](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L64)

Add a child trace to this trace.

**Arguments**:

-   `child` - The child trace to add.

#### duration

```python
def duration() -> float | None
```

Defined in: [src/strands/telemetry/metrics.py:72](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L72)

Calculate the duration of this trace.

**Returns**:

The duration in seconds, or None if the trace hasn’t ended yet.

#### add\_message

```python
def add_message(message: Message) -> None
```

Defined in: [src/strands/telemetry/metrics.py:80](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L80)

Add a message to the trace.

**Arguments**:

-   `message` - The message to add.

#### to\_dict

```python
def to_dict() -> dict[str, Any]
```

Defined in: [src/strands/telemetry/metrics.py:88](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L88)

Convert the trace to a dictionary representation.

**Returns**:

A dictionary containing all trace information, suitable for serialization.

## ToolMetrics

```python
@dataclass
class ToolMetrics()
```

Defined in: [src/strands/telemetry/metrics.py:109](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L109)

Metrics for a specific tool’s usage.

**Attributes**:

-   `tool` - The tool being tracked.
-   `call_count` - Number of times the tool has been called.
-   `success_count` - Number of successful tool calls.
-   `error_count` - Number of failed tool calls.
-   `total_time` - Total execution time across all calls in seconds.

#### add\_call

```python
def add_call(tool: ToolUse,
             duration: float,
             success: bool,
             metrics_client: "MetricsClient",
             attributes: dict[str, Any] | None = None) -> None
```

Defined in: [src/strands/telemetry/metrics.py:126](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L126)

Record a new tool call with its outcome.

**Arguments**:

-   `tool` - The tool that was called.
-   `duration` - How long the call took in seconds.
-   `success` - Whether the call was successful.
-   `metrics_client` - The metrics client for recording the metrics.
-   `attributes` - attributes of the metrics.

## EventLoopCycleMetric

```python
@dataclass
class EventLoopCycleMetric()
```

Defined in: [src/strands/telemetry/metrics.py:157](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L157)

Aggregated metrics for a single event loop cycle.

**Attributes**:

-   `event_loop_cycle_id` - Current eventLoop cycle id.
-   `usage` - Total token usage for the entire cycle (succeeded model invocation, excluding tool invocations).

## AgentInvocation

```python
@dataclass
class AgentInvocation()
```

Defined in: [src/strands/telemetry/metrics.py:170](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L170)

Metrics for a single agent invocation.

AgentInvocation contains all the event loop cycles and accumulated token usage for that invocation.

**Attributes**:

-   `cycles` - List of event loop cycles that occurred during this invocation.
-   `usage` - Accumulated token usage for this invocation across all cycles.

## EventLoopMetrics

```python
@dataclass
class EventLoopMetrics()
```

Defined in: [src/strands/telemetry/metrics.py:207](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L207)

Aggregated metrics for an event loop’s execution.

**Attributes**:

-   `cycle_count` - Number of event loop cycles executed.
-   `tool_metrics` - Metrics for each tool used, keyed by tool name.
-   `cycle_durations` - List of durations for each cycle in seconds.
-   `agent_invocations` - Agent invocation metrics containing cycles and usage data.
-   `traces` - List of execution traces.
-   `accumulated_usage` - Accumulated token usage across all model invocations (across all requests).
-   `accumulated_metrics` - Accumulated performance metrics across all model invocations.

#### latest\_context\_size

```python
@property
def latest_context_size() -> int | None
```

Defined in: [src/strands/telemetry/metrics.py:229](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L229)

Most recent context size from the last LLM call.

This represents the current context size as reported by the model.

**Returns**:

The total prompt the model processed on the most recent cycle, including cached tokens, or None if no data is available.

#### projected\_context\_size

```python
@property
def projected_context_size() -> int | None
```

Defined in: [src/strands/telemetry/metrics.py:245](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L245)

Projected context size for the next model call.

Computed from the most recent cycle’s usage as the total prompt the model processed (including cached tokens) plus the generated output that is now part of the conversation, approximating the input token count for the next model call.

**Returns**:

The projected token count, or None if no data is available.

#### latest\_agent\_invocation

```python
@property
def latest_agent_invocation() -> AgentInvocation | None
```

Defined in: [src/strands/telemetry/metrics.py:269](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L269)

Get the most recent agent invocation.

**Returns**:

The most recent AgentInvocation, or None if no invocations exist.

#### start\_cycle

```python
def start_cycle(attributes: dict[str, Any]) -> tuple[float, Trace]
```

Defined in: [src/strands/telemetry/metrics.py:277](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L277)

Start a new event loop cycle and create a trace for it.

**Arguments**:

-   `attributes` - attributes of the metrics, including event\_loop\_cycle\_id.

**Returns**:

A tuple containing the start time and the cycle trace object.

#### end\_cycle

```python
def end_cycle(start_time: float,
              cycle_trace: Trace,
              attributes: dict[str, Any] | None = None) -> None
```

Defined in: [src/strands/telemetry/metrics.py:305](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L305)

End the current event loop cycle and record its duration.

**Arguments**:

-   `start_time` - The timestamp when the cycle started.
-   `cycle_trace` - The trace object for this cycle.
-   `attributes` - attributes of the metrics.

#### add\_tool\_usage

```python
def add_tool_usage(tool: ToolUse,
                   duration: float,
                   tool_trace: Trace,
                   success: bool,
                   message: Message | None = None) -> None
```

Defined in: [src/strands/telemetry/metrics.py:320](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L320)

Record metrics for a tool invocation.

**Arguments**:

-   `tool` - The tool that was used.
-   `duration` - How long the tool call took in seconds.
-   `tool_trace` - The trace object for this tool call.
-   `success` - Whether the tool call was successful.
-   `message` - The message associated with the tool call, if any. Pass `None` when the call ended without producing a tool result (e.g. on interrupt).

#### update\_usage

```python
def update_usage(usage: Usage) -> None
```

Defined in: [src/strands/telemetry/metrics.py:380](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L380)

Update the accumulated token usage with new usage data.

**Arguments**:

-   `usage` - The usage data to add to the accumulated totals.

#### reset\_usage\_metrics

```python
def reset_usage_metrics() -> None
```

Defined in: [src/strands/telemetry/metrics.py:403](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L403)

Start a new agent invocation by creating a new AgentInvocation.

This should be called at the start of a new request to begin tracking a new agent invocation with fresh usage and cycle data.

#### update\_metrics

```python
def update_metrics(metrics: Metrics) -> None
```

Defined in: [src/strands/telemetry/metrics.py:411](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L411)

Update the accumulated performance metrics with new metrics data.

**Arguments**:

-   `metrics` - The metrics data to add to the accumulated totals.

#### get\_summary

```python
def get_summary() -> dict[str, Any]
```

Defined in: [src/strands/telemetry/metrics.py:422](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L422)

Generate a comprehensive summary of all collected metrics.

**Returns**:

A dictionary containing summarized metrics data. This includes cycle statistics, tool usage, traces, and accumulated usage information.

#### metrics\_to\_string

```python
def metrics_to_string(event_loop_metrics: EventLoopMetrics,
                      allowed_names: set[str] | None = None) -> str
```

Defined in: [src/strands/telemetry/metrics.py:561](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L561)

Convert event loop metrics to a human-readable string representation.

**Arguments**:

-   `event_loop_metrics` - The metrics to format.
-   `allowed_names` - Set of names that are allowed to be displayed unmodified.

**Returns**:

A formatted string representation of the metrics.

## MetricsClient

```python
class MetricsClient()
```

Defined in: [src/strands/telemetry/metrics.py:574](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L574)

Singleton client for managing OpenTelemetry metrics instruments.

The actual metrics export destination (console, OTLP endpoint, etc.) is configured through OpenTelemetry SDK configuration by users, not by this client.

This class uses a thread-safe double-checked locking pattern to ensure safe concurrent initialization across multiple threads.

#### \_\_new\_\_

```python
def __new__(cls) -> "MetricsClient"
```

Defined in: [src/strands/telemetry/metrics.py:602](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L602)

Create or return the singleton instance of MetricsClient.

Uses double-checked locking to ensure thread safety without acquiring the lock on every access after initialization.

**Returns**:

The single MetricsClient instance.

#### \_\_init\_\_

```python
def __init__() -> None
```

Defined in: [src/strands/telemetry/metrics.py:617](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L617)

Initialize the MetricsClient.

This method only runs once due to the singleton pattern. Sets up the OpenTelemetry meter and creates metric instruments. Uses a lock to prevent concurrent initialization races.

#### create\_instruments

```python
def create_instruments() -> None
```

Defined in: [src/strands/telemetry/metrics.py:637](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/telemetry/metrics.py#L637)

Create and initialize all OpenTelemetry metric instruments.