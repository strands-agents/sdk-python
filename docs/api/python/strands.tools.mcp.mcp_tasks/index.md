Types and configuration for MCP task-augmented tool execution.

This surface is experimental and subject to change. The finalized SEP-2663 models require mcp 2.x (`MCPCreateTaskResult`, `MCPGetTaskResult`, and the other task result types). On the runtime pin `mcp<2.0.0` they cannot round-trip server JSON; the corresponding client methods raise `RuntimeError`.

## TasksConfig

```python
class TasksConfig(TypedDict)
```

Defined in: [src/strands/tools/mcp/mcp\_tasks.py:46](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_tasks.py#L46)

Configuration for MCP task-augmented tool execution.

Experimental: this configuration and the task lifecycle it enables are subject to change as MCP Tasks evolve.

On MCP 2.x, enabling this configuration advertises the SEP-2663 Tasks extension and automatically completes task handles returned by tools. On MCP 1.x, the legacy 2025-11-25 task flow remains supported.

**Attributes**:

-   `poll_timeout` - Overall timeout for task completion. Defaults to 5 minutes.
-   `request_timeout` - Timeout for each task lifecycle request. Defaults to 1 minute.
-   `poll_interval` - Polling delay when the server omits `pollIntervalMs`. Defaults to 1 second.
-   `ttl` - Legacy 2025-11-25 task time-to-live. Defaults to 1 minute.

## MCPTaskError

```python
class MCPTaskError(Result)
```

Defined in: [src/strands/tools/mcp/mcp\_tasks.py:70](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_tasks.py#L70)

JSON-RPC error stored by a failed MCP task.

## MCPTask

```python
class MCPTask(Result)
```

Defined in: [src/strands/tools/mcp/mcp\_tasks.py:78](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_tasks.py#L78)

Operational state shared by every SEP-2663 task result.

#### validate\_timestamp

```python
@field_validator("created_at", "last_updated_at")
@classmethod
def validate_timestamp(cls, value: str) -> str
```

Defined in: [src/strands/tools/mcp/mcp\_tasks.py:93](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_tasks.py#L93)

Validate an ISO 8601 timestamp with a UTC offset.

#### validate\_chronology

```python
@model_validator(mode="after")
def validate_chronology() -> Self
```

Defined in: [src/strands/tools/mcp/mcp\_tasks.py:101](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_tasks.py#L101)

Validate that the task update does not predate its creation.

## MCPCreateTaskResult

```python
class MCPCreateTaskResult(MCPTask)
```

Defined in: [src/strands/tools/mcp/mcp\_tasks.py:110](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_tasks.py#L110)

Task handle returned instead of an immediate tool result.

## MCPGetTaskResult

```python
class MCPGetTaskResult(MCPTask)
```

Defined in: [src/strands/tools/mcp/mcp\_tasks.py:116](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_tasks.py#L116)

Status-specific task state returned by `tasks/get`.

#### validate\_status\_fields

```python
@model_validator(mode="before")
@classmethod
def validate_status_fields(cls, value: Any) -> Any
```

Defined in: [src/strands/tools/mcp/mcp\_tasks.py:126](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_tasks.py#L126)

Reject status-specific fields on every other task status.

#### validate\_status\_payload

```python
@model_validator(mode="after")
def validate_status_payload() -> Self
```

Defined in: [src/strands/tools/mcp/mcp\_tasks.py:143](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_tasks.py#L143)

Validate the payload associated with the task’s current status.

## \_MCPTaskAcknowledgement

```python
class _MCPTaskAcknowledgement(Result)
```

Defined in: [src/strands/tools/mcp/mcp\_tasks.py:155](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_tasks.py#L155)

Validated empty acknowledgement for a task lifecycle operation.

#### validate\_empty\_acknowledgement

```python
@model_validator(mode="before")
@classmethod
def validate_empty_acknowledgement(cls, value: Any) -> Any
```

Defined in: [src/strands/tools/mcp/mcp\_tasks.py:162](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_tasks.py#L162)

Reject task state in an operation’s empty acknowledgement.

## MCPUpdateTaskResult

```python
class MCPUpdateTaskResult(_MCPTaskAcknowledgement)
```

Defined in: [src/strands/tools/mcp/mcp\_tasks.py:181](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_tasks.py#L181)

Acknowledgement returned by `tasks/update`.

## MCPCancelTaskResult

```python
class MCPCancelTaskResult(_MCPTaskAcknowledgement)
```

Defined in: [src/strands/tools/mcp/mcp\_tasks.py:185](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/mcp/mcp_tasks.py#L185)

Acknowledgement returned by `tasks/cancel`.