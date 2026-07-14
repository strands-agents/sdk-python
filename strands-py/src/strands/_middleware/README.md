# Python Middleware

This implementation follows the behavioral spec defined in `strands-ts/src/middleware/README.md` with the following intentional divergences:

## Scope

`InvokeModelStage` and `ExecuteToolStage` are implemented. `AgentStreamStage` will be added as needed.

## Result encoding

TypeScript uses async generator `return` values propagated via `yield*`. Python async generators cannot `return` values.

Instead, the **last yielded event IS the result**. This matches the existing Python SDK convention where `ModelStopReason` is the last event from `stream_messages()`, `ToolResultEvent` is the last from tool execution, etc. The middleware chain is transparent — events (including the result event) flow through naturally. There is no separate sentinel type.

Pass-through is:
```python
async def passthrough(context, next_fn):
    async for event in next_fn(context):
        yield event
```

Short-circuit yields the result event directly:
```python
async def cached(context, next_fn):
    yield ModelStopReason(stop_reason="end_turn", message=cached_msg, usage=usage, metrics=metrics)
```

Output phase handlers take and return a `MiddlewareResult` wrapping the result event.
The registry wraps the result event before calling the handler and unwraps the returned
wrapper back into the stream, so Wrap handlers and the event-loop integration still see a
plain result event. Use `result.replace(value=...)` to produce the modified wrapper:
```python
def output_handler(result):  # result: MiddlewareResult
    stop_reason, message, usage, metrics = result.value["stop"]
    return result.replace(
        value=ModelStopReason(stop_reason="custom", message=message, usage=usage, metrics=metrics),
    )
```

Only the **Output** phase uses the wrapper. Wrap and Input handlers deal in raw
events/contexts.

The wrapper currently holds only `value`. Input already has a wrapper (the context
dataclass), so `MiddlewareResult` gives Output the same extensibility surface for future
metadata. Since Python async generators cannot return values, Wrap-phase metadata would
be yielded as events into the stream rather than attached to a return value. See the TS
spec ("Metadata transport") for rationale.

If we later want per-stage typed results (e.g., `InvokeModelResult` with named fields
instead of an opaque `.value`), those can derive from `MiddlewareResult`. Existing Output
handlers that accept `MiddlewareResult` continue to work; new handlers can narrow to the
subclass for typed access. This is a two-way door — no migration required.

## Per-stage result types

Each stage's result is the last event its chain yields. TypeScript wraps these in named
result objects (`InvokeModelResult`, `ExecuteToolResult`); Python uses the underlying event
directly, so there is no equivalent wrapper class:

- `InvokeModelStage` → `ModelStopReason` (the last event from `stream_messages()`).
- `ExecuteToolStage` → `ToolResultEvent` (the last event from tool execution). It already
  carries both `tool_result` and `exception`, so a separate `ExecuteToolResult` is redundant.

Short-circuiting a tool call yields a `ToolResultEvent` directly:
```python
async def cached(context, next_fn):
    yield ToolResultEvent({"toolUseId": context.tool_use["toolUseId"], "status": "success", ...})
```

## Middleware-initiated interrupts (ExecuteToolStage)

`ExecuteToolContext.interrupt(name, reason=..., response=...)` lets tool middleware gate
execution behind a human-in-the-loop approval, mirroring the TS `MiddlewareInterruptible`
contract. It returns a `MiddlewareInterruptResult` (a wrapper around `response`, kept for
forward-compatibility with TS) on resume, and raises `InterruptException` on first call.

`interrupt()` is **read-only** with respect to interrupt state — it inspects prior responses
but never registers the interrupt itself. The tool executor's `InterruptException` handler is
the single registration site. This matches TS, where middleware interrupts deliberately never
write to interrupt state (unlike hook/tool interrupts, which self-register).

A halted (or partially executed) tool call has no result, so interrupts must not be treated as
the stage result:

- **Middleware-initiated** (`context.interrupt()`) raises `InterruptException`, which unwinds
  the chain past the Output adapter; `ToolExecutor._stream` catches it and registers the
  interrupt.
- **Tool-originated** (a `ToolInterruptEvent` from `tool.stream()`, including sub-agent
  interrupts via `_AgentAsTool`) flows through the chain as a normal event. The Output adapter
  skips any event matching the `InterruptControlEvent` protocol (a truthy `is_interrupt`) when
  picking the positional result, so it is never mistaken for the result; `_stream` registers
  its interrupts and short-circuits. The protocol keeps the stage-agnostic registry from
  importing tool-specific event types.

Either way `_stream` surfaces a single `ToolInterruptEvent` to the event loop. Only
`ExecuteToolStage` supports interrupts — `InvokeModelStage` does not, matching TS (only
`ExecuteToolContext` and `AgentStreamContext` are `MiddlewareInterruptible`).

Interrupt IDs are `v1:middleware_execute_tool:<toolUseId>:<uuid5(name)>` — deterministic across
resumes so a resumed response resolves the same interrupt. This follows Python's `v1:`
interrupt-id scheme (`v1:tool_call:...`, `v1:before_tool_call:...`) and its convention of
hashing the name with `uuid5`. (TS uses a different, unversioned literal — id *strings* are
opaque per-SDK handles and are not compared across SDKs, so only the within-SDK scheme matters.)

### No interrupt `source`

The TS spec tags middleware interrupts with `source='middleware'` (distinguishing them from
`hook`/`tool` interrupts). Python's `Interrupt` type has no `source` field at all — not for
hooks, tools, or middleware — so there is nothing for the middleware path to set. This is a
pre-existing, SDK-wide gap in the Python interrupt system rather than a middleware-specific
choice; adding it means changing the core `Interrupt` type and every hook/tool call site, which
is out of scope here. Consumers currently disambiguate by the interrupt id prefix
(`v1:middleware_execute_tool:...`) instead.

## No removal / cleanup

Once registered, middleware cannot be removed. This matches the Python hook system which also does not support removal.

## Private module

The `_middleware/` package is not part of the public API. Internal consumers access it via `agent._middleware_registry.add_middleware(...)`.

## System prompt as a union type

`InvokeModelContext.system_prompt` is `str | list[SystemContentBlock] | None` (a single union field). The terminal decomposes this into the two-param form needed by `Model.stream()` via `split_system_prompt()`.

## Defensive copies

Context fields (`messages`, `system_prompt`, `tool_specs`, `tool_choice`) are deep-copied when building the middleware context. `invocation_state` is shared by reference. `model_state` is excluded from the context entirely — middleware cannot access or modify it. The terminal reads it directly from the agent at invocation time.

## Context transformation

Middleware creates modified contexts via `dataclasses.replace()`:
```python
from dataclasses import replace
modified = replace(context, system_prompt="Injected")
```

When this goes public, we should add a typed `.replace()` method to context dataclasses for better discoverability and ergonomics (following `datetime.replace()` precedent).

## Generator cleanup

Python's `compose()` uses `try/finally` with explicit `aclose()`. TypeScript relies on `yield*` delegation which calls `.return()` automatically. Both correctly clean up generators.
