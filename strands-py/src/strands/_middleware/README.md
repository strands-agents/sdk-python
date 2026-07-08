# Python Middleware

This implementation follows the behavioral spec defined in `strands-ts/src/middleware/README.md` with the following intentional divergences:

## Scope

Only `InvokeModelStage` is implemented. `ExecuteToolStage` and `AgentStreamStage` will be added as needed.

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
