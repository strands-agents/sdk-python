# Python Middleware

This implementation follows the behavioral spec defined in `strands-ts/src/middleware/README.md` with the following intentional divergences:

## Scope

Only `InvokeModelStage` is implemented. `ExecuteToolStage` and `AgentStreamStage` will be added as needed.

## Result encoding

Python async generators cannot `return` values. The implementation uses a `MiddlewareResult` sentinel yielded as the last item in the generator. Integration sites strip it from the event stream. Pass-through is:

```python
async def passthrough(context, next_fn):
    async for event in next_fn(context):
        yield event
```

## No removal / cleanup

Once registered, middleware cannot be removed. This matches the Python hook system which also does not support removal.

## Private module

The `_middleware/` package is not part of the public API. Internal consumers access it via `agent._middleware_registry.add_middleware(...)`.

## System prompt as a union type

`InvokeModelContext.system_prompt` is `str | list[SystemContentBlock] | None` (a single union field). The terminal decomposes this into the two-param form needed by `Model.stream()` via `split_system_prompt()`.

## Defensive copies

Context fields (`messages`, `system_prompt`, `tool_specs`, `tool_choice`) are deep-copied when building the middleware context. `invocation_state` is shared by reference. `model_state` is not on the context — the terminal reads it directly from the agent.

## Context transformation

Middleware creates modified contexts via `dataclasses.replace()`:
```python
from dataclasses import replace
modified = replace(context, system_prompt="Injected")
```

When this goes public, we should add a typed `.replace()` method to context dataclasses for better discoverability and ergonomics (following `datetime.replace()` precedent).

## Generator cleanup

Python's `compose()` uses `try/finally` with explicit `aclose()`. TypeScript relies on `yield*` delegation which calls `.return()` automatically. Both correctly clean up generators.
