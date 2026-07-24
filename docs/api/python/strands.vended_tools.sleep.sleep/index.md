Sleep tool: pause execution for a bounded, cooperative duration.

Provides :func:`make_sleep` (a factory that lets the caller configure the maximum permitted duration) and :data:`sleep` (a default instance with a 60-second cap). Sleeps are implemented with :func:`asyncio.sleep`, which unblocks immediately when the surrounding task is cancelled. When the tool is invoked through the standard :class:`DecoratedFunctionTool` path, the raised :class:`asyncio.CancelledError` is caught by the tool executor and surfaced as a tool-error result; direct callers awaiting the underlying coroutine observe the cancellation directly.

#### make\_sleep

```python
def make_sleep(*,
               max_duration: float = DEFAULT_MAX_DURATION,
               name: str = "sleep",
               description: str | None = None) -> DecoratedFunctionTool
```

Defined in: [src/strands/vended\_tools/sleep/sleep.py:26](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/sleep/sleep.py#L26)

Create a sleep tool with a configurable maximum duration.

The returned tool pauses execution for `duration` seconds via :func:`asyncio.sleep`. Cancelling the surrounding task unblocks the sleep immediately rather than waiting for the full duration; the resulting :class:`asyncio.CancelledError` is caught by the standard tool executor when the tool is invoked through :class:`DecoratedFunctionTool` and surfaced as a tool-error result to the model.

**Arguments**:

-   `max_duration` - Upper bound on `duration` in seconds. Must be a finite, positive number. Defaults to :data:`DEFAULT_MAX_DURATION` (60 s).
-   `name` - Tool name. Defaults to `"sleep"`.
-   `description` - Tool description shown to the model.

**Returns**:

A decorated tool that pauses execution for the requested duration.

**Raises**:

-   `ValueError` - If `max_duration` is not a positive, finite number.

#### sleep

Default sleep tool with a 60-second maximum duration.