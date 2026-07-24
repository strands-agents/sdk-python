Tool for gracefully ending the agent loop.

This tool is experimental and subject to change in future revisions without notice.

Provides :func:`make_stop` (a factory for customized stop tools) and :data:`stop` (the default instance). The tool shims onto the SDK’s existing loop-termination primitive: it sets `invocation_state["request_state"]["stop_event_loop"] = True`, which the event loop already checks after tool execution (see :mod:`strands.event_loop.event_loop`). The tool returns the model-supplied message when one was given, or a default when the model passed `None` or an empty string; the returned value becomes the tool result the model sees for its stop request.

The Python event loop halts on this flag with `stop_reason == "tool_use"` and the final `AgentResult.message` set to the model’s tool-use assistant message (the batch that included the stop call). The tool’s returned string appears in history as the corresponding `toolResult`, not as a separate final assistant turn. This differs from the TypeScript side, whose `AfterToolsEvent.endTurn` primitive synthesizes a new assistant message with the stop text and `stopReason == "endTurn"`. Callers that need the stop text as the last assistant message on Python should read it from the tool result on the final message, or append it themselves.

#### DEFAULT\_MAX\_MESSAGE\_LENGTH

Default cap on the stop `message` length. The cap exists so a runaway model can’t blow the conversation history in one shot; adjust via `make_stop(max_message_length=...)` when a longer summary is legitimate.

#### make\_stop

```python
def make_stop(
    *,
    name: str = "stop",
    description: str = DEFAULT_STOP_DESCRIPTION,
    max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH
) -> DecoratedFunctionTool
```

Defined in: [src/strands/experimental/tools/stop/stop.py:71](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/tools/stop/stop.py#L71)

Create a stop tool that gracefully ends the agent loop.

The tool sets `invocation_state["request_state"]["stop_event_loop"] = True`, which the event loop checks after tool execution to end the loop without invoking the model again.

**Arguments**:

-   `name` - Tool name. Defaults to `"stop"`.
-   `description` - Tool description shown to the model.
-   `max_message_length` - Maximum accepted length for the model-supplied `message` argument, in characters. Must be a positive integer. Defaults to :data:`DEFAULT_MAX_MESSAGE_LENGTH` (4096).

**Returns**:

A decorated tool that signals the event loop to stop after the current tool batch completes.

**Raises**:

-   `ValueError` - If `max_message_length` is not a positive integer.

#### stop

Default stop tool. Ends the agent loop when called by the model.