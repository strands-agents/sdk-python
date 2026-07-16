"""Tool for gracefully ending the agent loop.

Provides :func:`make_stop` (a factory for customized stop tools) and :data:`stop`
(the default instance). The tool shims onto the SDK's existing loop-termination
primitive: it sets ``invocation_state["request_state"]["stop_event_loop"] = True``,
which the event loop already checks after tool execution
(see :mod:`strands.event_loop.event_loop`). The tool returns the model-supplied
message when one was given, or a default when the model passed ``None`` or an
empty string; the returned value becomes the tool result the model sees for
its stop request.

The Python event loop halts on this flag with ``stop_reason == "tool_use"`` and
the final ``AgentResult.message`` set to the model's tool-use assistant message
(the batch that included the stop call). The tool's returned string appears in
history as the corresponding ``toolResult``, not as a separate final assistant
turn. This differs from the TypeScript side, whose ``AfterToolsEvent.endTurn``
primitive synthesizes a new assistant message with the stop text and
``stopReason == "endTurn"``. Callers that need the stop text as the last
assistant message on Python should read it from the tool result on the final
message, or append it themselves.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...tools.decorator import tool
from ...types.tools import ToolContext

if TYPE_CHECKING:
    from ...tools.decorator import DecoratedFunctionTool

_DEFAULT_MESSAGE = "Agent loop stopped."
_MAX_MESSAGE_LENGTH = 4096

DEFAULT_STOP_DESCRIPTION = (
    "Gracefully ends the agent loop when the task is complete. "
    "Call this tool once with an optional final message when no further work is needed. "
    "This is a cooperative stop, not an abort: any tools already requested in this turn still run."
)


def _validate_message(message: str | None) -> str:
    """Validate an optional stop message and return the effective value.

    Args:
        message: The model-supplied message, or ``None`` for the default.

    Returns:
        The validated message string. Empty or ``None`` becomes the default so
        the assistant-facing final turn is never blank.

    Raises:
        ValueError: If ``message`` is not a string or exceeds the length cap.
    """
    if message is None or message == "":
        return _DEFAULT_MESSAGE
    if not isinstance(message, str):
        raise ValueError(f"`message` must be a string, got {type(message).__name__}")
    if len(message) > _MAX_MESSAGE_LENGTH:
        raise ValueError(f"`message` length exceeds the maximum of {_MAX_MESSAGE_LENGTH} characters")
    return message


def make_stop(
    *,
    name: str = "stop",
    description: str = DEFAULT_STOP_DESCRIPTION,
) -> DecoratedFunctionTool:
    """Create a stop tool that gracefully ends the agent loop.

    The tool sets ``invocation_state["request_state"]["stop_event_loop"] = True``,
    which the event loop checks after tool execution to end the loop without
    invoking the model again.

    Args:
        name: Tool name. Defaults to ``"stop"``.
        description: Tool description shown to the model.

    Returns:
        A decorated tool that signals the event loop to stop after the current
        tool batch completes.
    """

    @tool(name=name, description=description, context="tool_context")
    async def stop_tool(tool_context: ToolContext, message: str | None = None) -> str:
        """Ends the agent loop gracefully. Call once when the task is complete.

        Args:
            tool_context: Injected by the framework. Not user-facing.
            message: Optional final message describing why the loop is ending.
                Capped at 4096 characters; longer values are rejected.
        """
        final_message = _validate_message(message)
        request_state = tool_context.invocation_state.setdefault("request_state", {})
        request_state["stop_event_loop"] = True
        return final_message

    return stop_tool


stop = make_stop()
"""Default stop tool. Ends the agent loop when called by the model."""
