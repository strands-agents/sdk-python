"""Experimental hook events emitted as part of invoking Agents and BidiAgents.

This module defines the events that are emitted as Agents and BidiAgents run through the lifecycle of a request.
"""

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from ...hooks.events import AfterModelCallEvent, AfterToolCallEvent, BeforeModelCallEvent, BeforeToolCallEvent
from ...hooks.registry import BaseHookEvent
from ...types.content import Message
from ...types.tools import AgentTool, ToolResult, ToolUse

if TYPE_CHECKING:
    from ..bidi.agent.agent import BidiAgent
    from ..bidi.models import BidiModelTimeoutError

# Deprecated aliases - warning emitted on access via __getattr__
_DEPRECATED_ALIASES = {
    "BeforeToolInvocationEvent": BeforeToolCallEvent,
    "AfterToolInvocationEvent": AfterToolCallEvent,
    "BeforeModelInvocationEvent": BeforeModelCallEvent,
    "AfterModelInvocationEvent": AfterModelCallEvent,
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_ALIASES:
        warnings.warn(
            f"{name} has been moved to production with an updated name. "
            f"Use {_DEPRECATED_ALIASES[name].__name__} from strands.hooks instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return _DEPRECATED_ALIASES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# BidiAgent Hook Events


@dataclass
class BidiHookEvent(BaseHookEvent):
    """Base class for BidiAgent hook events.

    Attributes:
        agent: The BidiAgent instance that triggered this event.
    """

    agent: "BidiAgent"


@dataclass
class BidiAgentInitializedEvent(BidiHookEvent):
    """Event triggered when a BidiAgent has finished initialization.

    This event is fired after the BidiAgent has been fully constructed and all
    built-in components have been initialized. Hook providers can use this
    event to perform setup tasks that require a fully initialized agent.
    """

    pass


@dataclass
class BidiBeforeInvocationEvent(BidiHookEvent):
    """Event triggered when BidiAgent starts a streaming session.

    This event is fired before the BidiAgent begins a streaming session,
    before any model connection or audio processing occurs. Hook providers can
    use this event to perform session-level setup, logging, or validation.

    This event is triggered at the beginning of agent.start().
    """

    pass


@dataclass
class BidiAfterInvocationEvent(BidiHookEvent):
    """Event triggered when BidiAgent ends a streaming session.

    This event is fired after the BidiAgent has completed a streaming session,
    regardless of whether it completed successfully or encountered an error.
    Hook providers can use this event for cleanup, logging, or state persistence.

    Note: This event uses reverse callback ordering, meaning callbacks registered
    later will be invoked first during cleanup.

    This event is triggered at the end of agent.stop().
    """

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order."""
        return True


@dataclass
class BidiMessageAddedEvent(BidiHookEvent):
    """Event triggered when BidiAgent adds a message to the conversation.

    This event is fired whenever the BidiAgent adds a new message to its internal
    message history, including user messages (from transcripts), assistant responses,
    and tool results. Hook providers can use this event for logging, monitoring, or
    implementing custom message processing logic.

    Note: This event is only triggered for messages added by the framework
    itself, not for messages manually added by tools or external code.

    Attributes:
        message: The message that was added to the conversation history.
    """

    message: Message


@dataclass
class BidiBeforeToolCallEvent(BidiHookEvent):
    """Event triggered before BidiAgent executes a tool.

    This event is fired just before the BidiAgent executes a tool during a streaming
    session, allowing hook providers to inspect, modify, or replace the tool that
    will be executed. The selected_tool can be modified by hook callbacks to change
    which tool gets executed.

    Attributes:
        selected_tool: The tool that will be invoked. Can be modified by hooks
            to change which tool gets executed. This may be None if tool lookup failed.
        tool_use: The tool parameters that will be passed to selected_tool.
        invocation_state: Keyword arguments that will be passed to the tool.
        cancel_tool: A user defined message that when set, will cancel the tool call.
            The message will be placed into a tool result with an error status. If set to `True`, Strands will cancel
            the tool call and use a default cancel message.
    """

    selected_tool: AgentTool | None
    tool_use: ToolUse
    invocation_state: dict[str, Any]
    cancel_tool: bool | str = False

    def _can_write(self, name: str) -> bool:
        return name in ["cancel_tool", "selected_tool", "tool_use"]


@dataclass
class BidiAfterToolCallEvent(BidiHookEvent):
    """Event triggered after BidiAgent executes a tool.

    This event is fired after the BidiAgent has finished executing a tool during
    a streaming session, regardless of whether the execution was successful or
    resulted in an error. Hook providers can use this event for cleanup, logging,
    or post-processing.

    Note: This event uses reverse callback ordering, meaning callbacks registered
    later will be invoked first during cleanup.

    Attributes:
        selected_tool: The tool that was invoked. It may be None if tool lookup failed.
        tool_use: The tool parameters that were passed to the tool invoked.
        invocation_state: Keyword arguments that were passed to the tool.
        result: The result of the tool invocation. Either a ToolResult on success
            or an Exception if the tool execution failed.
        exception: Exception if the tool execution failed, None if successful.
        cancel_message: The cancellation message if the user cancelled the tool call.
    """

    selected_tool: AgentTool | None
    tool_use: ToolUse
    invocation_state: dict[str, Any]
    result: ToolResult
    exception: Exception | None = None
    cancel_message: str | None = None

    def _can_write(self, name: str) -> bool:
        return name == "result"

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order."""
        return True


@dataclass
class BidiInterruptionEvent(BidiHookEvent):
    """Event triggered when model generation is interrupted.

    This event is fired when the user interrupts the assistant (e.g., by speaking
    during the assistant's response) or when an error causes interruption. This is
    specific to bidirectional streaming and doesn't exist in standard agents.

    Hook providers can use this event to log interruptions, implement custom
    interruption handling, or trigger cleanup logic.

    Attributes:
        reason: The reason for the interruption ("user_speech" or "error").
        interrupted_response_id: Optional ID of the response that was interrupted.
    """

    reason: Literal["user_speech", "error"]
    interrupted_response_id: str | None = None


@dataclass
class BidiBeforeConnectionRestartEvent(BidiHookEvent):
    """Event emitted before agent attempts to restart model connection after timeout.

    Attributes:
        timeout_error: Timeout error reported by the model.
    """

    timeout_error: "BidiModelTimeoutError"


@dataclass
class BidiAfterConnectionRestartEvent(BidiHookEvent):
    """Event emitted after agent attempts to restart model connection after timeout.

    Attribtues:
        exception: Populated if exception was raised during connection restart.
            None value means the restart was successful.
    """

    exception: Exception | None = None
