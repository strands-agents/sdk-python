"""Experimental hook events emitted as part of invoking Agents.

This module defines the events that are emitted as Agents run through the lifecycle of a request.
"""

from dataclasses import dataclass
from typing import Any, Optional

from ...hooks import HookEvent
from ...types.content import Message
from ...types.streaming import StopReason
from ...types.tools import AgentTool, ToolResult, ToolUse


@dataclass
class BeforeToolInvocationEvent(HookEvent):
    """Event triggered before a tool is invoked.

    This event is fired just before the agent executes a tool, allowing hook
    providers to inspect, modify, or replace the tool that will be executed.
    The selected_tool can be modified by hook callbacks to change which tool
    gets executed.

    Attributes:
        selected_tool: The tool that will be invoked. Can be modified by hooks
            to change which tool gets executed. This may be None if tool lookup failed.
        tool_use: The tool parameters that will be passed to selected_tool.
        invocation_state: Keyword arguments that will be passed to the tool.
    """

    selected_tool: Optional[AgentTool]
    tool_use: ToolUse
    invocation_state: dict[str, Any]

    def _can_write(self, name: str) -> bool:
        return name in ["selected_tool", "tool_use"]


@dataclass
class AfterToolInvocationEvent(HookEvent):
    """Event triggered after a tool invocation completes.

    This event is fired after the agent has finished executing a tool,
    regardless of whether the execution was successful or resulted in an error.
    Hook providers can use this event for cleanup, logging, or post-processing.

    Note: This event uses reverse callback ordering, meaning callbacks registered
    later will be invoked first during cleanup.

    Attributes:
        selected_tool: The tool that was invoked. It may be None if tool lookup failed.
        tool_use: The tool parameters that were passed to the tool invoked.
        invocation_state: Keyword arguments that were passed to the tool
        result: The result of the tool invocation. Either a ToolResult on success
            or an Exception if the tool execution failed.
    """

    selected_tool: Optional[AgentTool]
    tool_use: ToolUse
    invocation_state: dict[str, Any]
    result: ToolResult
    exception: Optional[Exception] = None

    def _can_write(self, name: str) -> bool:
        return name == "result"

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order."""
        return True


@dataclass
class BeforeModelInvocationEvent(HookEvent):
    """Event triggered before the model is invoked.

    This event is fired just before the agent calls the model for inference,
    allowing hook providers to inspect or modify the messages and configuration
    that will be sent to the model.

    Note: This event is not fired for invocations to structured_output.
    """

    pass


@dataclass
class AfterModelInvocationEvent(HookEvent):
    """Event triggered after the model invocation completes.

    This event is fired after the agent has finished calling the model,
    regardless of whether the invocation was successful or resulted in an error.
    Hook providers can use this event for cleanup, logging, or post-processing.

    Note: This event uses reverse callback ordering, meaning callbacks registered
    later will be invoked first during cleanup.

    Note: This event is not fired for invocations to structured_output.

    Attributes:
        stop_response: The model response data if invocation was successful, None if failed.
        exception: Exception if the model invocation failed, None if successful.
    """

    @dataclass
    class ModelStopResponse:
        """Model response data from successful invocation.

        Attributes:
            stop_reason: The reason the model stopped generating.
            message: The generated message from the model.
        """

        message: Message
        stop_reason: StopReason

    stop_response: Optional[ModelStopResponse] = None
    exception: Optional[Exception] = None

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order."""
        return True
