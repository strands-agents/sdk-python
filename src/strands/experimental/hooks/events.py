"""Hook events emitted as part of invoking Agents.

This module defines the events that are emitted as Agents run through the lifecycle of a request.
"""

from dataclasses import dataclass
from typing import Any, Optional

from ...types.tools import AgentTool, ToolResult, ToolUse
from .registry import HookEvent


@dataclass
class AgentInitializedEvent(HookEvent):
    """Event triggered when an agent has finished initialization.

    This event is fired after the agent has been fully constructed and all
    built-in components have been initialized. Hook providers can use this
    event to perform setup tasks that require a fully initialized agent.
    """

    pass


@dataclass
class StartRequestEvent(HookEvent):
    """Event triggered at the beginning of a new agent request.

    This event is fired when the agent begins processing a new user request,
    before any model inference or tool execution occurs. Hook providers can
    use this event to perform request-level setup, logging, or validation.

    This event is triggered at the beginning of the following api calls:
      - Agent.__call__
      - Agent.stream_async
      - Agent.structured_output
    """

    pass


@dataclass
class EndRequestEvent(HookEvent):
    """Event triggered at the end of an agent request.

    This event is fired after the agent has completed processing a request,
    regardless of whether it completed successfully or encountered an error.
    Hook providers can use this event for cleanup, logging, or state persistence.

    Note: This event uses reverse callback ordering, meaning callbacks registered
    later will be invoked first during cleanup.

    This event is triggered at the end of the following api calls:
      - Agent.__call__
      - Agent.stream_async
      - Agent.structured_output
    """

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order."""
        return True


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
        kwargs: Keyword arguments that will be passed to the tool.
    """

    selected_tool: Optional[AgentTool]
    tool_use: ToolUse
    kwargs: dict[str, Any]

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
        kwargs: Keyword arguments that were passed to the tool
        result: The result of the tool invocation. Either a ToolResult on success
            or an Exception if the tool execution failed.
    """

    selected_tool: Optional[AgentTool]
    tool_use: ToolUse
    kwargs: dict[str, Any]
    result: ToolResult
    exception: Optional[Exception] = None

    def _can_write(self, name: str) -> bool:
        return name == "result"

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order."""
        return True
