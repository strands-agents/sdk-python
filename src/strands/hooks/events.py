"""Hook events emitted as part of invoking Agents.

This module defines the events that are emitted as Agents run through the lifecycle of a request.
"""

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from typing_extensions import override

if TYPE_CHECKING:
    from ..agent.agent_result import AgentResult

from ..types.content import Message
from ..types.interrupt import _Interruptible
from ..types.streaming import StopReason
from ..types.tools import AgentTool, ToolResult, ToolUse
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
class BeforeInvocationEvent(HookEvent):
    """Event triggered at the beginning of a new agent request.

    This event is fired before the agent begins processing a new user request,
    before any model inference or tool execution occurs. Hook providers can
    use this event to perform request-level setup, logging, or validation.

    This event is triggered at the beginning of the following api calls:
      - Agent.__call__
      - Agent.stream_async
      - Agent.structured_output
    """

    pass


@dataclass
class AfterInvocationEvent(HookEvent):
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

    Attributes:
        result: The result of the agent invocation, if available.
            This will be None when invoked from structured_output methods, as those return typed output directly rather
            than AgentResult.
    """

    result: "AgentResult | None" = None

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order."""
        return True


@dataclass
class MessageAddedEvent(HookEvent):
    """Event triggered when a message is added to the agent's conversation.

    This event is fired whenever the agent adds a new message to its internal
    message history, including user messages, assistant responses, and tool
    results. Hook providers can use this event for logging, monitoring, or
    implementing custom message processing logic.

    Note: This event is only triggered for messages added by the framework
    itself, not for messages manually added by tools or external code.

    Attributes:
        message: The message that was added to the conversation history.
    """

    message: Message


@dataclass
class BeforeToolCallEvent(HookEvent, _Interruptible):
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
        cancel_tool: A user defined message that when set, will cancel the tool call.
            The message will be placed into a tool result with an error status. If set to `True`, Strands will cancel
            the tool call and use a default cancel message.
    """

    selected_tool: Optional[AgentTool]
    tool_use: ToolUse
    invocation_state: dict[str, Any]
    cancel_tool: bool | str = False

    def _can_write(self, name: str) -> bool:
        return name in ["cancel_tool", "selected_tool", "tool_use"]

    @override
    def _interrupt_id(self, name: str) -> str:
        """Unique id for the interrupt.

        Args:
            name: User defined name for the interrupt.

        Returns:
            Interrupt id.
        """
        return f"v1:before_tool_call:{self.tool_use['toolUseId']}:{uuid.uuid5(uuid.NAMESPACE_OID, name)}"


@dataclass
class AfterToolCallEvent(HookEvent):
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
        cancel_message: The cancellation message if the user cancelled the tool call.
    """

    selected_tool: Optional[AgentTool]
    tool_use: ToolUse
    invocation_state: dict[str, Any]
    result: ToolResult
    exception: Optional[Exception] = None
    cancel_message: str | None = None

    def _can_write(self, name: str) -> bool:
        return name == "result"

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order."""
        return True


@dataclass
class BeforeModelCallEvent(HookEvent):
    """Event triggered before the model is invoked.

    This event is fired just before the agent calls the model for inference,
    allowing hook providers to inspect or modify the messages and configuration
    that will be sent to the model.

    Note: This event is not fired for invocations to structured_output.
    """

    pass


@dataclass
class AfterModelCallEvent(HookEvent):
    """Event triggered after the model invocation completes.

    This event is fired after the agent has finished calling the model,
    regardless of whether the invocation was successful or resulted in an error.
    Hook providers can use this event for cleanup, logging, or post-processing.

    Note: This event uses reverse callback ordering, meaning callbacks registered
    later will be invoked first during cleanup.

    Note: This event is not fired for invocations to structured_output.

    Model Retrying:
        When ``retry_model`` is set to True by a hook callback, the agent will discard
        the current model response and invoke the model again. This has important
        implications for streaming consumers:

        - Streaming events from the discarded response will have already been emitted
          to callers before the retry occurs. Agent invokers consuming streamed events
          should be prepared to handle this scenario, potentially by tracking retry state
          or implementing idempotent event processing
        - The original model message is thrown away internally and not added to the
          conversation history

    Attributes:
        stop_response: The model response data if invocation was successful, None if failed.
        exception: Exception if the model invocation failed, None if successful.
        retry: Whether to retry the model invocation. Can be set by hook callbacks
            to trigger a retry. When True, the current response is discarded and the
            model is called again. Defaults to False.
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
    retry: bool = False

    def _can_write(self, name: str) -> bool:
        return name == "retry"

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order."""
        return True
