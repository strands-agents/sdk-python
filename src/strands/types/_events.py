"""event system for the Strands Agents framework.

This module defines the event types that are emitted during agent execution,
providing a structured way to observe to different events of the event loop and
agent lifecycle.
"""

from typing import TYPE_CHECKING, Any

from ..telemetry import EventLoopMetrics
from .content import Message
from .event_loop import Metrics, StopReason, Usage
from .streaming import ContentBlockDelta, StreamEvent

if TYPE_CHECKING:
    pass


class TypedEvent(dict):
    """Base class for all typed events in the agent system."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize the typed event with optional data.

        Args:
            data: Optional dictionary of event data to initialize with
        """
        super().__init__(data or {})


class InitEventLoopEvent(TypedEvent):
    """Event emitted at the very beginning of agent execution.

    This event is fired before any processing begins and provides access to the
    initial invocation state.

    Args:
            invocation_state: The invocation state passed into the request
    """

    def __init__(self, invocation_state: dict) -> None:
        """Initialize the event loop initialization event."""
        super().__init__({"callback": {"init_event_loop": True, **invocation_state}})


class StartEvent(TypedEvent):
    """Event emitted at the start of each event loop cycle.

    !!deprecated!!
        Use StartEventLoopEvent instead.

    This event events the beginning of a new processing cycle within the agent's
    event loop. It's fired before model invocation and tool execution begin.
    """

    def __init__(self) -> None:
        """Initialize the event loop start event."""
        super().__init__({"callback": {"start": True}})


class StartEventLoopEvent(TypedEvent):
    """Event emitted when the event loop cycle begins processing.

    This event is fired after StartEvent and indicates that the event loop
    has begun its core processing logic, including model invocation preparation.
    """

    def __init__(self) -> None:
        """Initialize the event loop processing start event."""
        super().__init__({"callback": {"start_event_loop": True}})


class ModelStreamChunkEvent(TypedEvent):
    """Event emitted during model response streaming for each raw chunk."""

    def __init__(self, chunk: StreamEvent) -> None:
        """Initialize with streaming delta data from the model.

        Args:
            chunk: Incremental streaming data from the model response
        """
        super().__init__({"callback": {"event": chunk}})


class ModelStreamEvent(TypedEvent):
    """Event emitted during model response streaming.

    This event is fired when the model produces streaming output during response
    generation.
    """

    def __init__(self, delta_data: dict[str, Any]) -> None:
        """Initialize with streaming delta data from the model.

        Args:
            delta_data: Incremental streaming data from the model response
        """
        super().__init__(delta_data)


class ToolUseStreamEvent(ModelStreamEvent):
    """Event emitted during tool use input streaming."""

    def __init__(self, delta: ContentBlockDelta, current_tool_use: dict[str, Any]) -> None:
        """Initialize with delta and current tool use state."""
        super().__init__({"callback": {"delta": delta, "current_tool_use": current_tool_use}})


class TextStreamEvent(ModelStreamEvent):
    """Event emitted during text content streaming."""

    def __init__(self, delta: ContentBlockDelta, text: str) -> None:
        """Initialize with delta and text content."""
        super().__init__({"callback": {"data": text, "delta": delta}})


class ReasoningTextStreamEvent(ModelStreamEvent):
    """Event emitted during reasoning text streaming."""

    def __init__(self, delta: ContentBlockDelta, reasoning_text: str | None) -> None:
        """Initialize with delta and reasoning text."""
        super().__init__({"callback": {"reasoningText": reasoning_text, "delta": delta, "reasoning": True}})


class ReasoningSignatureStreamEvent(ModelStreamEvent):
    """Event emitted during reasoning signature streaming."""

    def __init__(self, delta: ContentBlockDelta, reasoning_signature: str | None) -> None:
        """Initialize with delta and reasoning signature."""
        super().__init__({"callback": {"reasoning_signature": reasoning_signature, "delta": delta, "reasoning": True}})


class ModelStopReason(TypedEvent):
    """Event emitted during reasoning signature streaming."""

    def __init__(
        self,
        stop_reason: StopReason,
        message: Message,
        usage: Usage,
        metrics: Metrics,
    ) -> None:
        """Initialize with the final execution results.

        Args:
            stop_reason: Why the agent execution stopped
            message: Final message from the model
            usage: Usage information from the model
            metrics: Execution metrics and performance data
        """
        super().__init__({"stop": (stop_reason, message, usage, metrics)})


class EventLoopStopEvent(TypedEvent):
    """Event emitted when the agent execution completes normally."""

    def __init__(
        self,
        stop_reason: StopReason,
        message: Message,
        metrics: "EventLoopMetrics",
        request_state: Any,
    ) -> None:
        """Initialize with the final execution results.

        Args:
            stop_reason: Why the agent execution stopped
            message: Final message from the model
            metrics: Execution metrics and performance data
            request_state: Final state of the agent execution
        """
        super().__init__({"stop": (stop_reason, message, metrics, request_state)})


class EventLoopThrottleEvent(TypedEvent):
    """Event emitted when the event loop is throttled due to rate limiting."""

    def __init__(self, delay: int, invocation_state: dict[str, Any]) -> None:
        """Initialize with the throttle delay duration.

        Args:
            delay: Delay in seconds before the next retry attempt
            invocation_state: The invocation state passed into the request
        """
        super().__init__({"callback": {"event_loop_throttled_delay": delay, **invocation_state}})


class ModelMessageEvent(TypedEvent):
    """Event emitted when the model invocation has completed.

    This event is fired whenever the model generates a response message that
    gets added to the conversation history.
    """

    def __init__(self, message: Message) -> None:
        """Initialize with the model-generated message.

        Args:
            message: The response message from the model
        """
        super().__init__({"callback": {"message": message}})


class ToolResultMessageEvent(TypedEvent):
    """Event emitted when tool results are formatted as a message.

    This event is fired when tool execution results are converted into a
    message format to be added to the conversation history. It provides
    access to the formatted message containing tool results.
    """

    def __init__(self, message: Any) -> None:
        """Initialize with the model-generated message.

        Args:
            message: Message containing tool results for conversation history
        """
        super().__init__({"callback": {"message": message}})


class ForceStopEvent(TypedEvent):
    """Event emitted when the agent execution is forcibly stopped, either by a tool or by an exception."""

    def __init__(self, reason: str | Exception) -> None:
        """Initialize with the reason for forced stop.

        Args:
            reason: String description or exception that caused the forced stop
        """
        super().__init__(
            {
                "callback": {
                    "force_stop": True,
                    "force_stop_reason": str(reason),
                    # "force_stop_reason_exception": reason if reason and isinstance(reason, Exception) else MISSING,
                }
            }
        )
