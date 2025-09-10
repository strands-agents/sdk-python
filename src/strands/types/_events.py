"""event system for the Strands Agents framework.

This module defines the event types that are emitted during agent execution,
providing a structured way to observe to different events of the event loop and
agent lifecycle.
"""

from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

from ..telemetry import EventLoopMetrics
from .citations import Citation
from .content import Message
from .event_loop import Metrics, StopReason, Usage
from .streaming import ContentBlockDelta, StreamEvent
from .tools import ToolResult, ToolUse

if TYPE_CHECKING:
    from ..agent import AgentResult


class TypedEvent(dict):
    """Base class for all typed events in the agent system."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize the typed event with optional data.

        Args:
            data: Optional dictionary of event data to initialize with
        """
        super().__init__(data or {})

    @property
    def is_callback_event(self) -> bool:
        """True if this event should trigger the callback_handler to fire."""
        return True

    def as_dict(self) -> dict:
        """Convert this event to a raw dictionary for emitting purposes."""
        return {**self}

    def prepare(self, invocation_state: dict) -> None:
        """Prepare the event for emission by adding invocation state.

        This allows a subset of events to merge with the invocation_state without needing to
        pass around the invocation_state throughout the system.
        """
        ...


class InitEventLoopEvent(TypedEvent):
    """Event emitted at the very beginning of agent execution.

    This event is fired before any processing begins and provides access to the
    initial invocation state.

    Args:
            invocation_state: The invocation state passed into the request
    """

    def __init__(self) -> None:
        """Initialize the event loop initialization event."""
        super().__init__({"init_event_loop": True})

    @override
    def prepare(self, invocation_state: dict) -> None:
        self.update(invocation_state)


class StartEvent(TypedEvent):
    """Event emitted at the start of each event loop cycle.

    !!deprecated!!
        Use StartEventLoopEvent instead.

    This event events the beginning of a new processing cycle within the agent's
    event loop. It's fired before model invocation and tool execution begin.
    """

    def __init__(self) -> None:
        """Initialize the event loop start event."""
        super().__init__({"start": True})


class StartEventLoopEvent(TypedEvent):
    """Event emitted when the event loop cycle begins processing.

    This event is fired after StartEvent and indicates that the event loop
    has begun its core processing logic, including model invocation preparation.
    """

    def __init__(self) -> None:
        """Initialize the event loop processing start event."""
        super().__init__({"start_event_loop": True})


class ModelStreamChunkEvent(TypedEvent):
    """Event emitted during model response streaming for each raw chunk."""

    def __init__(self, chunk: StreamEvent) -> None:
        """Initialize with streaming delta data from the model.

        Args:
            chunk: Incremental streaming data from the model response
        """
        super().__init__({"event": chunk})

    @property
    def chunk(self) -> StreamEvent:
        return cast(StreamEvent, self.get("event"))


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

    @property
    def is_callback_event(self) -> bool:
        # Only invoke a callback if we're non-empty
        return len(self.keys()) > 0

    @override
    def prepare(self, invocation_state: dict) -> None:
        if "delta" in self:
            self.update(invocation_state)


class ToolUseStreamEvent(ModelStreamEvent):
    """Event emitted during tool use input streaming."""

    def __init__(self, delta: ContentBlockDelta, current_tool_use: dict[str, Any]) -> None:
        """Initialize with delta and current tool use state."""
        super().__init__({"delta": delta, "current_tool_use": current_tool_use})


class TextStreamEvent(ModelStreamEvent):
    """Event emitted during text content streaming."""

    def __init__(self, delta: ContentBlockDelta, text: str) -> None:
        """Initialize with delta and text content."""
        super().__init__({"data": text, "delta": delta})


class CitationStreamEvent(ModelStreamEvent):
    """Event emitted during citation streaming."""

    def __init__(self, delta: ContentBlockDelta, citation: Citation) -> None:
        """Initialize with delta and citation content."""
        super().__init__({"callback": {"citation": citation, "delta": delta}})


class ReasoningTextStreamEvent(ModelStreamEvent):
    """Event emitted during reasoning text streaming."""

    def __init__(self, delta: ContentBlockDelta, reasoning_text: str | None) -> None:
        """Initialize with delta and reasoning text."""
        super().__init__({"reasoningText": reasoning_text, "delta": delta, "reasoning": True})


class ReasoningSignatureStreamEvent(ModelStreamEvent):
    """Event emitted during reasoning signature streaming."""

    def __init__(self, delta: ContentBlockDelta, reasoning_signature: str | None) -> None:
        """Initialize with delta and reasoning signature."""
        super().__init__({"reasoning_signature": reasoning_signature, "delta": delta, "reasoning": True})


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

    @property
    @override
    def is_callback_event(self) -> bool:
        return False


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

    @property
    @override
    def is_callback_event(self) -> bool:
        return False


class EventLoopThrottleEvent(TypedEvent):
    """Event emitted when the event loop is throttled due to rate limiting."""

    def __init__(self, delay: int) -> None:
        """Initialize with the throttle delay duration.

        Args:
            delay: Delay in seconds before the next retry attempt
        """
        super().__init__({"event_loop_throttled_delay": delay})

    @override
    def prepare(self, invocation_state: dict) -> None:
        self.update(invocation_state)


class ToolResultEvent(TypedEvent):
    """Event emitted when a tool execution completes."""

    def __init__(self, tool_result: ToolResult) -> None:
        """Initialize with the completed tool result.

        Args:
            tool_result: Final result from the tool execution
        """
        super().__init__({"tool_result": tool_result})

    @property
    def tool_use_id(self) -> str:
        """The toolUseId associated with this result."""
        return cast(str, cast(ToolResult, self.get("tool_result")).get("toolUseId"))

    @property
    def tool_result(self) -> ToolResult:
        """Final result from the completed tool execution."""
        return cast(ToolResult, self.get("tool_result"))

    @property
    @override
    def is_callback_event(self) -> bool:
        return False


class ToolStreamEvent(TypedEvent):
    """Event emitted when a tool yields sub-events as part of tool execution."""

    def __init__(self, tool_use: ToolUse, tool_stream_data: Any) -> None:
        """Initialize with tool streaming data.

        Args:
            tool_use: The tool invocation producing the stream
            tool_stream_data: The yielded event from the tool execution
        """
        super().__init__({"tool_stream_event": {"tool_use": tool_use, "data": tool_stream_data}})

    @property
    def tool_use_id(self) -> str:
        """The toolUseId associated with this stream."""
        return cast(str, cast(ToolUse, cast(dict, self.get("tool_stream_event")).get("tool_use")).get("toolUseId"))


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
        super().__init__({"message": message})


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
        super().__init__({"message": message})


class ForceStopEvent(TypedEvent):
    """Event emitted when the agent execution is forcibly stopped, either by a tool or by an exception."""

    def __init__(self, reason: str | Exception) -> None:
        """Initialize with the reason for forced stop.

        Args:
            reason: String description or exception that caused the forced stop
        """
        super().__init__(
            {
                "force_stop": True,
                "force_stop_reason": str(reason),
            }
        )


class AgentResultEvent(TypedEvent):
    def __init__(self, result: "AgentResult"):
        super().__init__({"result": result})
