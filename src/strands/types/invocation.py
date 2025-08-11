"""Types for agent invocation state and context."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict
from uuid import UUID

from opentelemetry.trace import Span

if TYPE_CHECKING:
    from ..agent import Agent
    from ..models.model import Model
    from ..telemetry import Trace
    from ..types.content import Message
    from ..types.tools import ToolConfig


class InvocationState(TypedDict, total=False):
    """Type definition for invocation_state used throughout the agent framework.

    This TypedDict defines the structure of the invocation_state dictionary that is
    passed through the agent's event loop and tool execution pipeline. All fields
    are optional since invocation_state is built incrementally during execution.

    Core Framework Fields:
        agent: The Agent instance executing the invocation (added for backward compatibility).
        event_loop_cycle_id: Unique identifier for the current event loop cycle.
        event_loop_parent_cycle_id: id of the parent cycle for recursive calls.
        request_state: State dictionary maintained across event loop cycles.
        event_loop_cycle_trace: Trace object for monitoring the current cycle.
        event_loop_cycle_span: Span object for distributed tracing.
        event_loop_parent_span: Parent span for tracing hierarchy.

    Agent Context Fields:
        model: The model instance used by the agent for inference.
        system_prompt: The system prompt used to guide the agent's behavior.
        messages: The conversation history as a list of messages.
        tool_config: Configuration for tools available to the agent.

    Additional Fields:
        Any additional keyword arguments passed during agent invocation or added
        by hooks and tools during execution are also included in this state.
    """

    # Core agent reference
    agent: "Agent"  # Forward reference to avoid circular imports

    # Event loop cycle management
    event_loop_cycle_id: UUID
    event_loop_parent_cycle_id: UUID

    # State management
    request_state: Dict[str, Any]

    # Tracing and monitoring
    event_loop_cycle_trace: "Trace"  # "Trace"  # Trace object type varies by implementation
    event_loop_cycle_span: Span | None  # Span object type varies by implementation
    event_loop_parent_span: Span | None  # Parent span for tracing hierarchy

    # Agent context fields
    model: "Model"  # The model instance used by the agent
    system_prompt: Optional[str]  # The system prompt for the agent
    messages: List["Message"]  # The conversation history
    tool_config: "ToolConfig"  # Configuration for available tools
