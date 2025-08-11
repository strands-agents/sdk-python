"""Types for agent invocation state and context."""

from typing import Any, Dict, TypedDict, TYPE_CHECKING

from opentelemetry.trace import Span

if TYPE_CHECKING:
    from ..agent import Agent
    from ..telemetry import Trace

class InvocationState(TypedDict, total=False):
    """Type definition for invocation_state used throughout the agent framework.

    This TypedDict defines the structure of the invocation_state dictionary that is
    passed through the agent's event loop and tool execution pipeline. All fields
    are optional since invocation_state is built incrementally during execution.

    Core Framework Fields:
        agent: The Agent instance executing the invocation (added for backward compatibility).
        event_loop_cycle_id: Unique identifier for the current event loop cycle.
        request_state: State dictionary maintained across event loop cycles.
        event_loop_cycle_trace: Trace object for monitoring the current cycle.
        event_loop_cycle_span: Span object for distributed tracing.
        event_loop_parent_cycle_id: UUID of the parent cycle for recursive calls. # always uuid? or just string?
        event_loop_parent_span: Parent span for tracing hierarchy.

    Additional Fields:
        Any additional keyword arguments passed during agent invocation or added
        by hooks and tools during execution are also included in this state.
    """

    # Core agent reference
    agent: "Agent"  # Forward reference to avoid circular imports

    # Event loop cycle management
    event_loop_cycle_id: str
    event_loop_parent_cycle_id: str

    # State management
    request_state: Dict[str, Any]

    # Tracing and monitoring
    event_loop_cycle_trace: "Trace"  # "Trace"  # Trace object type varies by implementation
    event_loop_cycle_span: Span   # Span object type varies by implementation
    event_loop_parent_span: Span  # Parent span for tracing hierarchy
