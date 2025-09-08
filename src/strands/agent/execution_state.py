"""Agent execution state."""

from enum import Enum


class ExecutionState(Enum):
    """Represents the current execution state of an agent.

    ASSISTANT: Agent is waiting for user message (default).
    INTERRUPT: Agent is waiting for user feedback to resume tool execution.
    """
    
    ASSISTANT = "assistant"
    INTERRUPT = "interrupt"
