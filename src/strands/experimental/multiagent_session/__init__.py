"""Multi-agent session management for persistent execution.

This package provides session persistence capabilities for multi-agent orchestrators,
enabling resumable execution after interruptions or failures.
"""

from .multiagent_events import (
    AfterGraphInvocationEvent,
    AfterNodeInvocationEvent,
    BeforeGraphInvocationEvent,
    BeforeNodeInvocationEvent,
    MultiAgentInitializationEvent,
)
from .multiagent_state import MultiAgentState, MultiAgentType
from .multiagent_state_adapter import MultiAgentAdapter

__all__ = [
    "BeforeGraphInvocationEvent",
    "AfterGraphInvocationEvent",
    "MultiAgentInitializationEvent",
    "BeforeNodeInvocationEvent",
    "AfterNodeInvocationEvent",
    "MultiAgentState",
    "MultiAgentAdapter",
    "MultiAgentType",
]
