"""Multi-agent session management for persistent execution.

This package provides session persistence capabilities for multi-agent orchestrators,
enabling resumable execution after interruptions or failures.
"""

from .events import (
    AfterMultiAgentInvocationEvent,
    AfterNodeCallEvent,
    BeforeMultiAgentInvocationEvent,
    BeforeNodeCallEvent,
    MultiAgentInitializedEvent,
)

__all__ = [
    "AfterMultiAgentInvocationEvent",
    "AfterNodeCallEvent",
    "BeforeMultiAgentInvocationEvent",
    "BeforeNodeCallEvent",
    "MultiAgentInitializedEvent",
]
