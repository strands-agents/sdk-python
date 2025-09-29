"""Multi-agent session management for persistent execution.

This package provides session persistence capabilities for multi-agent orchestrators,
enabling resumable execution after interruptions or failures.
"""

from .multiagent_events import (
    AfterMultiAgentInvocationEvent,
    AfterNodeInvocationEvent,
    BeforeMultiAgentInvocationEvent,
    BeforeNodeInvocationEvent,
    MultiAgentInitializationEvent,
)
from .persistence_hooks import PersistentHook

__all__ = [
    "BeforeMultiAgentInvocationEvent",
    "AfterMultiAgentInvocationEvent",
    "MultiAgentInitializationEvent",
    "BeforeNodeInvocationEvent",
    "AfterNodeInvocationEvent",
    "PersistentHook",
]
