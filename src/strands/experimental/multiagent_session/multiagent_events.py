"""Multi-agent execution lifecycle events for hook system integration.

This module defines event classes that are triggered at key points during
multi-agent orchestrator execution, enabling hooks to respond to lifecycle
events for purposes like persistence, monitoring, and debugging.

Event Types:
- Initialization: When orchestrator starts up
- Before/After Graph: Start/end of overall execution
- Before/After Node: Start/end of individual node execution
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...hooks.registry import HookEvent
from .multiagent_state import MultiAgentState

if TYPE_CHECKING:
    from ...multiagent.base import MultiAgentBase


@dataclass
class MultiAgentInitializationEvent(HookEvent):
    """Event triggered when multi-agent orchestrator initializes.

    Attributes:
        orchestrator: The multi-agent orchestrator instance
        state: Current state of the orchestrator
    """

    orchestrator: "MultiAgentBase"
    state: MultiAgentState


@dataclass
class BeforeGraphInvocationEvent(HookEvent):
    """Event triggered before orchestrator execution begins.

    Attributes:
        orchestrator: The multi-agent orchestrator instance
        state: Current state before execution starts
    """

    orchestrator: "MultiAgentBase"
    state: MultiAgentState


@dataclass
class BeforeNodeInvocationEvent(HookEvent):
    """Event triggered before individual node execution.

    Attributes:
        orchestrator: The multi-agent orchestrator instance
        next_node_to_execute: ID of the node about to be executed
    """

    orchestrator: "MultiAgentBase"
    next_node_to_execute: str


@dataclass
class AfterNodeInvocationEvent(HookEvent):
    """Event triggered after individual node execution completes.

    Attributes:
        orchestrator: The multi-agent orchestrator instance
        executed_node: ID of the node that just completed execution
        state: Updated state after node execution
    """

    orchestrator: "MultiAgentBase"
    executed_node: str
    state: MultiAgentState


@dataclass
class AfterGraphInvocationEvent(HookEvent):
    """Event triggered after orchestrator execution completes.

    Attributes:
        orchestrator: The multi-agent orchestrator instance
        state: Final state after execution completes
    """

    orchestrator: "MultiAgentBase"
    state: MultiAgentState
