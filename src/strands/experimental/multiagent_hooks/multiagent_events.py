"""Multi-agent execution lifecycle events for hook system integration.

These events are fired by orchestrators (Graph/Swarm) at key points so
hooks can persist, monitor, or debug execution. No intermediate state model
is used—hooks read from the orchestrator directly.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...hooks.registry import BaseHookEvent

if TYPE_CHECKING:
    from ...multiagent.base import MultiAgentBase


@dataclass
class MultiAgentInitializationEvent(BaseHookEvent):
    """Event triggered when multi-agent orchestrator initializes.

    Attributes:
        orchestrator: The multi-agent orchestrator instance
        invocation_state: Configuration that user pass in
    """

    orchestrator: "MultiAgentBase"
    invocation_state: dict[str, Any] | None = None


@dataclass
class AfterNodeInvocationEvent(BaseHookEvent):
    """Event triggered after individual node execution completes.

    Attributes:
        orchestrator: The multi-agent orchestrator instance
        executed_node: ID of the node that just completed execution
        invocation_state: Configuration that user pass in
    """

    orchestrator: "MultiAgentBase"
    executed_node: str
    invocation_state: dict[str, Any] | None = None


@dataclass
class AfterMultiAgentInvocationEvent(BaseHookEvent):
    """Event triggered after orchestrator execution completes.

    Attributes:
        orchestrator: The multi-agent orchestrator instance
        invocation_state: Configuration that user pass in
    """

    orchestrator: "MultiAgentBase"
    invocation_state: dict[str, Any] | None = None
