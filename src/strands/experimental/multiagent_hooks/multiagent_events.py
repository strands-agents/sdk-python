"""Multi-agent execution lifecycle events for hook system integration.

These events are fired by orchestrators (Graph/Swarm) at key points so
hooks can persist, monitor, or debug execution. No intermediate state model
is used—hooks read from the orchestrator directly.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...hooks.registry import HookEventBase

if TYPE_CHECKING:
    from ...multiagent.base import MultiAgentBase


@dataclass
class MultiAgentInitializationEvent(HookEventBase):
    """Event triggered when multi-agent orchestrator initializes.

    Attributes:
        orchestrator: The multi-agent orchestrator instance
        invocation_state: Configuration that user pass in
    """

    orchestrator: "MultiAgentBase"
    invocation_state: dict[str, Any]


@dataclass
class BeforeMultiAgentInvocationEvent(HookEventBase):
    """Event triggered before orchestrator execution begins.

    Attributes:
        orchestrator: The multi-agent orchestrator instance
        invocation_state: Configuration that user pass in
    """

    orchestrator: "MultiAgentBase"
    invocation_state: dict[str, Any]


@dataclass
class BeforeNodeInvocationEvent(HookEventBase):
    """Event triggered before individual node execution.

    Attributes:
        orchestrator: The multi-agent orchestrator instance
        invocation_state: Configuration that user pass in
    """

    orchestrator: "MultiAgentBase"
    next_node_to_execute: str
    invocation_state: dict[str, Any]


@dataclass
class AfterNodeInvocationEvent(HookEventBase):
    """Event triggered after individual node execution completes.

    Attributes:
        orchestrator: The multi-agent orchestrator instance
        executed_node: ID of the node that just completed execution
        invocation_state: Configuration that user pass in
    """

    orchestrator: "MultiAgentBase"
    executed_node: str
    invocation_state: dict[str, Any]


@dataclass
class AfterMultiAgentInvocationEvent(HookEventBase):
    """Event triggered after orchestrator execution completes.

    Attributes:
        orchestrator: The multi-agent orchestrator instance
        invocation_state: Configuration that user pass in
    """

    orchestrator: "MultiAgentBase"
    invocation_state: dict[str, Any]
