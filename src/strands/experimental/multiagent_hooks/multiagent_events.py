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
class MultiagentInitializedEvent(BaseHookEvent):
    """Event triggered when multi-agent orchestrator initialized.

    Attributes:
        source: The multi-agent orchestrator instance
        invocation_state: Configuration that user pass in
    """

    source: "MultiAgentBase"
    invocation_state: dict[str, Any] | None = None


@dataclass
class BeforeNodeInvocationEvent(BaseHookEvent):
    """Event triggered before individual node execution completes. This event corresponds to the After event."""

    source: "MultiAgentBase"
    invocation_state: dict[str, Any] | None = None


@dataclass
class AfterNodeInvocationEvent(BaseHookEvent):
    """Event triggered after individual node execution completes.

    Attributes:
        source: The multi-agent orchestrator instance
        executed_node: ID of the node that just completed execution
        invocation_state: Configuration that user pass in
    """

    source: "MultiAgentBase"
    executed_node: str
    invocation_state: dict[str, Any] | None = None

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order."""
        return True


@dataclass
class BeforeMultiAgentInvocationEvent(BaseHookEvent):
    """Event triggered after orchestrator execution completes. This event corresponds to the After event.

    Attributes:
        source: The multi-agent orchestrator instance
        invocation_state: Configuration that user pass in
    """

    source: "MultiAgentBase"
    invocation_state: dict[str, Any] | None = None


@dataclass
class AfterMultiAgentInvocationEvent(BaseHookEvent):
    """Event triggered after orchestrator execution completes.

    Attributes:
        source: The multi-agent orchestrator instance
        invocation_state: Configuration that user pass in
    """

    source: "MultiAgentBase"
    invocation_state: dict[str, Any] | None = None
