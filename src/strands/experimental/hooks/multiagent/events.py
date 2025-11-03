"""Multi-agent execution lifecycle events for hook system integration.

These events are fired by orchestrators (Graph/Swarm) at key points so
hooks can persist, monitor, or debug execution. No intermediate state model
is used—hooks read from the orchestrator directly.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ....hooks import BaseHookEvent

if TYPE_CHECKING:
    from ....multiagent.base import MultiAgentBase


@dataclass
class MultiAgentInitializedEvent(BaseHookEvent):
    """Event triggered when multi-agent orchestrator initialized.

    Attributes:
        source: The multi-agent orchestrator instance
        invocation_state: Configuration that user passes in
    """

    source: "MultiAgentBase"
    invocation_state: dict[str, Any] | None = None


@dataclass
class BeforeNodeCallEvent(BaseHookEvent):
    """Event triggered before individual node execution starts.

    Attributes:
        source: The multi-agent orchestrator instance
        node_id: ID of the node about to execute
        invocation_state: Configuration that user passes in
    """

    source: "MultiAgentBase"
    node_id: str
    invocation_state: dict[str, Any] | None = None


@dataclass
class AfterNodeCallEvent(BaseHookEvent):
    """Event triggered after individual node execution completes.

    Attributes:
        source: The multi-agent orchestrator instance
        node_id: ID of the node that just completed execution
        invocation_state: Configuration that user passes in
    """

    source: "MultiAgentBase"
    node_id: str
    invocation_state: dict[str, Any] | None = None

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order."""
        return True


@dataclass
class BeforeMultiAgentInvocationEvent(BaseHookEvent):
    """Event triggered before orchestrator execution starts.

    Attributes:
        source: The multi-agent orchestrator instance
        invocation_state: Configuration that user passes in
    """

    source: "MultiAgentBase"
    invocation_state: dict[str, Any] | None = None


@dataclass
class AfterMultiAgentInvocationEvent(BaseHookEvent):
    """Event triggered after orchestrator execution completes.

    Attributes:
        source: The multi-agent orchestrator instance
        invocation_state: Configuration that user passes in
    """

    source: "MultiAgentBase"
    invocation_state: dict[str, Any] | None = None

    @property
    def should_reverse_callbacks(self) -> bool:
        """True to invoke callbacks in reverse order."""
        return True
