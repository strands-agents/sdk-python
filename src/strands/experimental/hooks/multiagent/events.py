"""Multi-agent execution lifecycle events for hook system integration.

These events are fired by orchestrators (Graph/Swarm) at key points so
hooks can persist, monitor, or debug execution. No intermediate state model
is used—hooks read from the orchestrator directly.
"""

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from typing_extensions import override

from ....hooks import BaseHookEvent
from ....types.interrupt import _Interruptible

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
class BeforeNodeCallEvent(BaseHookEvent, _Interruptible):
    """Event triggered before individual node execution starts.

    Attributes:
        source: The multi-agent orchestrator instance
        node_id: ID of the node about to execute
        invocation_state: Configuration that user passes in
        cancel_node: A user defined message that when set, will cancel the node execution with status FAILED.
            The message will be emitted under a MultiAgentNodeCancel event. If set to `True`, Strands will cancel the
            node using a default cancel message.
    """

    source: "MultiAgentBase"
    node_id: str
    invocation_state: dict[str, Any] | None = None
    cancel_node: bool | str = False

    def _can_write(self, name: str) -> bool:
        return name in ["cancel_node"]

    @override
    def _interrupt_id(self, name: str) -> str:
        """Unique id for the interrupt.

        Args:
            name: User defined name for the interrupt.

        Returns:
            Interrupt id.
        """
        node_id = uuid.uuid5(uuid.NAMESPACE_OID, self.node_id)
        call_id = uuid.uuid5(uuid.NAMESPACE_OID, name)
        return f"v1:before_node_call:{node_id}:{call_id}"


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
