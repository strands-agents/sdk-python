"""Configuration types for the ContextManager."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from typing_extensions import TypedDict

from ..types.content import Messages

if TYPE_CHECKING:
    from ..agent.agent import Agent


@dataclass
class ContextState:
    """State passed to strategies during apply().

    Attributes:
        messages: The agent's current message list (mutable in place).
        agent: The agent instance.
        utilization: Current context utilization ratio (0-1+). Above 1.0 means overflow.
    """

    messages: Messages
    agent: Agent
    utilization: float


@runtime_checkable
class ContextStrategy(Protocol):
    """A context reduction strategy.

    Strategies are applied in order during the pipeline. Each decides whether
    to act based on the current context state.
    """

    @property
    def name(self) -> str:
        """Stable identifier for logging and observability."""
        ...

    async def apply(self, context: ContextState) -> bool:
        """Attempt to reduce context. Returns True if it made changes."""
        ...


class ContextManagerConfig(TypedDict, total=False):
    """Full configuration for a ContextManager instance.

    Attributes:
        strategies: Strategies for context reduction, applied as an ordered pipeline.
    """

    strategies: list[ContextStrategy]
