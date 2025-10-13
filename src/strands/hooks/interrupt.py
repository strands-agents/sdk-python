"""Human-in-the-loop interrupt system for agent workflows."""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..agent import Agent


@dataclass
class Interrupt:
    """Represents an interrupt that can pause agent execution for human-in-the-loop workflows.

    Attributes:
        id: Unique identifier.
        name: User defined name.
        reason: User provided reason for raising the interrupt.
        response: Human response provided when resuming the agent after an interrupt.
    """

    id: str
    name: str
    reason: Any = None
    response: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for session management."""
        return asdict(self)


class InterruptException(Exception):
    """Exception raised when human input is required."""

    def __init__(self, interrupt: Interrupt) -> None:
        """Set the interrupt."""
        self.interrupt = interrupt


class InterruptHookEvent(Protocol):
    """Interface that adds interrupt support to hook events."""

    agent: "Agent"

    def interrupt(self, name: str, reason: Any = None, response: Any = None) -> Any:
        """Trigger the interrupt with a reason.

        Args:
            name: User defined name for the interrupt.
                Must be unique across hook callbacks.
            reason: User provided reason for the interrupt.
            response: Preemptive response from user if available.

        Returns:
            The response from a human user when resuming from an interrupt state.

        Raises:
            InterruptException: If human input is required.
            ValueError: If interrupt name is used more than once.
        """
        id = self._interrupt_id(name)
        state = self.agent.interrupt_state

        interrupt_ = state.setdefault(id, Interrupt(id, name, reason, response))
        if interrupt_.response:
            return interrupt_.response

        raise InterruptException(interrupt_)

    def _interrupt_id(self, name: str) -> str:
        """Unique id for the interrupt.

        Args:
            name: User defined name for the interrupt.
            reason: User provided reason for the interrupt.

        Returns:
            Interrupt id.
        """
        ...
