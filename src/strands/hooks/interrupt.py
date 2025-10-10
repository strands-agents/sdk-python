"""Human-in-the-loop interrupt system for agent workflows."""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ..types.content import ContentBlock

if TYPE_CHECKING:
    from ..agent import Agent


@dataclass
class Interrupt:
    """Represents an interrupt that can pause agent execution for human-in-the-loop workflows.

    Attributes:
        id_: Unique identifier.
        name: User defined name.
        reason: User provided reason for raising the interrupt.
        response: Human response provided when resuming the agent after an interrupt.
    """

    id_: str
    name: str
    reason: Any = None
    response: Any = None

    def to_dict(self) -> dict[str, Any]:
        """<TODO>."""
        return asdict(self)

    def to_reason_content(self) -> ContentBlock:
        """<TODO>."""
        return {
            "interruptReason": {"interruptId": self.id_, "interruptName": self.name, "reason": self.reason},
        }


class InterruptException(Exception):
    """Exception raised when human input is required."""

    def __init__(self, id_: str, name: str, reason: Any) -> None:
        """Initialize the exception with an interrupt instance.

        Args:
            id_: Unique identifier.
            name: User defined name.
            reason: User provided reason for raising the interrupt.
        """
        self.interrupt = Interrupt(id_, name, reason)


class InterruptHookEvent(Protocol):
    """Interface that adds interrupt support to hook events."""

    agent: "Agent"

    def interrupt(self, name: str, reason: Any = None) -> Any:
        """Trigger the interrupt with a reason.

        Args:
            name: User defined name for the interrupt.
            reason: User provided reason for the interrupt.

        Returns:
            The response from a human user when resuming from an interrupt state.

        Raises:
            InterruptException: If human input is required.
        """
        id_ = self.interrupt_id(name, reason)
        if id_ in self.agent.interrupt_state:
            return self.agent.interrupt_state[id_].response

        raise InterruptException(id_, name, reason)

    def interrupt_id(self, name: str, reason: Any) -> str:
        """Unique id for the interrupt.

        Args:
            name: User defined name for the interrupt.
            reason: User provided reason for the interrupt.

        Returns:
            Interrupt id.
        """
        ...
