"""Interrupt related type definitions for human-in-the-loop workflows."""

from typing import TYPE_CHECKING, Any, Protocol, TypedDict

from ..interrupt import Interrupt, InterruptException

if TYPE_CHECKING:
    from ..agent import Agent


class InterruptHookEvent(Protocol):
    """Interface that adds interrupt support to hook events."""

    agent: "Agent"

    def interrupt(self, name: str, reason: Any = None, response: Any = None) -> Any:
        """Trigger the interrupt with a reason.

        Args: name: User defined name for the interrupt.
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


class InterruptResponse(TypedDict):
    """User response to an interrupt.

    Attributes:
        interruptId: Unique identifier for the interrupt.
        response: User response to the interrupt.
    """

    interruptId: str
    response: Any


class InterruptResponseContent(TypedDict):
    """Content block containing a user response to an interrupt.

    Attributes:
        interruptResponse: User response to an interrupt event.
    """

    interruptResponse: InterruptResponse
