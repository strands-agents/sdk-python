"""Human-in-the-loop interrupt system for agent workflows."""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from ..types.tools import ToolResultContent

if TYPE_CHECKING:
    from ..agent import Agent


@dataclass
class Interrupt:
    """Represents an interrupt that can pause agent execution for human-in-the-loop workflows.

    Attributes:
        name: Unique identifier for the interrupt.
        event_name: Name of the hook event under which the interrupt was triggered.
        reasons: User provided reasons for raising the interrupt.
        response: Human response provided when resuming the agent after an interrupt.
        activated: Whether the interrupt is currently active.
    """

    name: str
    event_name: str
    reasons: list[Any]
    response: Any = None
    activated: bool = False

    def __call__(self, reason: Any) -> Any:
        """Trigger the interrupt with a reason.

        Args:
            reason: User provided reason for the interrupt.

        Returns:
            The response from a human user when resuming from an interrupt state.

        Raises:
            InterruptException: If human input is required.
        """
        if self.response:
            self.activated = False
            return self.response

        self.reasons.append(reason)
        self.activated = True
        raise InterruptException(self)

    def to_tool_result_content(self) -> list[ToolResultContent]:
        """Convert the interrupt to tool result content if there are reasons.

        Returns:
            Tool result content.
        """
        if self.reasons:
            return [
                {"json": {"interrupt": {"name": self.name, "event_name": self.event_name, "reasons": self.reasons}}},
            ]

        return []

    @classmethod
    def from_agent(cls, name: str, event_name: str, agent: "Agent") -> "Interrupt":
        """Initialize an interrupt from agent state.

        Creates an interrupt instance from stored agent state, which will be
        populated with the human response when resuming.

        Args:
            name: Unique identifier for the interrupt.
            event_name: Name of the hook event under which the interrupt was triggered.
            agent: The agent instance containing interrupt state.

        Returns:
            An Interrupt instance initialized from agent state.
        """
        interrupt = agent._interrupts.get((name, event_name))
        params = asdict(interrupt) if interrupt else {"name": name, "event_name": event_name, "reasons": []}

        return cls(**params)


class InterruptException(Exception):
    """Exception raised when human input is required."""

    def __init__(self, interrupt: Interrupt) -> None:
        """Initialize the exception with an interrupt instance.

        Args:
            interrupt: The interrupt that triggered this exception.
        """
        self.interrupt = interrupt


@dataclass
class InterruptEvent:
    """Interface that adds interrupt support to hook events.

    Attributes:
        interrupt: The interrupt instance associated with this event.
    """

    interrupt: Interrupt
