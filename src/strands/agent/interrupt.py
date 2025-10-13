"""Track the state of interrupt events raised by the user for human-in-the-loop workflows."""

from dataclasses import asdict, dataclass, field
from typing import Any

from ..interrupt import Interrupt


@dataclass
class InterruptState:
    """Track the state of interrupt events raised by the user.

    Note, interrupt state is cleared after resuming.

    Attributes:
        interrupts: Interrupts raised by the user.
        context: Additional context associated with an interrupt event.
        activated: True if agent is in an interrupt state, False otherwise.
    """

    interrupts: dict[str, Interrupt] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    activated: bool = False

    def __contains__(self, interrupt_id: str) -> bool:
        """True if interrupt exists in state, False otherwise."""
        return interrupt_id in self.interrupts

    def __getitem__(self, interrupt_id: str) -> Interrupt:
        """Get interrupt associated with the given id."""
        return self.interrupts[interrupt_id]

    def __setitem__(self, interrupt_id: str, interrupt: Interrupt) -> None:
        """Set the interrupt in state under the given id."""
        self.interrupts[interrupt_id] = interrupt

    def setdefault(self, interrupt_id: str, interrupt: Interrupt) -> Interrupt:
        """Set the interrupt in state under the given id if not already present.

        Args:
            interrupt_id: Unique id of the interrupt.
            interrupt: Interrupt instance to store in state if not already present.

        Returns:
            Interrupt instance in state.
        """
        if interrupt_id in self:
            return self[interrupt_id]

        self[interrupt_id] = interrupt
        return interrupt

    def activate(self, context: dict[str, Any] | None = None) -> None:
        """Activate the interrupt state.

        Args:
            context: Context associated with the interrupt event.
        """
        self.context = context or {}
        self.activated = True

    def deactivate(self) -> None:
        """Deacitvate the interrupt state.

        Interrupts and context are cleared.
        """
        self.interrupts = {}
        self.context = {}
        self.activated = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for session management."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterruptState":
        """Initiailize interrupt state from dict."""
        return cls(
            interrupts={
                interrupt_id: Interrupt(**interrupt_data) for interrupt_id, interrupt_data in data["interrupts"].items()
            },
            context=data["context"],
            activated=data["activated"],
        )
