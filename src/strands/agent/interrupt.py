"""Track the state of interrupt events raised by the user for human-in-the-loop workflows."""

from dataclasses import asdict, dataclass, field
from typing import Any

from ..hooks import Interrupt


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

    def set(self, interrupt: Interrupt) -> None:
        """Store the interrupt in state."""
        self.interrupts[interrupt.id_] = interrupt

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
