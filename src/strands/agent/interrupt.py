"""<TODO>."""

from typing import Any

from ..hooks import Interrupt


class InterruptState:
    """<TODO>."""

    def __init__(
        self,
        activated: bool = False,
        context: dict[str, Any] | None = None,
        interrupts: dict[str, Interrupt] | None = None,
    ) -> None:
        """<TODO>."""
        self.activated = activated
        self.context = context or {}
        self.interrupts = interrupts or {}

    def __contains__(self, interrupt_id: str) -> bool:
        """<TODO>."""
        return interrupt_id in self.interrupts

    def __getitem__(self, interrupt_id: str) -> Interrupt:
        """<TODO>."""
        return self.interrupts[interrupt_id]

    def __setitem__(self, interrupt_id: str, interrupt: Interrupt) -> None:
        """<TODO>."""
        self.interrupts[interrupt_id] = interrupt

    def set(self, context: dict[str, Any] | None = None) -> None:
        """<TODO>."""
        self.activated = True
        self.context = context or {}

    def clear(self) -> None:
        """<TODO>."""
        self.activated = False
        self.context = {}
        self.interrupts = {}

    def to_dict(self) -> dict[str, Any]:
        """<TODO>."""
        return {
            "activated": self.activated,
            "context": self.context,
            "interrupts": {interrupt_id: interrupt.to_dict() for interrupt_id, interrupt in self.interrupts.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterruptState":
        """<TODO>."""
        return cls(
            activated=data["activated"],
            context=data["context"],
            interrupts={
                interrupt_id: Interrupt(**interrupt_data) for interrupt_id, interrupt_data in data["interrupts"].items()
            },
        )
