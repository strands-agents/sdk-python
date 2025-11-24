"""Human-in-the-loop interrupt system for agent workflows."""

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .types.agent import AgentInput
    from .types.interrupt import InterruptResponseContent


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


@dataclass
class _InterruptState:
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

    def activate(self) -> None:
        """Activate the interrupt state."""
        self.activated = True

    def deactivate(self) -> None:
        """Deacitvate the interrupt state.

        Interrupts and context are cleared.
        """
        self.interrupts = {}
        self.context = {}
        self.activated = False

    def resume(self, prompt: "AgentInput") -> None:
        """Configure the interrupt state if resuming from an interrupt event.

        Args:
            prompt: User responses if resuming from interrupt.

        Raises:
            TypeError: If in interrupt state but user did not provide responses.
        """
        if not self.activated:
            return

        if not isinstance(prompt, list):
            raise TypeError(f"prompt_type={type(prompt)} | must resume from interrupt with list of interruptResponse's")

        invalid_types = [
            content_type for content in prompt for content_type in content if content_type != "interruptResponse"
        ]
        if invalid_types:
            raise TypeError(
                f"content_types=<{invalid_types}> | must resume from interrupt with list of interruptResponse's"
            )

        contents = cast(list["InterruptResponseContent"], prompt)
        for content in contents:
            interrupt_id = content["interruptResponse"]["interruptId"]
            interrupt_response = content["interruptResponse"]["response"]

            if interrupt_id not in self.interrupts:
                raise KeyError(f"interrupt_id=<{interrupt_id}> | no interrupt found")

            self.interrupts[interrupt_id].response = interrupt_response

        self.context["responses"] = contents

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for session management."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_InterruptState":
        """Initiailize interrupt state from serialized interrupt state.

        Interrupt state can be serialized with the `to_dict` method.
        """
        return cls(
            interrupts={
                interrupt_id: Interrupt(**interrupt_data) for interrupt_id, interrupt_data in data["interrupts"].items()
            },
            context=data["context"],
            activated=data["activated"],
        )
