"""Human-in-the-loop interrupt system for agent workflows."""

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .types.agent import AgentInput
    from .types.interrupt import InterruptResponseContent

_AGENT_STREAM_INTERRUPT_ID_PREFIX = "v1:middleware_agent_stream:"
"""Id prefix for interrupts scoped to a whole invocation pass rather than a single tool call."""


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
        interrupts: Interrupts raised by the user. An answered invocation-scoped interrupt can
            outlive ``activated`` being cleared: it is held until the interrupt cycle it belongs
            to ends (see ``end_tool_cycle`` and ``end_interrupt_cycle``), so ``activated`` alone
            does not imply this mapping is empty.
        context: Additional context associated with an interrupt event.
        activated: True if agent is in an interrupt state, False otherwise.
    """

    interrupts: dict[str, Interrupt] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    activated: bool = False
    _version: int = field(default=0, compare=False, repr=False)

    def activate(self) -> None:
        """Activate the interrupt state."""
        self.activated = True
        self._version += 1

    def deactivate(self) -> None:
        """Deacitvate the interrupt state.

        Interrupts and context are cleared.
        """
        self.interrupts = {}
        self.context = {}
        self.activated = False
        self._version += 1

    def end_tool_cycle(self) -> None:
        """Clear the state a completed tool cycle owns, keeping answered invocation-scoped ones.

        Called when a tool cycle finishes and nothing is pending for it, so its interrupts and
        context are done with. An answered invocation-scoped interrupt is kept: it belongs to the
        interrupt cycle rather than to this tool cycle, and the pass that reads its response can
        be a later one than the pass the human answered. An unanswered one is not kept — it is
        either about to be raised (and registered) again or no longer wanted.

        Use ``deactivate`` to reset the state completely, or ``end_interrupt_cycle`` to release
        the responses this method retains.
        """
        self.interrupts = {
            interrupt_id: interrupt
            for interrupt_id, interrupt in self.interrupts.items()
            if interrupt_id.startswith(_AGENT_STREAM_INTERRUPT_ID_PREFIX) and interrupt.response is not None
        }
        self.context = {}
        self.activated = False
        self._version += 1

    def end_interrupt_cycle(self) -> None:
        """Release invocation-scoped interrupts once their interrupt cycle is over.

        An answered invocation-scoped response is held for the whole interrupt cycle — every pass
        from the interrupt that asked the human through to the pass that completes with nothing
        owed a resume (see ``end_tool_cycle``) — so the human is asked once. Releasing it here
        stops it becoming a standing approval that a later cycle would silently resolve against.

        Runs on every pass that ends a cycle, so it bumps the version only when it actually
        released something and a no-op does not mark the state dirty for session writes.
        """
        remaining = {
            interrupt_id: interrupt
            for interrupt_id, interrupt in self.interrupts.items()
            if not interrupt_id.startswith(_AGENT_STREAM_INTERRUPT_ID_PREFIX)
        }
        if remaining == self.interrupts:
            return

        self.interrupts = remaining
        self._version += 1

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
        self._version += 1

    def _get_version(self) -> int:
        """Get the current version number of the interrupt state.

        The version is incremented each time the state is mutated — activate(), deactivate(),
        resume(), end_tool_cycle(), or end_interrupt_cycle().
        Consumers can compare versions to detect changes without requiring
        explicit dirty flag clearing.

        Returns:
            The current version number.
        """
        return self._version

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for session management."""
        return {
            "interrupts": {k: v.to_dict() for k, v in self.interrupts.items()},
            "context": self.context,
            "activated": self.activated,
        }

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
