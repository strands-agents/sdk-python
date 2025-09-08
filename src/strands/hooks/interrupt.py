"""<TODO>."""
import abc
from dataclasses import dataclass
from typing import Any


@dataclass
class Interrupt:
    """<TODO>."""

    name: str
    reasons: list[Any]
    resume: Any = None
    activated: bool = False

    def __call__(self, reason: Any) -> Any:
        """<TODO>."""
        if self.resume:
            self.activated = False
            return self.resume

        self.reasons.append(reason)
        self.activated = True
        raise InterruptException(self)


class InterruptException(Exception):
    """<TODO>."""
    def __init__(self, interrupt: Interrupt) -> None:
        self.interrupt = interrupt


@dataclass
class InterruptEvent:
    """<TODO>."""

    interrupt: Interrupt
