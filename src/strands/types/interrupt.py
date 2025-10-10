"""Interrupt related type definitions for human-in-the-loop workflows."""

from typing import Any, TypedDict


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


class InterruptReason(TypedDict):
    """Reason for an interrupt.

    Attributes:
        interruptId: Unique identifier for the interrupt.
        interruptName: User defined name for the interrupt.
        reason: User provided reason for the interrupt.
    """

    interruptId: str
    interruptName: str
    reason: Any


class InterruptReasonContent(TypedDict):
    """Content block containing a reason for raising an interrupt.

    Attributes:
        interruptReason: User reason for raising an interrupt.
    """

    interruptReason: InterruptReason
