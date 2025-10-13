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
