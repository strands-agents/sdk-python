"""Interrupt related type definitions for human-in-the-loop workflows."""

from typing import Any, TypedDict


class InterruptResponse(TypedDict):
    """User response to an interrupt.

    Attributes:
        name: Unique identifier for the interrupt.
        response: User response to the interrupt.
    """

    name: str
    response: Any


class InterruptContent(TypedDict):
    """Content block containing an interrupt response for human-in-the-loop workflows.

    Attributes:
        interruptResponse: User response to an interrupt event.
    """

    interruptResponse: InterruptResponse
