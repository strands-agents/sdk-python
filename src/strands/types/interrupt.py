"""Interrupt related type definitions for human-in-the-loop workflows."""

from typing import Any, TypedDict


class InterruptResponse(TypedDict):
    """User response to an interrupt.

    Attributes:
        name: Unique identifier for the interrupt.
        event_name: Name of the hook event under which the interrupt was triggered.
        response: User response to the interrupt.
    """

    name: str
    event_name: str
    response: Any


class InterruptContent(TypedDict):
    """Content block containing an interrupt response for human-in-the-loop workflows.

    Attributes:
        interruptResponse: User response to an interrupt event.
    """

    interruptResponse: InterruptResponse
