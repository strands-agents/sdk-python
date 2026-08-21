"""Configuration for custom reasoner integration."""

from collections.abc import Callable
from dataclasses import dataclass

from .reasoner import Reasoner


@dataclass
class ReasonerConfig:
    """Configuration for custom reasoner integration.

    Attributes:
        reasoner: The reasoning implementation (required).
        max_tool_iterations: Max tool call loops per turn before fallback.
        on_interrupted: Optional callback when barge-in cancels active reasoning.
        fallback_message: Message to return if reasoner yields nothing or errors.
    """

    reasoner: Reasoner
    max_tool_iterations: int = 5
    on_interrupted: Callable[[], None] | None = None
    fallback_message: str = "I'm not sure how to help with that."
