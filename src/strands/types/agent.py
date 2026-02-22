"""Agent-related type definitions for the SDK.

This module defines the types used for an Agent.
"""

from enum import Enum
from typing import TypeAlias

from .content import ContentBlock, Messages
from .interrupt import InterruptResponseContent

AgentInput: TypeAlias = str | list[ContentBlock] | list[InterruptResponseContent] | Messages | None


class ConcurrentInvocationMode(str, Enum):
    """Mode controlling concurrent invocation behavior.

    Values:
        THROW: Raises ConcurrencyException if concurrent invocation is attempted (default).
        UNSAFE_REENTRANT: Allows concurrent invocations without locking.

    Warning:
        The ``UNSAFE_REENTRANT`` mode makes no guarantees about resulting behavior and is
        provided only for advanced use cases where the caller understands the risks.
    """

    THROW = "throw"
    UNSAFE_REENTRANT = "unsafe_reentrant"
