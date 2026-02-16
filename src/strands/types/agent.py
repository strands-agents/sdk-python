"""Agent-related type definitions for the SDK.

This module defines the types used for an Agent.
"""

from typing import Literal, TypeAlias

from .content import ContentBlock, Messages
from .interrupt import InterruptResponseContent

AgentInput: TypeAlias = str | list[ContentBlock] | list[InterruptResponseContent] | Messages | None

ConcurrentInvocationMode = Literal["throw", "unsafe_reentrant"]
"""Mode controlling concurrent invocation behavior.

Values:
    throw: Raises ConcurrencyException if concurrent invocation is attempted (default).
    unsafe_reentrant: Allows concurrent invocations without locking (unsafe, restores pre-lock behavior).
"""
