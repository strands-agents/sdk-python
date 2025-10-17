"""Agent-related type definitions for the SDK.

This module defines the types used for an Agent.
"""

from typing import TypeAlias

from .content import ContentBlock, Messages
from .interrupt import InterruptResponse

AgentInput: TypeAlias = str | list[ContentBlock] | list[InterruptResponse] | Messages | None
