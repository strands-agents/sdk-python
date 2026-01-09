"""Agent-related type definitions for the SDK.

This module defines the types used for an Agent.
"""

from typing import TypeAlias

from .content import ContentBlock, Messages
from .interrupt import InterruptResponseContent

AgentInput: TypeAlias = str | list[ContentBlock] | list[InterruptResponseContent] | Messages | None
