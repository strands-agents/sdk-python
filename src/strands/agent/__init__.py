"""This package provides the core Agent interface and supporting components for building AI agents with the SDK."""

from typing import TYPE_CHECKING, Any

from .agent import Agent
from .agent_result import AgentResult
from .base import AgentBase
from .conversation_manager import (
    ConversationManager,
    NullConversationManager,
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)

if TYPE_CHECKING:
    from .a2a_agent import A2AAgent

__all__ = [
    "Agent",
    "AgentBase",
    "AgentResult",
    "ConversationManager",
    "NullConversationManager",
    "SlidingWindowConversationManager",
    "SummarizingConversationManager",
]


def __getattr__(name: str) -> Any:
    """Lazy load A2AAgent to defer import of optional a2a dependency."""
    if name == "A2AAgent":
        from .a2a_agent import A2AAgent

        return A2AAgent
    raise AttributeError(f"cannot import name '{name}' from '{__name__}' ({__file__})")
