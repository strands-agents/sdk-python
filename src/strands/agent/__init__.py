"""This package provides the core Agent interface and supporting components for building AI agents with the SDK."""

from .agent import Agent
from .agent_result import AgentResult
from .conversation_manager import (
    ConversationManager,
    NullConversationManager,
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)

__all__ = [
    "Agent",
    "AgentResult",
    "ConversationManager",
    "NullConversationManager",
    "SlidingWindowConversationManager",
    "SummarizingConversationManager",
]
