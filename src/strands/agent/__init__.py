"""This package provides the core Agent interface and supporting components for building AI agents with the SDK.

It includes:

- Agent: The main interface for interacting with AI models and tools
- ConversationManager: Classes for managing conversation history and context windows
"""

from .agent import Agent
from .agent_result import AgentResult
from .base import AgentBase
from .conversation_manager import (
    ConversationManager,
    NullConversationManager,
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)

__all__ = [
    "Agent",
    "AgentBase",
    "AgentResult",
    "ConversationManager",
    "NullConversationManager",
    "SlidingWindowConversationManager",
    "SummarizingConversationManager",
]
