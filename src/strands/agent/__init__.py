"""This package provides the core Agent interface and supporting components for building AI agents with the SDK.

It includes:

- Agent: The main interface for interacting with AI models and tools
- ConversationManager: Classes for managing conversation history and context windows
- Serializers: Pluggable serialization strategies for agent state (JSONSerializer, PickleSerializer)
"""

from .agent import Agent
from .agent_result import AgentResult
from .conversation_manager import (
    ConversationManager,
    NullConversationManager,
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)
from .serializers import JSONSerializer, PickleSerializer, StateSerializer
from .state import AgentState

__all__ = [
    "Agent",
    "AgentResult",
    "AgentState",
    "ConversationManager",
    "JSONSerializer",
    "NullConversationManager",
    "PickleSerializer",
    "SlidingWindowConversationManager",
    "StateSerializer",
    "SummarizingConversationManager",
]
