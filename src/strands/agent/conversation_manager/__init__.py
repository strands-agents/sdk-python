"""This package provides classes for managing conversation history during agent execution.

It includes:

- ConversationManager: Abstract base class defining the conversation management interface
- NullConversationManager: A no-op implementation that does not modify conversation history
- SlidingWindowConversationManager: An implementation that maintains a sliding window of messages to control context
  size while preserving conversation coherence
- SummarizingConversationManager: An implementation that summarizes older context instead
  of simply trimming it
- PruningConversationManager: An implementation that selectively prunes messages using configurable strategies

Conversation managers help control memory usage and context length while maintaining relevant conversation state, which
is critical for effective agent interactions.
"""

from .conversation_manager import ConversationManager
from .null_conversation_manager import NullConversationManager
from .pruning_conversation_manager import MessageContext, PruningContext, PruningConversationManager, PruningStrategy
from .sliding_window_conversation_manager import SlidingWindowConversationManager

# Import strategies
from .strategies import (
    LargeToolResultPruningStrategy,
)
from .summarizing_conversation_manager import SummarizingConversationManager

__all__ = [
    "ConversationManager",
    "NullConversationManager",
    "SlidingWindowConversationManager",
    "SummarizingConversationManager",
    "PruningConversationManager",
    "PruningStrategy",
    "PruningContext",
    "MessageContext",
    "LargeToolResultPruningStrategy",
]
