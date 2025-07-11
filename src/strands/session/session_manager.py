"""Session manager interface for agent session management."""

from abc import ABC, abstractmethod
from typing import Any

from ..hooks.events import AgentInitializedEvent, MessageAddedEvent
from ..hooks.registry import HookProvider, HookRegistry


class SessionManager(HookProvider, ABC):
    """Abstract interface for managing sessions.

    A session represents a complete interaction context including conversation
    history, user information, agent state, and metadata. This interface provides
    methods to manage sessions and their associated data.
    """

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register initialize and append_message as hooks for the Agent."""
        registry.add_callback(AgentInitializedEvent, self.initialize)
        registry.add_callback(MessageAddedEvent, self.append_message)

    @abstractmethod
    def append_message(self, event: MessageAddedEvent) -> None:
        """Append a message to the agent's session.

        Args:
            event: Event for a newly added Message
        """

    @abstractmethod
    def initialize(self, event: AgentInitializedEvent) -> None:
        """Initialize an agent with a session.

        Args:
            event: Event when an agent is initialized
        """
