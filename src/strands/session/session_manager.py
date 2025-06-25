"""Session manager interface for agent session management."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..hooks.events import AgentInitializedEvent, MessageAddedEvent
from ..hooks.registry import HookProvider, HookRegistry
from ..types.content import Message

if TYPE_CHECKING:
    from ..agent.agent import Agent


class SessionManager(HookProvider, ABC):
    """Abstract interface for managing sessions.

    A session represents a complete interaction context including conversation
    history, user information, agent state, and metadata. This interface provides
    methods to manage sessions and their associated data.
    """

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register initialize and append_message as hooks for the Agent."""
        registry.add_callback(AgentInitializedEvent, lambda event: self.initialize(event.agent))
        registry.add_callback(MessageAddedEvent, lambda event: self.append_message(event.message, event.agent))
        registry.add_callback(MessageAddedEvent, lambda event: self.sync_agent(event.agent))

    @abstractmethod
    def append_message(self, message: Message, agent: "Agent") -> None:
        """Append a message to the agent's session.

        Args:
            message: Message to add to the agent in the session
            agent: Agent to append the message to
        """

    @abstractmethod
    def sync_agent(self, agent: "Agent") -> None:
        """Sync the agent to the session.

        Args:
            agent: Agent to sync to the session
        """

    @abstractmethod
    def initialize(self, agent: "Agent") -> None:
        """Initialize an agent with a session.

        Args:
            agent: Agent to initialize
        """
