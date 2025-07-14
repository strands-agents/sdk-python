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

    A session manager is in charge of persisting the conversation and state of an agent across its interaction.
    Changes made to the agents conversation, state, or other attributes should be persisted immediately after
    they are changed. The different methods introduced in this class are called at important lifecycle events
    for an agent, and should be persisted in the session.
    """

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks for persisting the agent to the session."""
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
        """Serialize and sync the agent with the session storage.

        Args:
            agent: Agent who should be synchronized with the session storage
        """

    @abstractmethod
    def initialize(self, agent: "Agent") -> None:
        """Initialize an agent with a session.

        Args:
            agent: Agent to initialize
        """
