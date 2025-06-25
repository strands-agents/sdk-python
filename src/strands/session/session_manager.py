"""Session manager interface for agent session management."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..types.content import Message

if TYPE_CHECKING:
    from ..agent.agent import Agent


class SessionManager(ABC):
    """Abstract interface for managing agent sessions.

    A session represents a complete interaction context including conversation
    history, user information, agent state, and metadata. This interface provides
    methods to manage sessions and their associated data.
    """

    @abstractmethod
    def append_message_to_agent_session(self, agent: "Agent", message: Message) -> None:
        """Append a message to the agent's session.

        Args:
            agent: The agent whose session to update
            message: The message to append

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement update_session")

    @abstractmethod
    def initialize_agent(self, agent: "Agent") -> None:
        """Update session data from an agent's current state.

        Saves the agent's current conversation history and state back to
        the session storage.

        Args:
            agent: Agent instance to save session data from

        Raises:
            SessionException: If update operation fails
        """
        raise NotImplementedError("Subclasses must implement initialize_agent")
