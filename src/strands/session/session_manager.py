"""Session manager interface for agent session management."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..types.content import Message

if TYPE_CHECKING:
    from ..agent.agent import Agent


class SessionManager(ABC):
    """Abstract interface for managing sessions.

    A session represents a complete interaction context including conversation
    history, user information, agent state, and metadata. This interface provides
    methods to manage sessions and their associated data.
    """

    @abstractmethod
    def append_message(self, agent: "Agent", message: Message) -> None:
        """Append a message to the agent's session.

        Args:
            agent: The agent whose session to update
            message: The message to append

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement update_session")

    @abstractmethod
    def initialize(self, agent: "Agent") -> None:
        """Initialize an agent with a session.

        Args:
            agent: Agent instance to sync with a session
        Raises:
            SessionException: If update operation fails
        """
        raise NotImplementedError("Subclasses must implement initialize_agent")
