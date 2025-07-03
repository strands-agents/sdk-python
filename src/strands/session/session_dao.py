"""Session manager interface for agent session management."""

from abc import ABC, abstractmethod
from typing import List, Optional

from .session_models import Session, SessionAgent, SessionMessage


class SessionDAO(ABC):
    """Abstract base class for Session data access objects."""

    @abstractmethod
    def create_session(self, session: Session) -> Session:
        """Create a new Session."""

    @abstractmethod
    def read_session(self, session_id: str) -> Session:
        """Read a Session."""

    @abstractmethod
    def update_session(self, session: Session) -> None:
        """Update a Session."""

    @abstractmethod
    def list_sessions(self) -> List[Session]:
        """List Sessions."""

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Delete a session."""

    @abstractmethod
    def create_agent(self, session_id: str, session_agent: SessionAgent) -> None:
        """Create a new Agent in a Session."""

    @abstractmethod
    def read_agent(self, session_id: str, agent_id: str) -> SessionAgent:
        """Read an Agent."""

    @abstractmethod
    def update_agent(self, session_id: str, SessionAgent: SessionAgent) -> None:
        """Update an Agent."""

    @abstractmethod
    def delete_agent(self, session_id: str, agent_id: str) -> None:
        """Delete an agent."""
        raise NotImplementedError("Subclasses must implement delete_agent")

    @abstractmethod
    def list_agents(self, session_id: str) -> List[SessionAgent]:
        """List Agents."""

    @abstractmethod
    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage) -> None:
        """Create a new Message for the Agent."""

    @abstractmethod
    def read_message(self, session_id: str, agent_id: str, message_id: str) -> SessionMessage:
        """Read a Message."""

    @abstractmethod
    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage) -> None:
        """Update a Message."""

    @abstractmethod
    def delete_message(self, session_id: str, agent_id: str, message_id: str) -> None:
        """Delete a Message."""

    @abstractmethod
    def list_messages(
        self, session_id: str, agent_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[SessionMessage]:
        """List Messages from an Agent with pagination."""
