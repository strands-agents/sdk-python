"""Session manager interface for agent session management."""

from abc import ABC, abstractmethod
from typing import List, Optional

from .session_models import Session, SessionAgent, SessionMessage


class SessionDAO(ABC):
    """Abstract base class for session data access objects."""

    @abstractmethod
    def create_session(self, session: Session) -> Session:
        """Create a new session.

        Args:
            session: Session object to create

        Returns:
            The created session

        Raises:
            SessionException: If session creation fails
        """
        raise NotImplementedError("Subclasses must implement create_session")

    @abstractmethod
    def read_session(self, session_id: str) -> Session:
        """Read session data.

        Args:
            session_id: ID of the session to read

        Returns:
            Session object containing session data

        Raises:
            SessionException: If session doesn't exist or read fails
        """
        raise NotImplementedError("Subclasses must implement read_session")

    @abstractmethod
    def update_session(self, session: Session) -> None:
        """Update session data.

        Args:
            session: Updated session object to save

        Raises:
            SessionException: If session doesn't exist or update fails
        """
        raise NotImplementedError("Subclasses must implement update_session")

    @abstractmethod
    def list_sessions(self) -> List[Session]:
        """List all sessions.

        Returns:
            List of session objects

        Raises:
            SessionException: If list operation fails
        """
        raise NotImplementedError("Subclasses must implement list_sessions")

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Delete the session and all associated data.

        Args:
            session_id: ID of the session to delete

        Raises:
            SessionException: If session doesn't exist or delete fails
        """
        raise NotImplementedError("Subclasses must implement delete_session")

    @abstractmethod
    def create_agent(self, session_id: str, session_agent: SessionAgent) -> None:
        """Create a new agent in the session.

        Args:
            session_id: ID of the session
            session_agent: SessionAgent object to create

        Raises:
            SessionException: If agent creation fails
        """
        raise NotImplementedError("Subclasses must implement create_agent")

    @abstractmethod
    def read_agent(self, session_id: str, agent_id: str) -> SessionAgent:
        """Read agent data.

        Args:
            session_id: ID of the session
            agent_id: ID of the agent

        Returns:
            Agent data dictionary

        Raises:
            SessionException: If agent doesn't exist or read fails
        """
        raise NotImplementedError("Subclasses must implement read_agent")

    @abstractmethod
    def update_agent(self, session_id: str, SessionAgent: SessionAgent) -> None:
        """Update agent data.

        Args:
            session_id: ID of the session
            SessionAgent: Updated SessionAgent object

        Raises:
            SessionException: If agent doesn't exist or update fails
        """
        raise NotImplementedError("Subclasses must implement update_agent")

    @abstractmethod
    def delete_agent(self, session_id: str, agent_id: str) -> None:
        """Delete an agent from the session.

        Args:
            session_id: ID of the session
            agent_id: ID of the agent to delete

        Raises:
            SessionException: If agent doesn't exist or delete fails
        """
        raise NotImplementedError("Subclasses must implement delete_agent")

    @abstractmethod
    def list_agents(self, session_id: str) -> List[SessionAgent]:
        """List all agents in the session.

        Args:
            session_id: ID of the session

        Returns:
            List of agent data dictionaries

        Raises:
            SessionException: If session doesn't exist or list fails
        """
        raise NotImplementedError("Subclasses must implement list_agents")

    @abstractmethod
    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage) -> None:
        """Create a new message for the agent.

        Args:
            session_id: ID of the session
            agent_id: ID of the agent
            session_message: SessionMessage object to create

        Raises:
            SessionException: If message creation fails
        """
        raise NotImplementedError("Subclasses must implement create_message")

    @abstractmethod
    def read_message(self, session_id: str, agent_id: str, message_id: str) -> SessionMessage:
        """Read message data.

        Args:
            session_id: ID of the session
            agent_id: ID of the agent
            message_id: ID of the message

        Returns:
            Message data dictionary

        Raises:
            SessionException: If message doesn't exist or read fails
        """
        raise NotImplementedError("Subclasses must implement read_message")

    @abstractmethod
    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage) -> None:
        """Update message data.

        Args:
            session_id: ID of the session
            agent_id: ID of the agent
            session_message: Updated SessionMessage object

        Raises:
            SessionException: If message doesn't exist or update fails
        """
        raise NotImplementedError("Subclasses must implement update_message")

    @abstractmethod
    def delete_message(self, session_id: str, agent_id: str, message_id: str) -> None:
        """Delete a message from the agent.

        Args:
            session_id: ID of the session
            agent_id: ID of the agent
            message_id: ID of the message to delete

        Raises:
            SessionException: If message doesn't exist or delete fails
        """
        raise NotImplementedError("Subclasses must implement delete_message")

    @abstractmethod
    def list_messages(
        self, session_id: str, agent_id: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[SessionMessage]:
        """Read messages for an agent with pagination.

        Args:
            session_id: ID of the session
            agent_id: ID of the agent
            limit: Maximum number of messages to return (None for all)
            offset: Number of messages to skip

        Returns:
            List of message data dictionaries

        Raises:
            SessionException: If read fails
        """
        raise NotImplementedError("Subclasses must implement read_messages")
