"""File-based implementation of session manager."""

import logging
from typing import TYPE_CHECKING, Any, Optional

from ..agent.state import AgentState
from ..handlers.callback_handler import CompositeCallbackHandler
from ..types.content import Message
from .exceptions import SessionException
from .file_session_dao import FileSessionDAO
from .session_dao import SessionDAO
from .session_manager import SessionManager
from .session_models import Session, SessionAgent, SessionMessage, SessionType

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..agent.agent import Agent


class AgentSessionManager(SessionManager):
    """Session manager for a single Agent.

    This implementation stores sessions as JSON files in a specified directory.
    Each session is stored in a separate file named by its session_id.
    """

    def __init__(
        self,
        session_id: str,
        session_dao: Optional[SessionDAO] = None,
    ):
        """Initialize the FileSessionManager."""
        self.session_dao = session_dao or FileSessionDAO()
        self.session_id = session_id

    def append_message_to_agent_session(self, agent: "Agent", message: Message) -> None:
        """Append a message to the agent's session.

        Args:
            agent: The agent whose session to update
            message: The message to append
        """
        if agent.id is None:
            raise ValueError("`agent.id` must be set before appending message to session.")

        session_message = SessionMessage.from_dict(dict(message))
        self.session_dao.create_message(self.session_id, agent.id, session_message)

    def initialize_agent(self, agent: "Agent") -> None:
        """Restore agent data from the current session.

        Args:
            agent: Agent instance to restore session data to

        Raises:
            SessionException: If restore operation fails
        """
        if agent.id is None:
            raise ValueError("`agent.id` must be set before initializing session.")

        try:
            # Try to read existing session
            session = self.session_dao.read_session(self.session_id)

            if session.session_type != SessionType.AGENT:
                raise ValueError(f"Invalid session type: {session.session_type}")

            if agent.id not in [agent.agent_id for agent in self.session_dao.list_agents(self.session_id)]:
                raise ValueError(f"Agent {agent.id} not found in session {self.session_id}")

            # Initialize agent
            agent.messages = [
                session_message.to_message()
                for session_message in self.session_dao.list_messages(self.session_id, agent.id)
            ]
            agent.state = AgentState(self.session_dao.read_agent(self.session_id, agent.id).state)

        except SessionException:
            # Session doesn't exist, create new one
            logger.debug("Session not found, creating new session")
            # Session doesn't exist, create new one
            session = Session(session_id=self.session_id, session_type=SessionType.AGENT)
            session_agent = SessionAgent(agent_id=agent.id, session_id=self.session_id, state=agent.state.get())
            self.session_dao.create_session(session)
            self.session_dao.create_agent(self.session_id, session_agent)
            for message in agent.messages:
                session_message = SessionMessage.from_dict(dict(message))
                self.session_dao.create_message(self.session_id, agent.id, session_message)

        self.session = session

        # Attach a callback handler for persisting messages
        def session_callback(**kwargs: Any) -> None:
            try:
                # Handle message persistence
                if "message" in kwargs:
                    message = kwargs["message"]
                    self.append_message_to_agent_session(kwargs["agent"], message)
            except Exception as e:
                logger.error("Persistence operation failed", e)

        agent.callback_handler = CompositeCallbackHandler(agent.callback_handler, session_callback)
