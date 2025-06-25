"""Agent session manager implementation."""

import logging

from ..agent.agent import _DEFAULT_AGENT_ID, Agent
from ..agent.state import AgentState
from ..types.content import Message
from ..types.exceptions import SessionException
from ..types.session import (
    Session,
    SessionAgent,
    SessionMessage,
    SessionType,
)
from .session_manager import SessionManager
from .session_repository import SessionRepository

logger = logging.getLogger(__name__)

DEFAULT_SESSION_AGENT_ID = "default"


class AgentSessionManager(SessionManager):
    """Session manager for persisting agent's in a Session."""

    def __init__(
        self,
        session_id: str,
        session_repository: SessionRepository,
    ):
        """Initialize the AgentSessionManager."""
        self.session_repository = session_repository
        self.session_id = session_id
        session = session_repository.read_session(session_id)
        # Create a session if it does not exist yet
        if session is None:
            logger.debug("session_id=<%s> | Session not found, creating new session.", self.session_id)
            session = Session(session_id=session_id, session_type=SessionType.AGENT)
            session_repository.create_session(session)

        self.session = session
        self._default_agent_initialized = False

    def append_message(self, message: Message, agent: Agent) -> None:
        """Append a message to the agent's session.

        Args:
            message: Message to add to the agent in the session
            agent: Agent to append the message to
        """
        session_message = SessionMessage.from_message(message)
        if agent.agent_id is None:
            raise ValueError("`agent.agent_id` must be set before appending message to session.")
        self.session_repository.create_message(self.session_id, agent.agent_id, session_message)

    def sync_agent(self, agent: Agent) -> None:
        """Sync agent to the session.

        Args:
            agent: Agent to sync to the session.
        """
        self.session_repository.update_agent(
            self.session_id,
            SessionAgent.from_agent(agent),
        )

    def initialize(self, agent: Agent) -> None:
        """Initialize an agent with a session.

        Args:
            agent: Agent to initialize from the session
        """
        if agent.agent_id is _DEFAULT_AGENT_ID:
            if self._default_agent_initialized:
                raise SessionException("Set `agent_id` to support more than one agent in a session.")
            self._default_agent_initialized = True

        session_agent = self.session_repository.read_agent(self.session_id, agent.agent_id)

        if session_agent is None:
            logger.debug(
                "agent_id=<%s> | session_id=<%s> | Creating agent.",
                agent.agent_id,
                self.session_id,
            )

            session_agent = SessionAgent.from_agent(agent)
            self.session_repository.create_agent(self.session_id, session_agent)
            for message in agent.messages:
                session_message = SessionMessage.from_message(message)
                self.session_repository.create_message(self.session_id, agent.agent_id, session_message)
        else:
            logger.debug(
                "agent_id=<%s> | session_id=<%s> | Restoring agent.",
                agent.agent_id,
                self.session_id,
            )
            agent.messages = [
                session_message.to_message()
                for session_message in self.session_repository.list_messages(self.session_id, agent.agent_id)
            ]
            agent.state = AgentState(session_agent.state)
