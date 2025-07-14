"""Repository session manager implementation."""

import logging

from ..agent.agent import Agent
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


class RepositorySessionManager(SessionManager):
    """Session manager for persisting agents in a SessionRepository."""

    def __init__(
        self,
        session_id: str,
        session_repository: SessionRepository,
    ):
        """Initialize the RepositorySessionManager.

        If no session with the specified session_id exists yet, it will be created
        in the session_repository.

        Args:
          session_id: ID to use for the session. A new session with this id will be created if it does
              not exist in the reposiory yet
          session_repository: Underlying session repository to use to store the sessions state.
        """
        self.session_repository = session_repository
        self.session_id = session_id
        session = session_repository.read_session(session_id)
        # Create a session if it does not exist yet
        if session is None:
            logger.debug("session_id=<%s> | session not found, creating new session", self.session_id)
            session = Session(session_id=session_id, session_type=SessionType.AGENT)
            session_repository.create_session(session)

        self.session = session

        # Keep track of the initialized agent id's so that two agents in a session cannot share an id
        self._initialized_agent_ids: set[str] = set()

    def append_message(self, message: Message, agent: Agent) -> None:
        """Append a message to the agent's session.

        Args:
            message: Message to add to the agent in the session
            agent: Agent to append the message to
        """
        session_message = SessionMessage.from_message(message)
        self.session_repository.create_message(self.session_id, agent.agent_id, session_message)

    def sync_agent(self, agent: Agent) -> None:
        """Serialize and update the agent into the session repository.

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
        if agent.agent_id in self._initialized_agent_ids:
            raise SessionException("The `agent_id` of an agent must be unique in a session.")
        self._initialized_agent_ids.add(agent.agent_id)

        session_agent = self.session_repository.read_agent(self.session_id, agent.agent_id)

        if session_agent is None:
            logger.debug(
                "agent_id=<%s> | session_id=<%s> | creating agent",
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
                "agent_id=<%s> | session_id=<%s> | restoring agent",
                agent.agent_id,
                self.session_id,
            )
            agent.messages = [
                session_message.to_message()
                for session_message in self.session_repository.list_messages(self.session_id, agent.agent_id)
            ]
            agent.state = AgentState(session_agent.state)
