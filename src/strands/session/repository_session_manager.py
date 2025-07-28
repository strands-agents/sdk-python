"""Repository session manager implementation."""

import logging
from typing import TYPE_CHECKING, Any, Optional

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

if TYPE_CHECKING:
    from ..agent.agent import Agent

logger = logging.getLogger(__name__)


class RepositorySessionManager(SessionManager):
    """Session manager for persisting agents in a SessionRepository."""

    def __init__(self, session_id: str, session_repository: SessionRepository, **kwargs: Any):
        """Initialize the RepositorySessionManager.

        If no session with the specified session_id exists yet, it will be created
        in the session_repository.

        Args:
            session_id: ID to use for the session. A new session with this id will be created if it does
                not exist in the repository yet
            session_repository: Underlying session repository to use to store the sessions state.
            **kwargs: Additional keyword arguments for future extensibility.

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

        # Keep track of the latest message of each agent in case we need to redact it.
        self._latest_agent_message: dict[str, Optional[SessionMessage]] = {}

    def append_message(self, message: Message, agent: "Agent", **kwargs: Any) -> None:
        """Append a message to the agent's session.

        Args:
            message: Message to add to the agent in the session
            agent: Agent to append the message to
            **kwargs: Additional keyword arguments for future extensibility.
        """
        # Calculate the next index (0 if this is the first message, otherwise increment the previous index)
        latest_agent_message = self._latest_agent_message[agent.agent_id]
        if latest_agent_message:
            next_index = latest_agent_message.message_id + 1
        else:
            next_index = 0

        session_message = SessionMessage.from_message(message, next_index)
        self._latest_agent_message[agent.agent_id] = session_message
        self.session_repository.create_message(self.session_id, agent.agent_id, session_message)

    def redact_latest_message(self, redact_message: Message, agent: "Agent", **kwargs: Any) -> None:
        """Redact the latest message appended to the session.

        Args:
            redact_message: New message to use that contains the redact content
            agent: Agent to apply the message redaction to
            **kwargs: Additional keyword arguments for future extensibility.
        """
        latest_agent_message = self._latest_agent_message[agent.agent_id]
        if latest_agent_message is None:
            raise SessionException("No message to redact.")
        latest_agent_message.redact_message = redact_message
        return self.session_repository.update_message(self.session_id, agent.agent_id, latest_agent_message)

    def sync_agent(self, agent: "Agent", **kwargs: Any) -> None:
        """Serialize and update the agent into the session repository.

        Args:
            agent: Agent to sync to the session.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        self.session_repository.update_agent(
            self.session_id,
            SessionAgent.from_agent(agent),
        )

    def initialize(self, agent: "Agent", **kwargs: Any) -> None:
        """Initialize an agent with a session.

        Args:
            agent: Agent to initialize from the session
            **kwargs: Additional keyword arguments for future extensibility.
        """
        if agent.agent_id in self._latest_agent_message:
            raise SessionException("The `agent_id` of an agent must be unique in a session.")
        self._latest_agent_message[agent.agent_id] = None

        session_agent = self.session_repository.read_agent(self.session_id, agent.agent_id)

        if session_agent is None:
            logger.debug(
                "agent_id=<%s> | session_id=<%s> | creating agent",
                agent.agent_id,
                self.session_id,
            )

            session_agent = SessionAgent.from_agent(agent)
            self.session_repository.create_agent(self.session_id, session_agent)
            # Initialize messages with sequential indices
            session_message = None
            for i, message in enumerate(agent.messages):
                session_message = SessionMessage.from_message(message, i)
                self.session_repository.create_message(self.session_id, agent.agent_id, session_message)
            self._latest_agent_message[agent.agent_id] = session_message
        else:
            logger.debug(
                "agent_id=<%s> | session_id=<%s> | restoring agent",
                agent.agent_id,
                self.session_id,
            )
            agent.state = AgentState(session_agent.state)

            # Restore the conversation manager to its previous state, and get the optional prepend messages
            prepend_messages = agent.conversation_manager.restore_from_session(session_agent.conversation_manager_state)

            if prepend_messages is None:
                prepend_messages = []

            # List the messages currently in the session, using an offset of the messages previously removed
            # by the conversation manager.
            session_messages = self.session_repository.list_messages(
                session_id=self.session_id,
                agent_id=agent.agent_id,
                offset=agent.conversation_manager.removed_message_count,
            )
            if len(session_messages) > 0:
                self._latest_agent_message[agent.agent_id] = session_messages[-1]

            # Restore the agents messages array including the optional prepend messages
            agent.messages = prepend_messages + [session_message.to_message() for session_message in session_messages]
