"""Agent session manager implementation."""

import logging
from typing import TYPE_CHECKING

from ..agent.state import AgentState
from ..experimental.hooks.events import AgentInitializedEvent, MessageAddedEvent
from ..telemetry.metrics import EventLoopMetrics
from ..types.session import (
    SessionType,
    create_session,
    session_agent_from_agent,
    session_message_from_message,
    session_message_to_message,
)
from .session_manager import SessionManager
from .session_repository import SessionRepository

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass

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
            session = create_session(session_id=session_id, session_type=SessionType.AGENT)
            session_repository.create_session(session)
        else:
            if session["session_type"] != SessionType.AGENT:
                raise ValueError(f"Invalid session type: {session.session_type}")

        self.session = session
        self._default_agent_initialized = False

    def append_message(self, event: MessageAddedEvent) -> None:
        """Append a message to the agent's session.

        Args:
            event: Event for a newly added Message
        """
        agent = event.agent
        message = event.message

        if agent.agent_id is None:
            raise ValueError("`agent.agent_id` must be set before appending message to session.")

        session_message = session_message_from_message(message)
        self.session_repository.create_message(self.session_id, agent.agent_id, session_message)
        self.session_repository.update_agent(
            self.session_id,
            session_agent_from_agent(agent=agent),
        )

    def initialize(self, event: AgentInitializedEvent) -> None:
        """Initialize an agent with a session.

        Args:
            event: Event when an agent is initialized
        """
        agent = event.agent

        if agent.agent_id is None:
            if self._default_agent_initialized:
                raise ValueError(
                    "By default, only one agent with no `agent_id` can be initialized within session_manager."
                    "Set `agent_id` to support more than one agent in a session."
                )
            logger.debug(
                "agent_id=<%s> | session_id=<%s> | Using default agent_id.",
                agent.agent_id,
                self.session_id,
            )
            agent.agent_id = DEFAULT_SESSION_AGENT_ID
            self._default_agent_initialized = True

        session_agent = self.session_repository.read_agent(self.session_id, agent.agent_id)

        if session_agent is None:
            logger.debug(
                "agent_id=<%s> | session_id=<%s> | Creating agent.",
                agent.agent_id,
                self.session_id,
            )

            session_agent = session_agent_from_agent(agent)
            self.session_repository.create_agent(self.session_id, session_agent)
            for message in agent.messages:
                session_message = session_message_from_message(message)
                self.session_repository.create_message(self.session_id, agent.agent_id, session_message)
        else:
            logger.debug(
                "agent_id=<%s> | session_id=<%s> | Restoring agent.",
                agent.agent_id,
                self.session_id,
            )
            agent.messages = [
                session_message_to_message(session_message)
                for session_message in self.session_repository.list_messages(self.session_id, agent.agent_id)
            ]
            agent.state = AgentState(session_agent["state"])
            agent.event_loop_metrics = EventLoopMetrics.from_dict(session_agent["event_loop_metrics"])
