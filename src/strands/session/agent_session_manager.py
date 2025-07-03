"""File-based implementation of session manager."""

import logging
from typing import TYPE_CHECKING, Any, List
from uuid import uuid4

from ..agent.state import AgentState
from ..handlers.callback_handler import CompositeCallbackHandler
from ..types.content import Message
from ..types.exceptions import SessionException
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
        session_dao: SessionDAO,
    ):
        """Initialize the FileSessionManager."""
        self.session_dao = session_dao
        self.session_id = session_id

    def append_message(self, agent: "Agent", message: Message) -> None:
        """Append a message to the agent's session.

        Args:
            agent: The agent whose session to update
            message: The message to append
        """
        if agent.id is None:
            raise ValueError("`agent.id` must be set before appending message to session.")

        session_message = SessionMessage.from_dict(dict(message))
        self.session_dao.create_message(self.session_id, agent.id, session_message)
        self.session_dao.update_agent(
            self.session_id,
            SessionAgent(
                agent_id=agent.id,
                session_id=self.session_id,
                event_loop_metrics=agent.event_loop_metrics.to_dict(),
                state=agent.state.get(),
            ),
        )

    def initialize(self, agent: "Agent") -> None:
        """Restore agent data from the current session.

        Args:
            agent: Agent instance to restore session data to

        Raises:
            SessionException: If restore operation fails
        """
        try:
            # Try to read existing session
            session = self.session_dao.read_session(self.session_id)

            if agent.id is None:
                agents: List[SessionAgent] = self.session_dao.list_agents(self.session_id)
                if len(agents) == 0:
                    agent_id = str(uuid4())
                if len(agents) == 1:
                    agent_id = agents[0].agent_id
                    logger.debug(
                        "session_id=<%s> | agent_id=<%s> | Restoring agent data from session", self.session_id, agent_id
                    )
                else:
                    raise ValueError(
                        "If there is more than one agent in a session, agent.agent_id must be set manually."
                    )
            else:
                if agent.id not in [agent.agent_id for agent in self.session_dao.list_agents(self.session_id)]:
                    raise ValueError(f"Agent {agent.id} not found in session {self.session_id}")
                agent_id = agent.id

            if session.session_type != SessionType.AGENT:
                raise ValueError(f"Invalid session type: {session.session_type}")

            # Initialize agent
            agent.id = agent_id
            agent.messages = [
                session_message.to_message()
                for session_message in self.session_dao.list_messages(self.session_id, agent.id)
            ]
            agent.state = AgentState(self.session_dao.read_agent(self.session_id, agent.id).state)

        except SessionException:
            # Session doesn't exist, create new one
            logger.debug("session_id=<%s> | Session not found, creating new session")
            # Session doesn't exist, create new one
            if agent.id is None:
                agent_id = str(uuid4())
                logger.debug("agent_id=<%s> | Creating agent_id for agent since none was set.", agent_id)
                agent.id = agent_id
            session = Session(session_id=self.session_id, session_type=SessionType.AGENT)
            session_agent = SessionAgent(
                agent_id=agent.id,
                session_id=self.session_id,
                event_loop_metrics=agent.event_loop_metrics.to_dict(),
                state=agent.state.get(),
            )
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
                    self.append_message(kwargs["agent"], message)
            except Exception as e:
                logger.error("Persistence operation failed", e)

        agent.callback_handler = CompositeCallbackHandler(agent.callback_handler, session_callback)
