"""Session manager interface for agent session management."""

import logging
import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..experimental.hooks.multiagent_hooks.multiagent_events import (
    AfterMultiAgentInvocationEvent,
    AfterNodeCallEvent,
    MultiAgentInitializedEvent,
)
from ..hooks.events import AfterInvocationEvent, AgentInitializedEvent, MessageAddedEvent
from ..hooks.registry import HookProvider, HookRegistry
from ..types.content import Message
from ..types.session import SessionType

if TYPE_CHECKING:
    from ..agent.agent import Agent
    from ..multiagent.base import MultiAgentBase

logger = logging.getLogger(__name__)


class SessionManager(HookProvider, ABC):
    """Abstract interface for managing sessions.

    A session manager is in charge of persisting the conversation and state of an agent across its interaction.
    Changes made to the agents conversation, state, or other attributes should be persisted immediately after
    they are changed. The different methods introduced in this class are called at important lifecycle events
    for an agent, and should be persisted in the session.
    """

    def __init__(self, session_type: SessionType = SessionType.AGENT) -> None:
        """Initialize SessionManager with session type.

        Args:
            session_type: Type of session (AGENT or MULTI_AGENT)
        """
        self.session_type: SessionType = session_type
        self._lock = threading.RLock()

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks for persisting the agent to the session."""
        if self.session_type == SessionType.AGENT:
            # After the normal Agent initialization behavior, call the session initialize function to restore the agent
            registry.add_callback(AgentInitializedEvent, lambda event: self.initialize(event.agent))

            # For each message appended to the Agents messages, store that message in the session
            registry.add_callback(MessageAddedEvent, lambda event: self.append_message(event.message, event.agent))

            # Sync the agent into the session for each message in case the agent state was updated
            registry.add_callback(MessageAddedEvent, lambda event: self.sync_agent(event.agent))

            # After an agent was invoked, sync it with the session to capture any conversation manager state updates
            registry.add_callback(AfterInvocationEvent, lambda event: self.sync_agent(event.agent))

        elif self.session_type == SessionType.MULTI_AGENT:
            registry.add_callback(MultiAgentInitializedEvent, self._on_multiagent_initialized)
            registry.add_callback(AfterNodeCallEvent, lambda event: self._persist_multi_agent_state(event.source))
            registry.add_callback(
                AfterMultiAgentInvocationEvent, lambda event: self._persist_multi_agent_state(event.source)
            )

    @abstractmethod
    def redact_latest_message(self, redact_message: Message, agent: "Agent", **kwargs: Any) -> None:
        """Redact the message most recently appended to the agent in the session.

        Args:
            redact_message: New message to use that contains the redact content
            agent: Agent to apply the message redaction to
            **kwargs: Additional keyword arguments for future extensibility.
        """

    @abstractmethod
    def append_message(self, message: Message, agent: "Agent", **kwargs: Any) -> None:
        """Append a message to the agent's session.

        Args:
            message: Message to add to the agent in the session
            agent: Agent to append the message to
            **kwargs: Additional keyword arguments for future extensibility.
        """

    @abstractmethod
    def sync_agent(self, agent: "Agent", **kwargs: Any) -> None:
        """Serialize and sync the agent with the session storage.

        Args:
            agent: Agent who should be synchronized with the session storage
            **kwargs: Additional keyword arguments for future extensibility.
        """

    @abstractmethod
    def initialize(self, agent: "Agent", **kwargs: Any) -> None:
        """Initialize an agent with a session.

        Args:
            agent: Agent to initialize
            **kwargs: Additional keyword arguments for future extensibility.
        """

    def _persist_multi_agent_state(self, source: "MultiAgentBase") -> None:
        """Thread-safe persistence of multi-agent state.

        Args:
            source: Multi-agent orchestrator to persist
        """
        with self._lock:
            state = source.serialize_state()
            self.write_multi_agent_json(state)

    def write_multi_agent_json(self, state: dict[str, Any]) -> None:
        """Write multi-agent state to persistent storage.

        Args:
            state: Multi-agent state dictionary to persist
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support multi-agent persistence "
            "(write_multi_agent_json). Provide an implementation or use a "
            "SessionManager with session_type=SessionType.MULTI_AGENT."
        )

    def read_multi_agent_json(self) -> dict[str, Any]:
        """Read multi-agent state from persistent storage.

        Returns:
            Multi-agent state dictionary or empty dict if not found
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support multi-agent persistence "
            "(read_multi_agent_json). Provide an implementation or use a "
            "SessionManager with session_type=SessionType.MULTI_AGENT."
        )

    def _on_multiagent_initialized(self, event: MultiAgentInitializedEvent) -> None:
        """Initialization path: attempt to resume and then persist a fresh snapshot."""
        source: MultiAgentBase = event.source
        payload = self.read_multi_agent_json()
        # payload can be {} or Graph/Swarm state json
        if payload:
            source.deserialize_state(payload)
        else:
            try:
                self._persist_multi_agent_state(source)
            except NotImplementedError:
                pass
