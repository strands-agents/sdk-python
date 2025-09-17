"""Session manager interface for agent session management."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from ..hooks.events import AfterInvocationEvent, AgentInitializedEvent, MessageAddedEvent
from ..hooks.registry import HookProvider, HookRegistry
from ..types.content import Message
from ..multiagent import Graph
from ..experimental.multiagent_session import MultiAgentState

if TYPE_CHECKING:
    from ..agent.agent import Agent


class SessionManager(HookProvider, ABC):
    """Abstract interface for managing sessions.

    A session manager is in charge of persisting the conversation and state of an agent across its interaction.
    Changes made to the agents conversation, state, or other attributes should be persisted immediately after
    they are changed. The different methods introduced in this class are called at important lifecycle events
    for an agent, and should be persisted in the session.
    """

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks for persisting the agent to the session."""
        # After the normal Agent initialization behavior, call the session initialize function to restore the agent
        registry.add_callback(AgentInitializedEvent, lambda event: self.initialize(event.agent))

        # For each message appended to the Agents messages, store that message in the session
        registry.add_callback(MessageAddedEvent, lambda event: self.append_message(event.message, event.agent))

        # Sync the agent into the session for each message in case the agent state was updated
        registry.add_callback(MessageAddedEvent, lambda event: self.sync_agent(event.agent))

        # After an agent was invoked, sync it with the session to capture any conversation manager state updates
        registry.add_callback(AfterInvocationEvent, lambda event: self.sync_agent(event.agent))

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
    @abstractmethod
    def write_multi_agent_state(self, session_id, state: MultiAgentState) -> None:
        """
        Write the multi-agent state to the session storage.
        :param session_id:
        :param state:
        :param kwargs:
        """

    @abstractmethod
    def read_multi_agent_state(self) -> MultiAgentState:
        """
        Read the multi-agent state from the session storage.
        :param session_id:
        :param kwargs:
        :return: MultiAgentState
        """

    @abstractmethod
    def write_multi_agent_metadata(self, grap: Graph, graph_hash: str):
        """
        Write the multi-agent metadata to the session storage.
        :param grap:
        :param graph_hash:
        :param kwargs:
        """

    @abstractmethod
    def read_multi_agent_metadata(self) -> Graph:
        """
        Read the multi-agent metadata from the session storage.
        """

    @abstractmethod
    def write_multi_agent_graph(self, graph: Graph, graph_hash: str) -> None:
        """
        Write the multi-agent graph to the session storage.
        :param graph:
        :param graph_hash:
        :param kwargs:
        """

    @abstractmethod
    def read_multi_agent_graph(self) -> Optional[dict]:
        """
        Read the multi-agent graph from the session storage.
        :return: a
        """
