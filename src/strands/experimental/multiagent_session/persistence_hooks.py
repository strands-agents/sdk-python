"""Multi-agent session persistence hook implementation.

This module provides automatic session persistence for multi-agent orchestrators
(Graph and Swarm) by hooking into their execution lifecycle events.

Key Features:
- Automatic state persistence at key execution points
- Thread-safe persistence operations
- Support for both Graph and Swarm orchestrators
- Seamless integration with SessionManager
"""

import threading
from typing import Optional

from ...hooks.registry import HookProvider, HookRegistry
from ...multiagent.base import MultiAgentBase
from ...session import SessionManager
from .multiagent_events import (
    AfterGraphInvocationEvent,
    AfterNodeInvocationEvent,
    BeforeGraphInvocationEvent,
    BeforeNodeInvocationEvent,
    MultiAgentInitializationEvent,
    MultiAgentState,
)
from .multiagent_state_adapter import MultiAgentAdapter


def _get_multiagent_state(
    multiagent_state: Optional[MultiAgentState],
    orchestrator: MultiAgentBase,
) -> MultiAgentState:
    if multiagent_state is not None:
        return multiagent_state

    return MultiAgentAdapter.create_multi_agent_state(orchestrator=orchestrator)


class MultiAgentHook(HookProvider):
    """Hook provider for automatic multi-agent session persistence.

    This hook automatically persists multi-agent orchestrator state at key
    execution points to enable resumable execution after interruptions.

    Args:
        session_manager: SessionManager instance for state persistence
        session_id: Unique identifier for the session
    """

    def __init__(self, session_manager: SessionManager, session_id: str):
        """Initialize the multi-agent persistence hook.

        Args:
            session_manager: SessionManager instance for state persistence
            session_id: Unique identifier for the session
        """
        self._session_manager = session_manager
        self._session_id = session_id
        self._lock = threading.RLock()

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        """Register persistence callbacks for multi-agent execution events.

        Args:
            registry: Hook registry to register callbacks with
            **kwargs: Additional keyword arguments (unused)
        """
        registry.add_callback(MultiAgentInitializationEvent, self._on_initialization)
        registry.add_callback(BeforeGraphInvocationEvent, self._on_before_graph)
        registry.add_callback(BeforeNodeInvocationEvent, self._on_before_node)
        registry.add_callback(AfterNodeInvocationEvent, self._on_after_node)
        registry.add_callback(AfterGraphInvocationEvent, self._on_after_graph)

    def _on_initialization(self, event: MultiAgentInitializationEvent):
        """Persist state when multi-agent orchestrator initializes."""
        self._persist(_get_multiagent_state(event.state, event.orchestrator))

    def _on_before_graph(self, event: BeforeGraphInvocationEvent):
        """Hook called before graph execution starts."""
        pass

    def _on_before_node(self, event: BeforeNodeInvocationEvent):
        """Hook called before individual node execution."""
        pass

    def _on_after_node(self, event: AfterNodeInvocationEvent):
        """Persist state after each node completes execution."""
        multi_agent_state = _get_multiagent_state(multiagent_state=event.state, orchestrator=event.orchestrator)
        self._persist(multi_agent_state)

    def _on_after_graph(self, event: AfterGraphInvocationEvent):
        """Persist final state after graph execution completes."""
        multiagent_state = _get_multiagent_state(multiagent_state=event.state, orchestrator=event.orchestrator)
        self._persist(multiagent_state)

    def _persist(self, multiagent_state: MultiAgentState) -> None:
        """Persist the provided MultiAgentState using the configured SessionManager.

        This method is synchronized across threads/tasks to avoid write races.

        Args:
            multiagent_state: State to persist
        """
        with self._lock:
            self._session_manager.write_multi_agent_state(multiagent_state)
