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
from typing import TYPE_CHECKING

from ...hooks.registry import HookProvider, HookRegistry
from ...session import SessionManager
from .multiagent_events import (
    AfterMultiAgentInvocationEvent,
    AfterNodeInvocationEvent,
    BeforeMultiAgentInvocationEvent,
    BeforeNodeInvocationEvent,
    MultiAgentInitializationEvent,
)

if TYPE_CHECKING:
    from ...multiagent.base import MultiAgentBase


class PersistentHook(HookProvider):
    """Hook provider for automatic multi-agent session persistence.

    This hook automatically persists multi-agent orchestrator state at key
    execution points to enable resumable execution after interruptions.

    """

    def __init__(self, session_manager: SessionManager):
        """Initialize the multi-agent persistence hook.

        Args:
            session_manager: SessionManager instance for state persistence
        """
        self._session_manager = session_manager
        self._lock = threading.RLock()

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        """Register persistence callbacks for multi-agent execution events.

        Args:
            registry: Hook registry to register callbacks with
            **kwargs: Additional keyword arguments (unused)
        """
        registry.add_callback(MultiAgentInitializationEvent, self._on_initialization)
        registry.add_callback(BeforeMultiAgentInvocationEvent, self._on_before_multiagent)
        registry.add_callback(BeforeNodeInvocationEvent, self._on_before_node)
        registry.add_callback(AfterNodeInvocationEvent, self._on_after_node)
        registry.add_callback(AfterMultiAgentInvocationEvent, self._on_after_multiagent)

    # TODO: We can add **kwarg or invocation_state later if we need to persist
    def _on_initialization(self, event: MultiAgentInitializationEvent) -> None:
        """Persist state when multi-agent orchestrator initializes."""
        self._persist(event.orchestrator)

    def _on_before_multiagent(self, event: BeforeMultiAgentInvocationEvent) -> None:
        """Persist state when multi-agent orchestrator initializes."""
        pass

    def _on_before_node(self, event: BeforeNodeInvocationEvent) -> None:
        """Hook called before individual node execution."""
        pass

    def _on_after_node(self, event: AfterNodeInvocationEvent) -> None:
        """Persist state after each node completes execution."""
        self._persist(event.orchestrator)

    def _on_after_multiagent(self, event: AfterMultiAgentInvocationEvent) -> None:
        """Persist final state after graph execution completes."""
        self._persist(event.orchestrator)

    def _persist(self, orchestrator: "MultiAgentBase") -> None:
        """Persist the provided MultiAgentState using the configured SessionManager.

        This method is synchronized across threads/tasks to avoid write races.

        Args:
            orchestrator: State to persist
        """
        current_state = orchestrator.get_state_from_orchestrator()
        with self._lock:
            self._session_manager.write_multi_agent_json(current_state)
