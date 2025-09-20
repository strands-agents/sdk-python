import threading
from typing import Optional

from .multiagent_state_adapter import MultiAgentAdapter

from ...hooks.registry import HookProvider, HookRegistry
from ...session import SessionManager
from .multi_agent_events import (
    AfterGraphInvocationEvent,
    AfterNodeInvocationEvent,
    BeforeGraphInvocationEvent,
    BeforeNodeInvocationEvent,
    MultiAgentInitializationEvent,
    MultiAgentState,
)


def _get_multiagent_state(
    multiagent_state: Optional[MultiAgentState],
    graph,
) -> MultiAgentState:
    if multiagent_state is not None:
        return multiagent_state

    # recompute and return
    return MultiAgentAdapter.create_multi_agent_state(graph)


class MultiAgentHook(HookProvider):
    def __init__(self, session_manager: SessionManager, session_id: str):
        self._session_manager = session_manager
        self._session_id = session_id
        self._lock = threading.RLock()

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        registry.add_callback(MultiAgentInitializationEvent, self._on_initialization)
        registry.add_callback(BeforeGraphInvocationEvent, self._on_before_graph)
        registry.add_callback(BeforeNodeInvocationEvent, self._on_before_node)
        registry.add_callback(AfterNodeInvocationEvent, self._on_after_node)
        registry.add_callback(AfterGraphInvocationEvent, self._on_after_graph)

    def _on_initialization(self, event: MultiAgentInitializationEvent):
        multi_agent_state = _get_multiagent_state(event.state, event.graph)
        self._persist(multi_agent_state)

    def _on_before_graph(self, event: BeforeGraphInvocationEvent):
        # TODO: To add logic here if needed.
        pass

    def _on_before_node(self, event: BeforeNodeInvocationEvent):
        # TODO: This allows human-in-the-loop,extra parameter required.
        pass

    def _on_after_node(self, event: AfterNodeInvocationEvent):
        multi_agent_state = _get_multiagent_state(event.state, event.graph)
        self._persist(multi_agent_state)

    def _on_after_graph(self, event: AfterGraphInvocationEvent):
        multiagent_state = _get_multiagent_state(event.state, event.graph)
        self._persist(multiagent_state)

    def _persist(self, multiagent_state: MultiAgentState) -> None:
        """Persist the provided MultiAgentState using the configured SessionManager.
        This method is synchronized across threads/tasks to avoid write races.
        """
        with self._lock:
            self._session_manager.write_multi_agent_state(multiagent_state)
