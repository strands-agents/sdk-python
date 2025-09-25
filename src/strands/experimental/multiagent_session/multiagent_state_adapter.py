"""Multi-agent state adapter for session persistence.

This module provides bidirectional conversion between multi-agent orchestrator
runtime state and serializable MultiAgentState objects for session persistence.

Key Features:
- State serialization for Graph and Swarm orchestrators
- State restoration from persisted sessions
- Node result summarization for efficient storage
- Type-safe state conversion with error handling
"""

import ast
import logging
from typing import Any

from .multiagent_state import MultiAgentState, MultiAgentType

logger = logging.getLogger(__name__)


class MultiAgentAdapter:
    """Adapter for converting between orchestrator runtime state and persistent state.

    This class provides static methods for bidirectional conversion between
    multi-agent orchestrator objects (Graph/Swarm) and serializable MultiAgentState.
    """

    @staticmethod
    def apply_multi_agent_state(orchestrator: object, multi_agent_state: MultiAgentState):
        """Apply persisted state to a multi-agent orchestrator.

        Args:
            orchestrator: Graph or Swarm instance to restore state to
            multi_agent_state: Persisted state to apply

        Raises:
            ValueError: If state type is incompatible with orchestrator
        """
        from ...multiagent.base import Status
        from ...multiagent.graph import Graph
        from ...multiagent.swarm import Swarm

        state_type = getattr(multi_agent_state, "type", None)
        type_val = str(getattr(state_type, "value", state_type))
        if isinstance(orchestrator, Graph) and type_val == "graph":
            graph = orchestrator
            graph.state.status = Status(multi_agent_state.status)
            graph.state.completed_nodes = {
                graph.nodes[node_id] for node_id in multi_agent_state.completed_nodes if node_id in graph.nodes
            }
            graph.state.results = {
                node_id: result for node_id, result in getattr(multi_agent_state, "node_results", {}).items()
            }
            execution_node_ids = getattr(multi_agent_state, "execution_order", []) or []
            graph.state.execution_order = [
                graph.nodes[node_id]
                for node_id in execution_node_ids
                if node_id in graph.nodes and graph.nodes[node_id] in graph.state.completed_nodes
            ]

            graph.state.task = getattr(multi_agent_state, "current_task", "")
            for node in graph.state.completed_nodes:
                node.execution_status = Status.COMPLETED

            return

        elif isinstance(orchestrator, Swarm) and type_val == "swarm":
            swarm = orchestrator

            swarm.state.completion_status = Status(multi_agent_state.status)

            swarm.state.node_history = [
                swarm.nodes[nid] for nid in (multi_agent_state.completed_nodes or []) if nid in swarm.nodes
            ]

            completed_ids = {n.node_id for n in swarm.state.node_history}
            saved_results = getattr(multi_agent_state, "node_results", {}) or {}
            swarm.state.results = {k: v for k, v in saved_results.items() if k in completed_ids}

            # current_node
            next_ids = getattr(multi_agent_state, "next_node_to_execute", []) or []
            swarm.state.current_node = swarm.nodes.get(next_ids[0]) if next_ids else None

            swarm.state.task = multi_agent_state.current_task
            # hydrate context
            context = getattr(multi_agent_state, "context", {}) or {}
            shared_context = context.get("shared_context") or {}
            swarm.shared_context.context = shared_context
            swarm.state.handoff_message = context.get("handoff_message")

        else:
            raise ValueError("Persisted state type incompatible with current orchestrator")

    @staticmethod
    def create_multi_agent_state(orchestrator: object, msg: str = None) -> MultiAgentState | None:
        """Create serializable state from multi-agent orchestrator.

        Args:
            orchestrator: Graph or Swarm instance to extract state from
            msg: Optional error message to include in state

        Returns:
            MultiAgentState object ready for persistence, or None if unsupported type
        """
        from ...multiagent.base import Status
        from ...multiagent.graph import Graph
        from ...multiagent.swarm import Swarm

        if isinstance(orchestrator, Graph):
            graph = orchestrator

            serialized_results = {
                node_id: MultiAgentAdapter.summarize_node_result_for_persist(node_result)
                for node_id, node_result in (graph.state.results or {}).items()
            }

            inflight = [n.node_id for n in graph.nodes.values() if n.execution_status == Status.EXECUTING]
            next_nodes_ids = inflight or [n.node_id for n in graph._compute_ready_nodes_for_resume()]

            return MultiAgentState(
                type=MultiAgentType.GRAPH,
                status=graph.state.status,
                completed_nodes={node.node_id for node in graph.state.completed_nodes},
                node_results=serialized_results,
                next_node_to_execute=next_nodes_ids,
                current_task=graph.state.task,
                error_message=msg,
                execution_order=[n.node_id for n in graph.state.execution_order],
                context={},
            )

        elif isinstance(orchestrator, Swarm):
            swarm = orchestrator
            current_executing_node = (
                [swarm.state.current_node.node_id] if swarm.state.completion_status == Status.EXECUTING else []
            )

            serialized_results = {
                node_id: MultiAgentAdapter.summarize_node_result_for_persist(res) or str(res)
                for node_id, res in (swarm.state.results or {}).items()
            }

            shared_ctx = {}
            if hasattr(swarm, "shared_context") and swarm.shared_context is not None:
                shared_ctx = getattr(swarm.shared_context, "context", {}) or {}

            return MultiAgentState(
                type=MultiAgentType.SWARM,
                status=swarm.state.completion_status,
                completed_nodes={node.node_id for node in swarm.state.node_history},
                node_results=serialized_results,
                next_node_to_execute=current_executing_node,
                current_task=swarm.state.task,
                error_message=msg,
                execution_order=[node.node_id for node in swarm.state.node_history],
                context={
                    "shared_context": shared_ctx,
                    "handoff_message": getattr(swarm.state, "handoff_message", None),
                },
            )
        return None

    @staticmethod
    def summarize_node_result_for_persist(raw: Any) -> dict[str, Any]:
        """Summarize node execution result for efficient persistence.

        Args:
            raw: Raw node result (NodeResult, dict, string, or other)

        Returns:
            Normalized dict with 'agent_outputs' key containing string results
        """
        if hasattr(raw, "get_agent_results") and callable(raw.get_agent_results):
            try:
                results = raw.get_agent_results()
                texts = [str(r) for r in results]
                return {"agent_outputs": texts}
            except Exception:
                pass

        # Already a dict
        if isinstance(raw, dict):
            return MultiAgentAdapter._normalize_persisted_like_dict(raw)

        # String that might itself be a dict represent
        if isinstance(raw, str):
            try:
                parsed = ast.literal_eval(raw)
                if isinstance(parsed, dict):
                    return MultiAgentAdapter._normalize_persisted_like_dict(parsed)
            except Exception as e:
                logger.debug("Failed to parse persisted node result: %s", e)
            return {"agent_outputs": [raw]}

        return {"agent_outputs": [str(raw)]}

    @staticmethod
    def _normalize_persisted_like_dict(data: dict[str, Any]) -> dict[str, Any]:
        """Normalize dictionary data to standard agent_outputs format.

        Args:
            data: Dictionary containing result data

        Returns:
            Normalized dict with 'agent_outputs' key
        """
        if "agent_outputs" in data and isinstance(data["agent_outputs"], list):
            return {"agent_outputs": [str(x) for x in data["agent_outputs"]]}

        if "summary" in data:
            return {"agent_outputs": [str(data["summary"])]}
        return {"agent_outputs": [str(data)]}
