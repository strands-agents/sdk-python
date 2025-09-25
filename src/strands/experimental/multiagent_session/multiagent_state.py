"""Multi-agent state data structures for session persistence.

This module defines the core data structures used to represent the state
of multi-agent orchestrators in a serializable format for session persistence.

Key Components:
- MultiAgentType: Enum for orchestrator types (Graph/Swarm)
- MultiAgentState: Serializable state container with conversion methods
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from ...types.content import ContentBlock

if TYPE_CHECKING:
    from ...multiagent.base import Status


# TODO: Move to Base after experimental
class MultiAgentType(Enum):
    """Enumeration of supported multi-agent orchestrator types.

    Attributes:
        SWARM: Collaborative agent swarm orchestrator
        GRAPH: Directed graph-based agent orchestrator
    """

    SWARM = "swarm"
    GRAPH = "graph"


@dataclass
class MultiAgentState:
    """Serializable state container for multi-agent orchestrators.

    This class represents the complete execution state of a multi-agent
    orchestrator (Graph or Swarm) in a format suitable for persistence
    and restoration across sessions.

    Attributes:
        completed_nodes: Set of node IDs that have completed execution
        node_results: Dictionary mapping node IDs to their execution results
        status: Current execution status of the orchestrator
        next_node_to_execute: List of node IDs ready for execution
        current_task: The original task being executed
        execution_order: Ordered list of executed node IDs
        error_message: Optional error message if execution failed
        type: Type of orchestrator (Graph or Swarm)
        context: Additional context data (primarily for Swarm)
    """

    # Mutual
    completed_nodes: Set[str] = field(default_factory=set)
    node_results: Dict[str, Any] = field(default_factory=dict)
    status: "Status" = "pending"
    next_node_to_execute: Optional[List[str]] = None
    current_task: Optional[str | List[ContentBlock]] = None
    execution_order: list[str] = field(default_factory=list)
    error_message: Optional[str] = None
    type: Optional[MultiAgentType] = field(default=MultiAgentType.GRAPH)
    # Swarm
    context: Optional[dict] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert MultiAgentState to JSON-serializable dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """

        def _serialize(v: Any) -> Any:
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            if isinstance(v, set):
                return list(v)
            if isinstance(v, dict):
                return {str(k): _serialize(val) for k, val in v.items()}
            if isinstance(v, (list, tuple)):
                return [_serialize(x) for x in v]
            if hasattr(v, "to_dict"):
                return v.to_dict()
            # last resort: stringize anything non-serializable (locks, objects, etc.)
            return str(v)

        return {
            "status": self.status,
            "completed_nodes": list(self.completed_nodes),
            "next_node_to_execute": list(self.next_node_to_execute) if self.next_node_to_execute else [],
            "node_results": _serialize(self.node_results),
            "current_task": self.current_task,
            "error_message": self.error_message,
            "execution_order": self.execution_order,
            "type": self.type,
            "context": _serialize(self.context),
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create MultiAgentState from dictionary data.

        Args:
            data: Dictionary containing state data

        Returns:
            MultiAgentState instance
        """
        data["completed_nodes"] = set(data.get("completed_nodes", []))
        return cls(**data)
