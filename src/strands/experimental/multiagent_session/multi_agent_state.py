from dataclasses import asdict, dataclass, field, is_dataclass, fields
from typing import Any, Dict, List, Optional, Set,TYPE_CHECKING

from ...types.content import ContentBlock
if TYPE_CHECKING:
    from ...multiagent.base import Status


@dataclass
class MultiAgentState:
    completed_nodes: Set[str] = field(default_factory=set)
    node_results: Dict[str, Any] = field(default_factory=dict)
    status: "Status" = "pending"# There is an import circular issue
    next_node_to_execute: Optional[List[str]] = None
    current_task: Optional[str | List[ContentBlock]] = None
    execution_order : list[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        def _serialize(v: Any) -> Any:
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            if isinstance(v, set):
                return list(v)
            if isinstance(v, dict):
                return {str(k): _serialize(val) for k, val in v.items()}
            if isinstance(v, (list, tuple)):
                return [_serialize(x) for x in v]
            if hasattr(v, 'to_dict'):
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
            "execution_order": self.execution_order
        }


    @classmethod
    def from_dict(cls, data: dict):
        data["completed_nodes"] = set(data.get("completed_nodes", []))
        return cls(**data)
