import json
from dataclasses import dataclass, field, asdict
from typing import Set, Any, List, Optional, Dict

from.base import Status
from ..types.content import ContentBlock, Message, Messages
from .graph import Graph,GraphState

@dataclass
class MultiAgentState:
    completed_nodes: Set[str] = field(default_factory=set)
    node_results: Dict[str, Any] = field(default_factory=dict)
    status: Status = Status.PENDING
    next_node_to_execute: Optional[List[str]] = None
    current_task: Optional[str | List[ContentBlock]] = None
    error_message: Optional[str] = None
    
    def to_dict(self):
        data = asdict(self)
        data['completed_nodes'] = sorted(list(self.completed_nodes))
        return data

    @classmethod
    def from_dict(cls, data: dict):
        data['completed_nodes'] = set(data.get('completed_nodes', []))
        return cls(**data)

