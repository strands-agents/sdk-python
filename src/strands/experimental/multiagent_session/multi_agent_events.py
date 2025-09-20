from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...hooks.registry import HookEvent
from .multi_agent_state import MultiAgentState

if TYPE_CHECKING:
    from ...multiagent.graph import Graph

"""""
Why we use str instead of real ObjectType?
GraphNode contains  executor: Agent | MultiAgentBase, which is not serializable now.
Same reason for Graph 
""" ""


@dataclass
class MultiAgentInitializationEvent(HookEvent):
    graph: "Graph"
    state: MultiAgentState


@dataclass
class BeforeGraphInvocationEvent(HookEvent):
    graph: "Graph"
    state: MultiAgentState


@dataclass
class BeforeNodeInvocationEvent(HookEvent):
    graph: "Graph"
    next_node_to_execute: str  # node_id


@dataclass
class AfterNodeInvocationEvent(HookEvent):
    graph: "Graph"
    executed_node: str  # node_id
    state: MultiAgentState


@dataclass
class AfterGraphInvocationEvent(HookEvent):
    graph: "Graph"
    state: MultiAgentState
