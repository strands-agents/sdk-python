from typing import TYPE_CHECKING
from .multi_agent_state import MultiAgentState

if TYPE_CHECKING:
    from ...multiagent.graph import Graph


class MultiAgentAdapter:
    @staticmethod
    def create_multi_agent_state(graph: "Graph", msg: str = None) -> MultiAgentState:
        from ...multiagent.graph import Status
        serialized_results = {
            node_id: graph._summarize_node_result_for_persist(node_result)
            for node_id, node_result in (graph.state.results or {}).items()
        }

        inflight = [n.node_id for n in graph.nodes.values() if n.execution_status == Status.EXECUTING]

        # 2) otherwise, dependency-ready nodes; 3) otherwise, entry points not completed
        if inflight:
            next_nodes_ids = inflight
        else:
            next_nodes_ids = [n.node_id for n in graph._compute_ready_nodes_for_resume()]

        # status may be an Enum or a string depending on caller; persist the string value
        status_str = graph.state.status.value if isinstance(graph.state.status, Status) else str(graph.state.status)


        return MultiAgentState(
            status= Status(status_str), # type: ignore
            completed_nodes={node.node_id for node in graph.state.completed_nodes},
            node_results=serialized_results,
            next_node_to_execute= next_nodes_ids,
            current_task=graph.state.task,
            error_message=msg,
            execution_order=[n.node_id for n in graph.state.execution_order]
        )

    @staticmethod
    def apply_multi_agent_state(graph: "Graph", multi_agent_state: MultiAgentState):
        from ...multiagent.graph import Status
        
        graph.state.status = Status(multi_agent_state.status)
        graph.state.completed_nodes = {graph.nodes[node_id] for node_id in multi_agent_state.completed_nodes if node_id in graph.nodes}
        graph.state.results = {node_id: result for node_id, result in multi_agent_state.node_results.items()}
        execution_node_ids = getattr(multi_agent_state, "execution_order", []) or []
        graph.state.execution_order = [
            graph.nodes[node_id] for node_id in execution_node_ids
            if node_id in graph.nodes and graph.nodes[node_id] in graph.state.completed_nodes
        ]

        graph.state.task = multi_agent_state.current_task
        graph.state.error_message = multi_agent_state.error_message
