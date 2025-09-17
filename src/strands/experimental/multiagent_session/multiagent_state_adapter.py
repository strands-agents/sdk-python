from ...multiagent.graph import Graph,Status
from .multi_agent_state import MultiAgentState

class MultiAgentAdapter:

    @staticmethod
    def create_multi_agent_state(graph : Graph, msg: str = None) -> MultiAgentState:
        serialized_results = {}
        for node_id, node_result in graph.state.results.values():
            if hasattr(node_result, 'to_dict'):
                serialized_results[node_id] = node_result.to_dict()
            else:
                serialized_results[node_id] = str(node_result)

        return MultiAgentState(
            status=graph.state.status.value,
            completed_nodes={node.node_id for node in graph.state.completed_nodes},
            node_results=serialized_results,
            next_node_to_execute=graph.state.execution_order,
            current_task=graph.state.task,
            error_message= msg
        )

    @staticmethod
    def apply_multi_agent_state(graph : Graph, multi_agent_state : MultiAgentState):
        graph.state.status = Status(multi_agent_state.status)
        graph.state.completed_nodes = {graph.nodes[node_id] for node_id in multi_agent_state.completed_nodes}
        graph.state.results = {node_id: result for node_id, result in multi_agent_state.node_results.items()}
        graph.state.execution_order = multi_agent_state.next_node_to_execute
        graph.state.task = multi_agent_state.current_task
        graph.state.error_message = multi_agent_state.error_message