"""Directed Acyclic Graph (DAG) Multi-Agent Pattern Implementation.

This module provides a deterministic DAG-based agent orchestration system where
agents or MultiAgentBase instances (like Swarm or Graph) are nodes in a graph,
executed according to edge dependencies, with output from one node passed as input
to connected nodes.

Key Features:
- Agents and MultiAgentBase instances (Swarm, Graph, etc.) as graph nodes
- Deterministic execution order based on DAG structure
- Output propagation along edges
- Topological sort for execution ordering
- Clear dependency management
- Supports nested graphs (Graph as a node in another Graph)
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Tuple

from ..agent import Agent, AgentResult
from ..types.event_loop import Metrics, Usage
from .base import MultiAgentBase, MultiAgentResult, NodeResult, Status

logger = logging.getLogger(__name__)


@dataclass
class GraphState:
    """Graph execution state."""

    # Execution state
    status: Status = Status.PENDING
    completed_nodes: set["GraphNode"] = field(default_factory=set)
    failed_nodes: set["GraphNode"] = field(default_factory=set)
    execution_order: list["GraphNode"] = field(default_factory=list)
    task: str = ""

    # Results
    results: dict[str, NodeResult] = field(default_factory=dict)

    # Accumulated metrics
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_count: int = 0
    execution_time: int = 0

    # Graph structure info
    total_nodes: int = 0
    edges: list[Tuple["GraphNode", "GraphNode"]] = field(default_factory=list)
    entry_points: list["GraphNode"] = field(default_factory=list)


@dataclass
class GraphResult(MultiAgentResult):
    """Result from graph execution - extends MultiAgentResult with graph-specific details."""

    status: Status = Status.PENDING
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    execution_order: list["GraphNode"] = field(default_factory=list)
    edges: list[Tuple["GraphNode", "GraphNode"]] = field(default_factory=list)
    entry_points: list["GraphNode"] = field(default_factory=list)


@dataclass
class GraphEdge:
    """Represents an edge in the graph with an optional condition."""

    from_node: "GraphNode"
    to_node: "GraphNode"
    condition: Callable[[GraphState], bool] | None = None

    def __hash__(self) -> int:
        """Return hash for GraphEdge based on from_node and to_node."""
        return hash((self.from_node.node_id, self.to_node.node_id))

    def should_traverse(self, state: GraphState) -> bool:
        """Check if this edge should be traversed based on condition."""
        if self.condition is None:
            return True
        return self.condition(state)


@dataclass
class GraphNode:
    """Represents a node in the graph."""

    node_id: str
    executor: Agent | MultiAgentBase
    dependencies: set["GraphNode"] = field(default_factory=set)
    status: Status = Status.PENDING
    result: NodeResult | None = None
    execution_time: int = 0

    def __hash__(self) -> int:
        """Return hash for GraphNode based on node_id."""
        return hash(self.node_id)

    def __eq__(self, other: Any) -> bool:
        """Return equality for GraphNode based on node_id."""
        if not isinstance(other, GraphNode):
            return False
        return self.node_id == other.node_id


class GraphBuilder:
    """Builder pattern for constructing graphs."""

    def __init__(self) -> None:
        """Initialize GraphBuilder with empty collections."""
        self.nodes: dict[str, GraphNode] = {}
        self.edges: set[GraphEdge] = set()
        self.entry_points: set[GraphNode] = set()

    def add_node(self, executor: Agent | MultiAgentBase, node_id: str | None = None) -> "GraphBuilder":
        """Add an Agent or MultiAgentBase instance as a node to the graph."""
        # Auto-generate node_id if not provided
        if node_id is None:
            node_id = getattr(executor, "id", None) or getattr(executor, "name", None) or f"node_{len(self.nodes)}"

        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists")

        self.nodes[node_id] = GraphNode(node_id=node_id, executor=executor)
        return self

    def add_edge(
        self, from_node: str, to_node: str, condition: Callable[[GraphState], bool] | None = None
    ) -> "GraphBuilder":
        """Add an edge between two nodes with optional condition function that receives full GraphState."""
        # Validate nodes exist
        for node_name, node_id in [("Source", from_node), ("Target", to_node)]:
            if node_id not in self.nodes:
                raise ValueError(f"{node_name} node '{node_id}' not found")

        from_node_obj = self.nodes[from_node]
        to_node_obj = self.nodes[to_node]

        # Add edge and update dependencies
        self.edges.add(GraphEdge(from_node=from_node_obj, to_node=to_node_obj, condition=condition))
        to_node_obj.dependencies.add(from_node_obj)
        return self

    def set_entry_point(self, node_id: str) -> "GraphBuilder":
        """Set a node as an entry point for graph execution."""
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' not found")
        self.entry_points.add(self.nodes[node_id])
        return self

    def build(self) -> "Graph":
        """Build and validate the graph."""
        if not self.nodes:
            raise ValueError("Graph must contain at least one node")

        # Auto-detect entry points if none specified
        if not self.entry_points:
            self.entry_points = {node for node_id, node in self.nodes.items() if not node.dependencies}
            logger.debug(
                "entry_points=<%s> | auto-detected entrypoints", ", ".join(node.node_id for node in self.entry_points)
            )
            if not self.entry_points:
                raise ValueError("No entry points found - all nodes have dependencies")

        # Validate entry points and check for cycles
        self._validate_graph()

        return Graph(nodes=self.nodes.copy(), edges=self.edges.copy(), entry_points=self.entry_points.copy())

    def _validate_graph(self) -> None:
        """Validate graph structure and detect cycles."""
        # Validate entry points exist
        entry_point_ids = {node.node_id for node in self.entry_points}
        invalid_entries = entry_point_ids - set(self.nodes.keys())
        if invalid_entries:
            raise ValueError(f"Entry points not found in nodes: {invalid_entries}")

        # Check for cycles using DFS with color coding
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {node_id: WHITE for node_id in self.nodes}

        def has_cycle_from(node_id: str) -> bool:
            if colors[node_id] == GRAY:
                return True  # Back edge found - cycle detected
            if colors[node_id] == BLACK:
                return False

            colors[node_id] = GRAY
            # Check all outgoing edges for cycles
            for edge in self.edges:
                if edge.from_node.node_id == node_id and has_cycle_from(edge.to_node.node_id):
                    return True
            colors[node_id] = BLACK
            return False

        # Check for cycles from each unvisited node
        if any(colors[node_id] == WHITE and has_cycle_from(node_id) for node_id in self.nodes):
            raise ValueError("Graph contains cycles - must be a directed acyclic graph")


class Graph(MultiAgentBase):
    """Directed Acyclic Graph multi-agent orchestration."""

    def __init__(self, nodes: dict[str, GraphNode], edges: set[GraphEdge], entry_points: set[GraphNode]) -> None:
        """Initialize Graph."""
        super().__init__()

        self.nodes = nodes
        self.edges = edges
        self.entry_points = entry_points
        self.state = GraphState()

    def execute(self, task: str) -> GraphResult:
        """Execute task synchronously."""

        def execute() -> GraphResult:
            return asyncio.run(self.execute_async(task))

        with ThreadPoolExecutor() as executor:
            future = executor.submit(execute)
            return future.result()

    async def execute_async(self, task: str) -> GraphResult:
        """Execute the graph asynchronously."""
        logger.debug("task=<%s> | starting graph execution", task)

        # Initialize state
        self.state = GraphState(
            status=Status.EXECUTING,
            task=task,
            total_nodes=len(self.nodes),
            edges=[(edge.from_node, edge.to_node) for edge in self.edges],
            entry_points=list(self.entry_points),
        )

        start_time = time.time()
        try:
            await self._execute_graph()
            self.state.status = Status.COMPLETED
            logger.debug("status=<%s> | graph execution completed", self.state.status)

        except Exception:
            logger.exception("graph execution failed")
            self.state.status = Status.FAILED
            raise
        finally:
            self.state.execution_time = round((time.time() - start_time) * 1000)

        return self._build_result()

    async def _execute_graph(self) -> None:
        """Unified execution flow with conditional routing."""
        ready_nodes = list(self.entry_points)

        while ready_nodes:
            current_batch = ready_nodes.copy()
            ready_nodes.clear()

            # Execute current batch of ready nodes
            for node in current_batch:
                if node not in self.state.completed_nodes:
                    await self._execute_node(node)

                    # Find newly ready nodes after this execution
                    ready_nodes.extend(self._find_newly_ready_nodes())

    def _find_newly_ready_nodes(self) -> list["GraphNode"]:
        """Find nodes that became ready after the last execution."""
        newly_ready = []
        for _node_id, node in self.nodes.items():
            if (
                node not in self.state.completed_nodes
                and node not in self.state.failed_nodes
                and self._is_node_ready_with_conditions(node)
            ):
                newly_ready.append(node)
        return newly_ready

    def _is_node_ready_with_conditions(self, node: GraphNode) -> bool:
        """Check if a node is ready considering conditional edges."""
        # Get incoming edges to this node
        incoming_edges = [edge for edge in self.edges if edge.to_node == node]

        if not incoming_edges:
            return node in self.entry_points

        # Check if at least one incoming edge condition is satisfied
        for edge in incoming_edges:
            if edge.from_node in self.state.completed_nodes:
                if edge.should_traverse(self.state):
                    logger.debug(
                        "from=<%s>, to=<%s> | edge ready via satisfied condition", edge.from_node.node_id, node.node_id
                    )
                    return True
                else:
                    logger.debug(
                        "from=<%s>, to=<%s> | edge condition not satisfied", edge.from_node.node_id, node.node_id
                    )
        return False

    async def _execute_node(self, node: GraphNode) -> None:
        """Execute a single node with error handling."""
        node.status = Status.EXECUTING
        logger.debug("node_id=<%s> | executing node", node.node_id)

        start_time = time.time()
        try:
            # Build node input from satisfied dependencies
            node_input = self._build_node_input(node)

            # Execute based on node type and create unified NodeResult
            if isinstance(node.executor, MultiAgentBase):
                multi_agent_result = await node.executor.execute_async(node_input)

                # Create NodeResult with MultiAgentResult directly
                node_result = NodeResult(
                    result=multi_agent_result,  # type is MultiAgentResult
                    execution_time=multi_agent_result.execution_time,
                    status=Status.COMPLETED,
                    accumulated_usage=multi_agent_result.accumulated_usage,
                    accumulated_metrics=multi_agent_result.accumulated_metrics,
                    execution_count=multi_agent_result.execution_count,
                )

            elif isinstance(node.executor, Agent):
                agent_response = None  # Initialize with None to handle case where no result is yielded
                async for event in node.executor.stream_async(node_input):
                    if "result" in event:
                        agent_response = event["result"]

                if not agent_response:
                    raise ValueError(f"Node '{node.node_id}' did not return a result")

                # Extract metrics from agent response
                usage = Usage(inputTokens=0, outputTokens=0, totalTokens=0)
                metrics = Metrics(latencyMs=0)
                if hasattr(agent_response, "metrics") and agent_response.metrics:
                    if hasattr(agent_response.metrics, "accumulated_usage"):
                        usage = agent_response.metrics.accumulated_usage
                    if hasattr(agent_response.metrics, "accumulated_metrics"):
                        metrics = agent_response.metrics.accumulated_metrics

                node_result = NodeResult(
                    result=agent_response,  # type is AgentResult
                    execution_time=round((time.time() - start_time) * 1000),
                    status=Status.COMPLETED,
                    accumulated_usage=usage,
                    accumulated_metrics=metrics,
                    execution_count=1,
                )
            else:
                raise ValueError(f"Node '{node.node_id}' of type '{type(node.executor)}' is not supported")

            # Mark as completed
            node.status = Status.COMPLETED
            node.result = node_result
            node.execution_time = node_result.execution_time
            self.state.completed_nodes.add(node)
            self.state.results[node.node_id] = node_result
            self.state.execution_order.append(node)

            # Accumulate metrics
            self._accumulate_metrics(node_result)

            logger.debug(
                "node_id=<%s>, execution_time=<%dms> | node completed successfully", node.node_id, node.execution_time
            )

        except Exception as e:
            logger.error("node_id=<%s>, error=<%s> | node failed", node.node_id, e)
            execution_time = round((time.time() - start_time) * 1000)

            # Create a NodeResult for the failed node
            node_result = NodeResult(
                result=e,  # Store exception as result
                execution_time=execution_time,
                status=Status.FAILED,
                accumulated_usage=Usage(inputTokens=0, outputTokens=0, totalTokens=0),
                accumulated_metrics=Metrics(latencyMs=execution_time),
                execution_count=1,
            )

            node.status = Status.FAILED
            node.result = node_result
            node.execution_time = execution_time
            self.state.failed_nodes.add(node)
            self.state.results[node.node_id] = node_result  # Store in results for consistency

            raise

    def _accumulate_metrics(self, node_result: NodeResult) -> None:
        """Accumulate metrics from a node result."""
        self.state.accumulated_usage["inputTokens"] += node_result.accumulated_usage.get("inputTokens", 0)
        self.state.accumulated_usage["outputTokens"] += node_result.accumulated_usage.get("outputTokens", 0)
        self.state.accumulated_usage["totalTokens"] += node_result.accumulated_usage.get("totalTokens", 0)
        self.state.accumulated_metrics["latencyMs"] += node_result.accumulated_metrics.get("latencyMs", 0)
        self.state.execution_count += node_result.execution_count

    def _build_node_input(self, node: GraphNode) -> str:
        """Build input text for a node based on dependency outputs."""
        # Get satisfied dependencies
        dependency_results = {}
        for edge in self.edges:
            if (
                edge.to_node == node
                and edge.from_node in self.state.completed_nodes
                and edge.from_node.node_id in self.state.results
            ):
                if edge.should_traverse(self.state):
                    dependency_results[edge.from_node.node_id] = self.state.results[edge.from_node.node_id]

        if not dependency_results:
            return self.state.task

        # Combine task with dependency outputs
        input_parts = [f"Original Task: {self.state.task}", "\nInputs from previous nodes:"]

        for dep_id, node_result in dependency_results.items():
            input_parts.append(f"\nFrom {dep_id}:")
            # Get all agent results from this node (flattened if nested)
            agent_results = node_result.get_agent_results()
            for result in agent_results:
                agent_name = getattr(result, "agent_name", "Agent")
                result_text = self._extract_result_text(result)
                input_parts.append(f"  - {agent_name}: {result_text}")

        return "\n".join(input_parts)

    def _extract_result_text(self, result: AgentResult) -> str:
        """Extract text content from an agent result."""
        if hasattr(result, "message") and result.message:
            message = result.message
            if isinstance(message, dict) and "content" in message:
                content = message["content"]
                if isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                    return "\n".join(texts)
        else:
            return str(result)

    def _build_result(self) -> GraphResult:
        """Build graph result from current state."""
        return GraphResult(
            results=self.state.results,
            accumulated_usage=self.state.accumulated_usage,
            accumulated_metrics=self.state.accumulated_metrics,
            execution_count=self.state.execution_count,
            execution_time=self.state.execution_time,
            status=self.state.status,
            total_nodes=self.state.total_nodes,
            completed_nodes=len(self.state.completed_nodes),
            failed_nodes=len(self.state.failed_nodes),
            execution_order=self.state.execution_order,
            edges=self.state.edges,
            entry_points=self.state.entry_points,
        )
