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

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from ..agent import Agent, AgentResult
from ..types.event_loop import Metrics, Usage
from .base import MultiAgentBase, MultiAgentResult, NodeResult

logger = logging.getLogger(__name__)


class Status(Enum):
    """Execution status for both graphs and nodes."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GraphState:
    """Graph execution state."""

    # Execution state
    status: Status = Status.PENDING
    completed_nodes: Set[str] = field(default_factory=set)
    failed_nodes: Set[str] = field(default_factory=set)
    execution_order: List[str] = field(default_factory=list)
    task: str = ""

    # Results
    results: Dict[str, NodeResult] = field(default_factory=dict)

    # Accumulated metrics
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_count: int = 0
    execution_time: float = 0.0

    # Graph structure info
    total_nodes: int = 0
    edges: List[Tuple[str, str]] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)


@dataclass
class GraphResult(MultiAgentResult):
    """Result from graph execution - extends MultiAgentResult with graph-specific details."""

    status: Status = Status.PENDING
    total_nodes: int = 0
    completed_nodes: int = 0
    failed_nodes: int = 0
    execution_order: List[str] = field(default_factory=list)
    edges: List[Tuple[str, str]] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)


@dataclass
class GraphEdge:
    """Represents an edge in the graph with optional condition."""

    from_node: str
    to_node: str
    condition: Optional[Callable[[GraphState], bool]] = None

    def __hash__(self) -> int:
        """Return hash for GraphEdge based on from_node and to_node."""
        return hash((self.from_node, self.to_node))

    def should_traverse(self, state: GraphState) -> bool:
        """Check if this edge should be traversed based on condition."""
        if self.condition is None:
            return True
        return self.condition(state)


@dataclass
class GraphNode:
    """Represents a node in the graph."""

    node_id: str
    executor: Union[Agent, MultiAgentBase]
    dependencies: Set[str] = field(default_factory=set)
    status: Status = Status.PENDING
    result: Optional[NodeResult] = None
    error: Optional[Exception] = None
    execution_time: float = 0.0

    def is_ready(self, completed_nodes: Set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return self.dependencies.issubset(completed_nodes)


class GraphBuilder:
    """Builder pattern for constructing graphs."""

    def __init__(self) -> None:
        """Initialize GraphBuilder with empty collections."""
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: Set[GraphEdge] = set()
        self.entry_points: Set[str] = set()

    def add_node(self, executor: Union[Agent, MultiAgentBase], node_id: Optional[str] = None) -> "GraphBuilder":
        """Add an Agent or MultiAgentBase instance as a node to the graph."""
        # Auto-generate node_id if not provided
        if node_id is None:
            node_id = getattr(executor, "id", None) or getattr(executor, "name", None) or f"node_{len(self.nodes)}"

        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists")

        self.nodes[node_id] = GraphNode(node_id=node_id, executor=executor)
        return self

    def add_edge(
        self, from_node: str, to_node: str, condition: Optional[Callable[[GraphState], bool]] = None
    ) -> "GraphBuilder":
        """Add an edge between two nodes with optional condition function that receives full GraphState."""
        # Validate nodes exist
        for node_name, node_id in [("Source", from_node), ("Target", to_node)]:
            if node_id not in self.nodes:
                raise ValueError(f"{node_name} node '{node_id}' not found")

        # Add edge and update dependencies
        self.edges.add(GraphEdge(from_node=from_node, to_node=to_node, condition=condition))
        self.nodes[to_node].dependencies.add(from_node)
        return self

    def set_entry_point(self, node_id: str) -> "GraphBuilder":
        """Set a node as an entry point for graph execution."""
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' not found")
        self.entry_points.add(node_id)
        return self

    def build(self) -> "Graph":
        """Build and validate the graph."""
        if not self.nodes:
            raise ValueError("Graph must contain at least one node")

        # Auto-detect entry points if none specified
        if not self.entry_points:
            self.entry_points = {node_id for node_id, node in self.nodes.items() if not node.dependencies}
            logger.debug("entry_points=<%s> | auto-detected entrypoints", ", ".join(self.entry_points))
            if not self.entry_points:
                raise ValueError("No entry points found - all nodes have dependencies")

        # Validate entry points and check for cycles
        self._validate_graph()

        return Graph(nodes=self.nodes.copy(), edges=self.edges.copy(), entry_points=self.entry_points.copy())

    def _validate_graph(self) -> None:
        """Validate graph structure and detect cycles."""
        # Validate entry points exist
        invalid_entries = self.entry_points - set(self.nodes.keys())
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
                if edge.from_node == node_id and has_cycle_from(edge.to_node):
                    return True
            colors[node_id] = BLACK
            return False

        # Check for cycles from each unvisited node
        if any(colors[node_id] == WHITE and has_cycle_from(node_id) for node_id in self.nodes):
            raise ValueError("Graph contains cycles - must be a directed acyclic graph")


class Graph(MultiAgentBase):
    """Directed Acyclic Graph multi-agent orchestration."""

    def __init__(self, nodes: Dict[str, GraphNode], edges: Set[GraphEdge], entry_points: Set[str]) -> None:
        """Initialize Graph."""
        super().__init__()

        self.nodes = nodes
        self.edges = edges
        self.entry_points = entry_points
        self.state = GraphState()

    async def execute(self, task: str) -> GraphResult:
        """Execute the graph."""
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
            if self.state.status == Status.EXECUTING:
                self.state.status = Status.COMPLETED
            logger.debug("status=<%s> | graph execution completed", self.state.status)

        except Exception as e:
            logger.error("error=<%s> | graph execution failed", e)
            self.state.status = Status.FAILED
            raise
        finally:
            self.state.execution_time = round((time.time() - start_time) * 1000)

        return self._build_result()

    async def resume(self, task: str, state: GraphState) -> MultiAgentResult:
        """Resume graph from previous state."""
        raise NotImplementedError("resume not implemented")  # TODO

    async def _execute_graph(self) -> None:
        """Unified execution flow with conditional routing."""
        ready_nodes = list(self.entry_points)

        while ready_nodes:
            current_batch = ready_nodes.copy()
            ready_nodes.clear()

            # Execute current batch of ready nodes
            for node_id in current_batch:
                if node_id not in self.state.completed_nodes:
                    await self._execute_node(node_id)

                    # Find newly ready nodes after this execution
                    ready_nodes.extend(self._find_newly_ready_nodes())

    def _find_newly_ready_nodes(self) -> List[str]:
        """Find nodes that became ready after the last execution."""
        newly_ready = []
        for node_id, _node in self.nodes.items():
            if (
                node_id not in self.state.completed_nodes
                and node_id not in self.state.failed_nodes
                and self._is_node_ready_with_conditions(node_id)
            ):
                newly_ready.append(node_id)
        return newly_ready

    def _is_node_ready_with_conditions(self, node_id: str) -> bool:
        """Check if a node is ready considering conditional edges."""
        # Get incoming edges to this node
        incoming_edges = [edge for edge in self.edges if edge.to_node == node_id]

        if not incoming_edges:
            return node_id in self.entry_points

        # Check if at least one incoming edge condition is satisfied
        for edge in incoming_edges:
            if edge.from_node in self.state.completed_nodes:
                if edge.should_traverse(self.state):
                    logger.debug("from=<%s>, to=<%s> | edge ready via satisfied condition", edge.from_node, node_id)
                    return True
                else:
                    logger.debug("from=<%s>, to=<%s> | edge condition not satisfied", edge.from_node, node_id)
        return False

    async def _execute_node(self, node_id: str) -> None:
        """Execute a single node with error handling."""
        node = self.nodes[node_id]
        node.status = Status.EXECUTING
        logger.debug("node_id=<%s> | executing node", node_id)

        start_time = time.time()
        try:
            # Build node input from satisfied dependencies
            node_input = self._build_node_input(node_id)

            # Execute based on node type and create unified NodeResult
            if isinstance(node.executor, MultiAgentBase):
                multi_agent_result = await node.executor.execute(node_input)

                # Create NodeResult with MultiAgentResult directly
                node_result = NodeResult(
                    results=multi_agent_result,  # MultiAgentResult
                    execution_time=multi_agent_result.execution_time,
                    status=Status.COMPLETED,
                    accumulated_usage=multi_agent_result.accumulated_usage,
                    accumulated_metrics=multi_agent_result.accumulated_metrics,
                    execution_count=multi_agent_result.execution_count,
                )

            elif isinstance(node.executor, Agent):
                agent_response = node.executor(node_input)

                # Extract metrics from agent response
                usage = Usage(inputTokens=0, outputTokens=0, totalTokens=0)
                metrics = Metrics(latencyMs=0)
                if hasattr(agent_response, "metrics") and agent_response.metrics:
                    if hasattr(agent_response.metrics, "accumulated_usage"):
                        usage = agent_response.metrics.accumulated_usage
                    if hasattr(agent_response.metrics, "accumulated_metrics"):
                        metrics = agent_response.metrics.accumulated_metrics

                node_result = NodeResult(
                    results=agent_response,  # Single AgentResult
                    execution_time=round((time.time() - start_time) * 1000),
                    status=Status.COMPLETED,
                    accumulated_usage=usage,
                    accumulated_metrics=metrics,
                    execution_count=1,
                )
            else:
                raise ValueError(f"Node '{node_id}' of type '{type(node.executor)}' is not supported")

            # Mark as completed
            node.status = Status.COMPLETED
            node.result = node_result
            node.execution_time = node_result.execution_time
            self.state.completed_nodes.add(node_id)
            self.state.results[node_id] = node_result
            self.state.execution_order.append(node_id)

            # Accumulate metrics
            self._accumulate_metrics(node_result)

            logger.debug(
                "node_id=<%s>, execution_time=<%dms> | node completed successfully", node_id, node.execution_time
            )

        except Exception as e:
            logger.error("node_id=<%s>, error=<%s> | node failed", node_id, e)
            node.status = Status.FAILED
            node.error = e
            self.state.failed_nodes.add(node_id)
            raise

    def _accumulate_metrics(self, node_result: NodeResult) -> None:
        """Accumulate metrics from a node result."""
        self.state.accumulated_usage["inputTokens"] += node_result.accumulated_usage.get("inputTokens", 0)
        self.state.accumulated_usage["outputTokens"] += node_result.accumulated_usage.get("outputTokens", 0)
        self.state.accumulated_usage["totalTokens"] += node_result.accumulated_usage.get("totalTokens", 0)
        self.state.accumulated_metrics["latencyMs"] += node_result.accumulated_metrics.get("latencyMs", 0)
        self.state.execution_count += node_result.execution_count

    def _build_node_input(self, node_id: str) -> str:
        """Build input text for a node based on dependency outputs."""
        # Get satisfied dependencies
        dependency_results = {}
        for edge in self.edges:
            if (
                edge.to_node == node_id
                and edge.from_node in self.state.completed_nodes
                and edge.from_node in self.state.results
            ):
                if edge.should_traverse(self.state):
                    dependency_results[edge.from_node] = self.state.results[edge.from_node]

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

    def __str__(self) -> str:
        """Create a simple text visualization of the graph."""
        lines = [f"Nodes ({len(self.nodes)}):"]

        for node_id, node in self.nodes.items():
            node_type = type(node.executor).__name__
            status_info = f" [{node.status.value}]" if node.status != Status.PENDING else ""
            lines.append(f"  {node_id} ({node_type}){status_info}")

            # Show nested structure for Graph nodes
            if isinstance(node.executor, Graph):
                sub_graph_viz = str(node.executor)
                for line in sub_graph_viz.split("\n"):
                    if line.strip():
                        lines.append(f"    └─ {line}")
            elif isinstance(node.executor, MultiAgentBase):
                # Try to show agents if available
                if hasattr(node.executor, "agents"):
                    for agent in node.executor.agents:
                        agent_name = getattr(agent, "name", "unknown")
                        lines.append(f"    └─ {agent_name} (Agent)")

        lines.append(f"Entry Points: {list(self.entry_points)}")
        lines.append("Edges:")
        for edge in sorted(self.edges, key=lambda e: (e.from_node, e.to_node)):
            condition_info = " [conditional]" if edge.condition is not None else ""
            lines.append(f"  {edge.from_node} -> {edge.to_node}{condition_info}")

        return "\n".join(lines)
