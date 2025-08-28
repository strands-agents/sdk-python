"""Directed Graph Multi-Agent Pattern Implementation.

This module provides a deterministic graph-based agent orchestration system where
agents or MultiAgentBase instances (like Swarm or Graph) are nodes in a graph,
executed according to edge dependencies, with output from one node passed as input
to connected nodes.

Key Features:
- Agents and MultiAgentBase instances (Swarm, Graph, etc.) as graph nodes
- Deterministic execution based on dependency resolution
- Output propagation along edges
- Support for cyclic graphs (feedback loops)
- Clear dependency management
- Supports nested graphs (Graph as a node in another Graph)
"""

import asyncio
import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

from opentelemetry import trace as trace_api

from ..agent import Agent
from ..agent.state import AgentState
from ..telemetry import get_tracer
from ..types.content import ContentBlock, Messages
from ..types.event_loop import Metrics, Usage
from .base import MultiAgentBase, MultiAgentResult, NodeResult, Status

logger = logging.getLogger(__name__)


@dataclass
class GraphState:
    """Graph execution state.

    Attributes:
        status: Current execution status of the graph.
        completed_nodes: Set of nodes that have completed execution.
        failed_nodes: Set of nodes that failed during execution.
        execution_order: List of nodes in the order they were executed.
        task: The original input prompt/query provided to the graph execution.
              This represents the actual work to be performed by the graph as a whole.
              Entry point nodes receive this task as their input if they have no dependencies.
    """

    # Task (with default empty string)
    task: str | list[ContentBlock] = ""

    # Execution state
    status: Status = Status.PENDING
    completed_nodes: set["GraphNode"] = field(default_factory=set)
    failed_nodes: set["GraphNode"] = field(default_factory=set)
    execution_order: list["GraphNode"] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

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

    def should_continue(
        self,
        max_node_executions: Optional[int],
        execution_timeout: Optional[float],
    ) -> Tuple[bool, str]:
        """Check if the graph should continue execution.

        Returns: (should_continue, reason)
        """
        # Check node execution limit (only if set)
        if max_node_executions is not None and len(self.execution_order) >= max_node_executions:
            return False, f"Max node executions reached: {max_node_executions}"

        # Check timeout (only if set)
        if execution_timeout is not None:
            elapsed = time.time() - self.start_time
            if elapsed > execution_timeout:
                return False, f"Execution timed out: {execution_timeout}s"

        return True, "Continuing"


@dataclass
class GraphResult(MultiAgentResult):
    """Result from graph execution - extends MultiAgentResult with graph-specific details."""

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
    """Represents a node in the graph.

    The execution_status tracks the node's lifecycle within graph orchestration:
    - PENDING: Node hasn't started executing yet
    - EXECUTING: Node is currently running
    - COMPLETED/FAILED: Node finished executing (regardless of result quality)
    """

    node_id: str
    executor: Agent | MultiAgentBase
    dependencies: set["GraphNode"] = field(default_factory=set)
    execution_status: Status = Status.PENDING
    result: NodeResult | None = None
    execution_time: int = 0
    _initial_messages: Messages = field(default_factory=list, init=False)
    _initial_state: AgentState = field(default_factory=AgentState, init=False)

    def __post_init__(self) -> None:
        """Capture initial executor state after initialization."""
        # Deep copy the initial messages and state to preserve them
        if hasattr(self.executor, "messages"):
            self._initial_messages = copy.deepcopy(self.executor.messages)

        if hasattr(self.executor, "state") and hasattr(self.executor.state, "get"):
            self._initial_state = AgentState(self.executor.state.get())

    def reset_executor_state(self) -> None:
        """Reset GraphNode executor state to initial state when graph was created.

        This is useful when nodes are executed multiple times and need to start
        fresh on each execution, providing stateless behavior.
        """
        if hasattr(self.executor, "messages"):
            self.executor.messages = copy.deepcopy(self._initial_messages)

        if hasattr(self.executor, "state"):
            self.executor.state = AgentState(self._initial_state.get())

        # Reset execution status
        self.execution_status = Status.PENDING
        self.result = None

    def __hash__(self) -> int:
        """Return hash for GraphNode based on node_id."""
        return hash(self.node_id)

    def __eq__(self, other: Any) -> bool:
        """Return equality for GraphNode based on node_id."""
        if not isinstance(other, GraphNode):
            return False
        return self.node_id == other.node_id


def _validate_node_executor(
    executor: Agent | MultiAgentBase, existing_nodes: dict[str, GraphNode] | None = None
) -> None:
    """Validate a node executor for graph compatibility.

    Args:
        executor: The executor to validate
        existing_nodes: Optional dict of existing nodes to check for duplicates
    """
    # Check for duplicate node instances
    if existing_nodes:
        seen_instances = {id(node.executor) for node in existing_nodes.values()}
        if id(executor) in seen_instances:
            raise ValueError("Duplicate node instance detected. Each node must have a unique object instance.")

    # Validate Agent-specific constraints
    if isinstance(executor, Agent):
        # Check for session persistence
        if executor._session_manager is not None:
            raise ValueError("Session persistence is not supported for Graph agents yet.")


class GraphBuilder:
    """Builder pattern for constructing graphs."""

    def __init__(self) -> None:
        """Initialize GraphBuilder with empty collections."""
        self.nodes: dict[str, GraphNode] = {}
        self.edges: set[GraphEdge] = set()
        self.entry_points: set[GraphNode] = set()

        # Configuration options
        self._max_node_executions: Optional[int] = None
        self._execution_timeout: Optional[float] = None
        self._node_timeout: Optional[float] = None
        self._reset_on_revisit: bool = False

    def add_node(self, executor: Agent | MultiAgentBase, node_id: str | None = None) -> GraphNode:
        """Add an Agent or MultiAgentBase instance as a node to the graph."""
        _validate_node_executor(executor, self.nodes)

        # Auto-generate node_id if not provided
        if node_id is None:
            node_id = getattr(executor, "id", None) or getattr(executor, "name", None) or f"node_{len(self.nodes)}"

        if node_id in self.nodes:
            raise ValueError(f"Node '{node_id}' already exists")

        node = GraphNode(node_id=node_id, executor=executor)
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        from_node: str | GraphNode,
        to_node: str | GraphNode,
        condition: Callable[[GraphState], bool] | None = None,
    ) -> GraphEdge:
        """Add an edge between two nodes with optional condition function that receives full GraphState."""

        def resolve_node(node: str | GraphNode, node_type: str) -> GraphNode:
            if isinstance(node, str):
                if node not in self.nodes:
                    raise ValueError(f"{node_type} node '{node}' not found")
                return self.nodes[node]
            else:
                if node not in self.nodes.values():
                    raise ValueError(f"{node_type} node object has not been added to the graph, use graph.add_node")
                return node

        from_node_obj = resolve_node(from_node, "Source")
        to_node_obj = resolve_node(to_node, "Target")

        # Add edge and update dependencies
        edge = GraphEdge(from_node=from_node_obj, to_node=to_node_obj, condition=condition)
        self.edges.add(edge)
        to_node_obj.dependencies.add(from_node_obj)
        return edge

    def set_entry_point(self, node_id: str) -> "GraphBuilder":
        """Set a node as an entry point for graph execution."""
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' not found")
        self.entry_points.add(self.nodes[node_id])
        return self

    def reset_on_revisit(self, enabled: bool = True) -> "GraphBuilder":
        """Control whether nodes reset their state when revisited.

        When enabled, nodes will reset their messages and state to initial values
        each time they are revisited (re-executed). This is useful for stateless
        behavior where nodes should start fresh on each revisit.

        Args:
            enabled: Whether to reset node state when revisited (default: True)
        """
        self._reset_on_revisit = enabled
        return self

    def set_max_node_executions(self, max_executions: int) -> "GraphBuilder":
        """Set maximum number of node executions allowed.

        Args:
            max_executions: Maximum total node executions (None for no limit)
        """
        self._max_node_executions = max_executions
        return self

    def set_execution_timeout(self, timeout: float) -> "GraphBuilder":
        """Set total execution timeout.

        Args:
            timeout: Total execution timeout in seconds (None for no limit)
        """
        self._execution_timeout = timeout
        return self

    def set_node_timeout(self, timeout: float) -> "GraphBuilder":
        """Set individual node execution timeout.

        Args:
            timeout: Individual node timeout in seconds (None for no limit)
        """
        self._node_timeout = timeout
        return self

    def build(self) -> "Graph":
        """Build and validate the graph with configured settings."""
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

        return Graph(
            nodes=self.nodes.copy(),
            edges=self.edges.copy(),
            entry_points=self.entry_points.copy(),
            max_node_executions=self._max_node_executions,
            execution_timeout=self._execution_timeout,
            node_timeout=self._node_timeout,
            reset_on_revisit=self._reset_on_revisit,
        )

    def _validate_graph(self) -> None:
        """Validate graph structure."""
        # Validate entry points exist
        entry_point_ids = {node.node_id for node in self.entry_points}
        invalid_entries = entry_point_ids - set(self.nodes.keys())
        if invalid_entries:
            raise ValueError(f"Entry points not found in nodes: {invalid_entries}")

        # Warn about potential infinite loops if no execution limits are set
        if self._max_node_executions is None and self._execution_timeout is None:
            logger.warning("Graph without execution limits may run indefinitely if cycles exist")


class Graph(MultiAgentBase):
    """Directed Graph multi-agent orchestration with configurable revisit behavior."""

    def __init__(
        self,
        nodes: dict[str, GraphNode],
        edges: set[GraphEdge],
        entry_points: set[GraphNode],
        max_node_executions: Optional[int] = None,
        execution_timeout: Optional[float] = None,
        node_timeout: Optional[float] = None,
        reset_on_revisit: bool = False,
    ) -> None:
        """Initialize Graph with execution limits and reset behavior.

        Args:
            nodes: Dictionary of node_id to GraphNode
            edges: Set of GraphEdge objects
            entry_points: Set of GraphNode objects that are entry points
            max_node_executions: Maximum total node executions (default: None - no limit)
            execution_timeout: Total execution timeout in seconds (default: None - no limit)
            node_timeout: Individual node timeout in seconds (default: None - no limit)
            reset_on_revisit: Whether to reset node state when revisited (default: False)
        """
        super().__init__()

        # Validate nodes for duplicate instances
        self._validate_graph(nodes)

        self.nodes = nodes
        self.edges = edges
        self.entry_points = entry_points
        self.max_node_executions = max_node_executions
        self.execution_timeout = execution_timeout
        self.node_timeout = node_timeout
        self.reset_on_revisit = reset_on_revisit
        self.state = GraphState()
        self.tracer = get_tracer()

    def __call__(self, task: str | list[ContentBlock], **kwargs: Any) -> GraphResult:
        """Invoke the graph synchronously."""

        def execute() -> GraphResult:
            return asyncio.run(self.invoke_async(task))

        with ThreadPoolExecutor() as executor:
            future = executor.submit(execute)
            return future.result()

    async def invoke_async(self, task: str | list[ContentBlock], **kwargs: Any) -> GraphResult:
        """Invoke the graph asynchronously."""
        logger.debug("task=<%s> | starting graph execution", task)

        # Initialize state
        start_time = time.time()
        self.state = GraphState(
            status=Status.EXECUTING,
            task=task,
            total_nodes=len(self.nodes),
            edges=[(edge.from_node, edge.to_node) for edge in self.edges],
            entry_points=list(self.entry_points),
            start_time=start_time,
        )

        span = self.tracer.start_multiagent_span(task, "graph")
        with trace_api.use_span(span, end_on_exit=True):
            try:
                logger.debug(
                    "max_node_executions=<%s>, execution_timeout=<%s>s, node_timeout=<%s>s | graph execution config",
                    self.max_node_executions or "None",
                    self.execution_timeout or "None",
                    self.node_timeout or "None",
                )

                await self._execute_graph()

                # Set final status based on execution results
                if self.state.failed_nodes:
                    self.state.status = Status.FAILED
                elif self.state.status == Status.EXECUTING:  # Only set to COMPLETED if still executing and no failures
                    self.state.status = Status.COMPLETED

                logger.debug("status=<%s> | graph execution completed", self.state.status)

            except Exception:
                logger.exception("graph execution failed")
                self.state.status = Status.FAILED
                raise
            finally:
                self.state.execution_time = round((time.time() - start_time) * 1000)
            return self._build_result()

    def _validate_graph(self, nodes: dict[str, GraphNode]) -> None:
        """Validate graph nodes for duplicate instances."""
        # Check for duplicate node instances
        seen_instances = set()
        for node in nodes.values():
            if id(node.executor) in seen_instances:
                raise ValueError("Duplicate node instance detected. Each node must have a unique object instance.")
            seen_instances.add(id(node.executor))

            # Validate Agent-specific constraints for each node
            _validate_node_executor(node.executor)

    async def _execute_graph(self) -> None:
        """Unified execution flow with conditional routing."""
        ready_nodes = list(self.entry_points)

        while ready_nodes:
            # Check execution limits before continuing
            should_continue, reason = self.state.should_continue(
                max_node_executions=self.max_node_executions,
                execution_timeout=self.execution_timeout,
            )
            if not should_continue:
                self.state.status = Status.FAILED
                logger.debug("reason=<%s> | stopping execution", reason)
                return  # Let the top-level exception handler deal with it

            current_batch = ready_nodes.copy()
            ready_nodes.clear()

            # Execute current batch of ready nodes concurrently
            tasks = [
                asyncio.create_task(self._execute_node(node))
                for node in current_batch
                if node not in self.state.completed_nodes
            ]

            for task in tasks:
                await task

            # Find newly ready nodes after batch execution
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
        """Execute a single node with error handling and timeout protection."""
        # Reset the node's state if reset_on_revisit is enabled and it's being revisited
        if self.reset_on_revisit and node in self.state.completed_nodes:
            logger.debug("node_id=<%s> | resetting node state for revisit", node.node_id)
            node.reset_executor_state()
            # Remove from completed nodes since we're re-executing it
            self.state.completed_nodes.remove(node)

        node.execution_status = Status.EXECUTING
        logger.debug("node_id=<%s> | executing node", node.node_id)

        start_time = time.time()
        try:
            # Build node input from satisfied dependencies
            node_input = self._build_node_input(node)

            # Execute with timeout protection (only if node_timeout is set)
            try:
                # Execute based on node type and create unified NodeResult
                if isinstance(node.executor, MultiAgentBase):
                    if self.node_timeout is not None:
                        multi_agent_result = await asyncio.wait_for(
                            node.executor.invoke_async(node_input),
                            timeout=self.node_timeout,
                        )
                    else:
                        multi_agent_result = await node.executor.invoke_async(node_input)

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
                    if self.node_timeout is not None:
                        agent_response = await asyncio.wait_for(
                            node.executor.invoke_async(node_input),
                            timeout=self.node_timeout,
                        )
                    else:
                        agent_response = await node.executor.invoke_async(node_input)

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

            except asyncio.TimeoutError:
                timeout_msg = f"Node '{node.node_id}' execution timed out after {self.node_timeout}s"
                logger.exception(
                    "node=<%s>, timeout=<%s>s | node execution timed out after timeout",
                    node.node_id,
                    self.node_timeout,
                )
                raise Exception(timeout_msg) from None

            # Mark as completed
            node.execution_status = Status.COMPLETED
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

            node.execution_status = Status.FAILED
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

    def _build_node_input(self, node: GraphNode) -> list[ContentBlock]:
        """Build input text for a node based on dependency outputs.

        Example formatted output:
        ```
        Original Task: Analyze the quarterly sales data and create a summary report

        Inputs from previous nodes:

        From data_processor:
          - Agent: Sales data processed successfully. Found 1,247 transactions totaling $89,432.
          - Agent: Key trends: 15% increase in Q3, top product category is Electronics.

        From validator:
          - Agent: Data validation complete. All records verified, no anomalies detected.
        ```
        """
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
            # No dependencies - return task as ContentBlocks
            if isinstance(self.state.task, str):
                return [ContentBlock(text=self.state.task)]
            else:
                return self.state.task

        # Combine task with dependency outputs
        node_input = []

        # Add original task
        if isinstance(self.state.task, str):
            node_input.append(ContentBlock(text=f"Original Task: {self.state.task}"))
        else:
            # Add task content blocks with a prefix
            node_input.append(ContentBlock(text="Original Task:"))
            node_input.extend(self.state.task)

        # Add dependency outputs
        node_input.append(ContentBlock(text="\nInputs from previous nodes:"))

        for dep_id, node_result in dependency_results.items():
            node_input.append(ContentBlock(text=f"\nFrom {dep_id}:"))
            # Get all agent results from this node (flattened if nested)
            agent_results = node_result.get_agent_results()
            for result in agent_results:
                agent_name = getattr(result, "agent_name", "Agent")
                result_text = str(result)
                node_input.append(ContentBlock(text=f"  - {agent_name}: {result_text}"))

        return node_input

    def _build_result(self) -> GraphResult:
        """Build graph result from current state."""
        return GraphResult(
            status=self.state.status,
            results=self.state.results,
            accumulated_usage=self.state.accumulated_usage,
            accumulated_metrics=self.state.accumulated_metrics,
            execution_count=self.state.execution_count,
            execution_time=self.state.execution_time,
            total_nodes=self.state.total_nodes,
            completed_nodes=len(self.state.completed_nodes),
            failed_nodes=len(self.state.failed_nodes),
            execution_order=self.state.execution_order,
            edges=self.state.edges,
            entry_points=self.state.entry_points,
        )
