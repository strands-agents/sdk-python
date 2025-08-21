"""Swarm Multi-Agent Pattern Implementation.

This module provides a collaborative agent orchestration system where
agents work together as a team to solve complex tasks, with shared context
and autonomous coordination.

Key Features:
- Self-organizing agent teams with shared working memory
- Tool-based coordination
- Autonomous agent collaboration without central control
- Dynamic task distribution based on agent capabilities
- Collective intelligence through shared context
"""

import asyncio
import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Tuple

from opentelemetry import trace as trace_api

from ..agent import Agent, AgentResult
from ..agent.state import AgentState
from ..telemetry import get_tracer
from ..tools.decorator import tool
from ..types.content import ContentBlock, Messages
from ..types.event_loop import Metrics, Usage
from .base import MultiAgentBase, MultiAgentResult, NodeResult, SharedContext, Status, MultiAgentNode

logger = logging.getLogger(__name__)


@dataclass
class SwarmNode(MultiAgentNode):
    """Represents a node (e.g. Agent) in the swarm."""

    executor: Agent
    _initial_messages: Messages = field(default_factory=list, init=False)
    _initial_state: AgentState = field(default_factory=AgentState, init=False)

    def __post_init__(self) -> None:
        """Capture initial executor state after initialization."""
        # Deep copy the initial messages and state to preserve them
        self._initial_messages = copy.deepcopy(self.executor.messages)
        self._initial_state = AgentState(self.executor.state.get())

    def __hash__(self) -> int:
        """Return hash for SwarmNode based on node_id."""
        return hash(self.node_id)

    def __eq__(self, other: Any) -> bool:
        """Return equality for SwarmNode based on node_id."""
        if not isinstance(other, SwarmNode):
            return False
        return self.node_id == other.node_id

    def __str__(self) -> str:
        """Return string representation of SwarmNode."""
        return self.node_id

    def __repr__(self) -> str:
        """Return detailed representation of SwarmNode."""
        return f"SwarmNode(node_id='{self.node_id}')"

    def reset_executor_state(self) -> None:
        """Reset SwarmNode executor state to initial state when swarm was created."""
        self.executor.messages = copy.deepcopy(self._initial_messages)
        self.executor.state = AgentState(self._initial_state.get())


@dataclass
class SwarmState:
    """Current state of swarm execution."""

    current_node: SwarmNode  # The agent currently executing
    task: str | list[ContentBlock]  # The original task from the user that is being executed
    completion_status: Status = Status.PENDING  # Current swarm execution status
    shared_context: SharedContext = field(default_factory=SharedContext)  # Context shared between agents
    node_history: list[SwarmNode] = field(default_factory=list)  # Complete history of agents that have executed
    start_time: float = field(default_factory=time.time)  # When swarm execution began
    results: dict[str, NodeResult] = field(default_factory=dict)  # Results from each agent execution
    # Total token usage across all agents
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    # Total metrics across all agents
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_time: int = 0  # Total execution time in milliseconds
    handoff_message: str | None = None  # Message passed during agent handoff

    def should_continue(
        self,
        *,
        max_handoffs: int,
        max_iterations: int,
        execution_timeout: float,
        repetitive_handoff_detection_window: int,
        repetitive_handoff_min_unique_agents: int,
    ) -> Tuple[bool, str]:
        """Check if the swarm should continue.

        Returns: (should_continue, reason)
        """
        # Check handoff limit
        if len(self.node_history) >= max_handoffs:
            return False, f"Max handoffs reached: {max_handoffs}"

        # Check iteration limit
        if len(self.node_history) >= max_iterations:
            return False, f"Max iterations reached: {max_iterations}"

        # Check timeout
        elapsed = time.time() - self.start_time
        if elapsed > execution_timeout:
            return False, f"Execution timed out: {execution_timeout}s"

        # Check for repetitive handoffs (agents passing back and forth)
        if repetitive_handoff_detection_window > 0 and len(self.node_history) >= repetitive_handoff_detection_window:
            recent = self.node_history[-repetitive_handoff_detection_window:]
            unique_nodes = len(set(recent))
            if unique_nodes < repetitive_handoff_min_unique_agents:
                return (
                    False,
                    (
                        f"Repetitive handoff: {unique_nodes} unique nodes "
                        f"out of {repetitive_handoff_detection_window} recent iterations"
                    ),
                )

        return True, "Continuing"


@dataclass
class SwarmResult(MultiAgentResult):
    """Result from swarm execution - extends MultiAgentResult with swarm-specific details."""

    node_history: list[SwarmNode] = field(default_factory=list)


class Swarm(MultiAgentBase):
    """Self-organizing collaborative agent teams with shared working memory."""

    def __init__(
        self,
        nodes: list[Agent],
        *,
        max_handoffs: int = 20,
        max_iterations: int = 20,
        execution_timeout: float = 900.0,
        node_timeout: float = 300.0,
        repetitive_handoff_detection_window: int = 0,
        repetitive_handoff_min_unique_agents: int = 0,
    ) -> None:
        """Initialize Swarm with agents and configuration.

        Args:
            nodes: List of nodes (e.g. Agent) to include in the swarm
            max_handoffs: Maximum handoffs to agents and users (default: 20)
            max_iterations: Maximum node executions within the swarm (default: 20)
            execution_timeout: Total execution timeout in seconds (default: 900.0)
            node_timeout: Individual node timeout in seconds (default: 300.0)
            repetitive_handoff_detection_window: Number of recent nodes to check for repetitive handoffs
                Disabled by default (default: 0)
            repetitive_handoff_min_unique_agents: Minimum unique agents required in recent sequence
                Disabled by default (default: 0)
        """
        super().__init__()

        self.max_handoffs = max_handoffs
        self.max_iterations = max_iterations
        self.execution_timeout = execution_timeout
        self.node_timeout = node_timeout
        self.repetitive_handoff_detection_window = repetitive_handoff_detection_window
        self.repetitive_handoff_min_unique_agents = repetitive_handoff_min_unique_agents

        self.shared_context = SharedContext()
        self.nodes: dict[str, SwarmNode] = {}
        self.state = SwarmState(
            current_node=SwarmNode("", Agent()),  # Placeholder, will be set properly
            task="",
            completion_status=Status.PENDING,
        )
        self.tracer = get_tracer()

        self._setup_swarm(nodes)
        self._inject_swarm_tools()

    def __call__(self, task: str | list[ContentBlock], **kwargs: Any) -> SwarmResult:
        """Invoke the swarm synchronously."""

        def execute() -> SwarmResult:
            return asyncio.run(self.invoke_async(task))

        with ThreadPoolExecutor() as executor:
            future = executor.submit(execute)
            return future.result()

    async def invoke_async(self, task: str | list[ContentBlock], **kwargs: Any) -> SwarmResult:
        """Invoke the swarm asynchronously."""
        logger.debug("starting swarm execution")

        # Initialize swarm state with configuration
        initial_node = next(iter(self.nodes.values()))  # First SwarmNode
        self.state = SwarmState(
            current_node=initial_node,
            task=task,
            completion_status=Status.EXECUTING,
            shared_context=self.shared_context,
        )

        start_time = time.time()
        span = self.tracer.start_multiagent_span(task, "swarm")
        with trace_api.use_span(span, end_on_exit=True):
            try:
                logger.debug("current_node=<%s> | starting swarm execution with node", self.state.current_node.node_id)
                logger.debug(
                    "max_handoffs=<%d>, max_iterations=<%d>, timeout=<%s>s | swarm execution config",
                    self.max_handoffs,
                    self.max_iterations,
                    self.execution_timeout,
                )

                await self._execute_swarm()
            except Exception:
                logger.exception("swarm execution failed")
                self.state.completion_status = Status.FAILED
                raise
            finally:
                self.state.execution_time = round((time.time() - start_time) * 1000)

            return self._build_result()


# Backward compatibility aliases
# These ensure that existing imports continue to work
__all__ = ["SwarmNode", "SharedContext", "Status"]
