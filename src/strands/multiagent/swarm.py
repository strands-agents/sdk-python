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
import json
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
from .base import MultiAgentBase, MultiAgentResult, NodeResult, Status

logger = logging.getLogger(__name__)


@dataclass
class SwarmNode:
    """Represents a node (e.g. Agent) in the swarm."""

    node_id: str
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
class SharedContext:
    """Shared context between swarm nodes."""

    context: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_context(self, node: SwarmNode, key: str, value: Any) -> None:
        """Add context."""
        self._validate_key(key)
        self._validate_json_serializable(value)

        if node.node_id not in self.context:
            self.context[node.node_id] = {}
        self.context[node.node_id][key] = value

    def _validate_key(self, key: str) -> None:
        """Validate that a key is valid.

        Args:
            key: The key to validate

        Raises:
            ValueError: If key is invalid
        """
        if key is None:
            raise ValueError("Key cannot be None")
        if not isinstance(key, str):
            raise ValueError("Key must be a string")
        if not key.strip():
            raise ValueError("Key cannot be empty")

    def _validate_json_serializable(self, value: Any) -> None:
        """Validate that a value is JSON serializable.

        Args:
            value: The value to validate

        Raises:
            ValueError: If value is not JSON serializable
        """
        try:
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Value is not JSON serializable: {type(value).__name__}. "
                f"Only JSON-compatible types (str, int, float, bool, list, dict, None) are allowed."
            ) from e


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

    def _setup_swarm(self, nodes: list[Agent]) -> None:
        """Initialize swarm configuration."""
        # Validate nodes before setup
        self._validate_swarm(nodes)

        # Validate agents have names and create SwarmNode objects
        for i, node in enumerate(nodes):
            if not node.name:
                node_id = f"node_{i}"
                node.name = node_id
                logger.debug("node_id=<%s> | agent has no name, dynamically generating one", node_id)

            node_id = str(node.name)

            # Ensure node IDs are unique
            if node_id in self.nodes:
                raise ValueError(f"Node ID '{node_id}' is not unique. Each agent must have a unique name.")

            self.nodes[node_id] = SwarmNode(node_id=node_id, executor=node)

        swarm_nodes = list(self.nodes.values())
        logger.debug("nodes=<%s> | initialized swarm with nodes", [node.node_id for node in swarm_nodes])

    def _validate_swarm(self, nodes: list[Agent]) -> None:
        """Validate swarm structure and nodes."""
        # Check for duplicate object instances
        seen_instances = set()
        for node in nodes:
            if id(node) in seen_instances:
                raise ValueError("Duplicate node instance detected. Each node must have a unique object instance.")
            seen_instances.add(id(node))

            # Check for session persistence
            if node._session_manager is not None:
                raise ValueError("Session persistence is not supported for Swarm agents yet.")

    def _inject_swarm_tools(self) -> None:
        """Add swarm coordination tools to each agent."""
        # Create tool functions with proper closures
        swarm_tools = [
            self._create_handoff_tool(),
        ]

        for node in self.nodes.values():
            # Check for existing tools with conflicting names
            existing_tools = node.executor.tool_registry.registry
            conflicting_tools = []

            if "handoff_to_agent" in existing_tools:
                conflicting_tools.append("handoff_to_agent")

            if conflicting_tools:
                raise ValueError(
                    f"Agent '{node.node_id}' already has tools with names that conflict with swarm coordination tools: "
                    f"{', '.join(conflicting_tools)}. Please rename these tools to avoid conflicts."
                )

            # Use the agent's tool registry to process and register the tools
            node.executor.tool_registry.process_tools(swarm_tools)

        logger.debug(
            "tool_count=<%d>, node_count=<%d> | injected coordination tools into agents",
            len(swarm_tools),
            len(self.nodes),
        )

    def _create_handoff_tool(self) -> Callable[..., Any]:
        """Create handoff tool for agent coordination."""
        swarm_ref = self  # Capture swarm reference

        @tool
        def handoff_to_agent(agent_name: str, message: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
            """Transfer control to another agent in the swarm for specialized help.

            Args:
                agent_name: Name of the agent to hand off to
                message: Message explaining what needs to be done and why you're handing off
                context: Additional context to share with the next agent

            Returns:
                Confirmation of handoff initiation
            """
            try:
                context = context or {}

                # Validate target agent exists
                target_node = swarm_ref.nodes.get(agent_name)
                if not target_node:
                    return {"status": "error", "content": [{"text": f"Error: Agent '{agent_name}' not found in swarm"}]}

                # Execute handoff
                swarm_ref._handle_handoff(target_node, message, context)

                return {"status": "success", "content": [{"text": f"Handed off to {agent_name}: {message}"}]}
            except Exception as e:
                return {"status": "error", "content": [{"text": f"Error in handoff: {str(e)}"}]}

        return handoff_to_agent

    def _handle_handoff(self, target_node: SwarmNode, message: str, context: dict[str, Any]) -> None:
        """Handle handoff to another agent."""
        # If task is already completed, don't allow further handoffs
        if self.state.completion_status != Status.EXECUTING:
            logger.debug(
                "task_status=<%s> | ignoring handoff request - task already completed",
                self.state.completion_status,
            )
            return

        # Update swarm state
        previous_agent = self.state.current_node
        self.state.current_node = target_node

        # Store handoff message for the target agent
        self.state.handoff_message = message

        # Store handoff context as shared context
        if context:
            for key, value in context.items():
                self.shared_context.add_context(previous_agent, key, value)

        logger.debug(
            "from_node=<%s>, to_node=<%s> | handed off from agent to agent",
            previous_agent.node_id,
            target_node.node_id,
        )

    def _build_node_input(self, target_node: SwarmNode) -> str:
        """Build input text for a node based on shared context and handoffs.

        Example formatted output:
        ```
        Handoff Message: The user needs help with Python debugging - I've identified the issue but need someone with more expertise to fix it.

        User Request: My Python script is throwing a KeyError when processing JSON data from an API

        Previous agents who worked on this: data_analyst → code_reviewer

        Shared knowledge from previous agents:
        • data_analyst: {"issue_location": "line 42", "error_type": "missing key validation", "suggested_fix": "add key existence check"}
        • code_reviewer: {"code_quality": "good overall structure", "security_notes": "API key should be in environment variable"}

        Other agents available for collaboration:
        Agent name: data_analyst. Agent description: Analyzes data and provides deeper insights
        Agent name: code_reviewer.
        Agent name: security_specialist. Agent description: Focuses on secure coding practices and vulnerability assessment

        You have access to swarm coordination tools if you need help from other agents. If you don't hand off to another agent, the swarm will consider the task complete.
        ```
        """  # noqa: E501
        context_info: dict[str, Any] = {
            "task": self.state.task,
            "node_history": [node.node_id for node in self.state.node_history],
            "shared_context": {k: v for k, v in self.shared_context.context.items()},
        }
        context_text = ""

        # Include handoff message prominently at the top if present
        if self.state.handoff_message:
            context_text += f"Handoff Message: {self.state.handoff_message}\n\n"

        # Include task information if available
        if "task" in context_info:
            task = context_info.get("task")
            if isinstance(task, str):
                context_text += f"User Request: {task}\n\n"
            elif isinstance(task, list):
                context_text += "User Request: Multi-modal task\n\n"

        # Include detailed node history
        if context_info.get("node_history"):
            context_text += f"Previous agents who worked on this: {' → '.join(context_info['node_history'])}\n\n"

        # Include actual shared context, not just a mention
        shared_context = context_info.get("shared_context", {})
        if shared_context:
            context_text += "Shared knowledge from previous agents:\n"
            for node_name, context in shared_context.items():
                if context:  # Only include if node has contributed context
                    context_text += f"• {node_name}: {context}\n"
            context_text += "\n"

        # Include available nodes with descriptions if available
        other_nodes = [node_id for node_id in self.nodes.keys() if node_id != target_node.node_id]
        if other_nodes:
            context_text += "Other agents available for collaboration:\n"
            for node_id in other_nodes:
                node = self.nodes.get(node_id)
                context_text += f"Agent name: {node_id}."
                if node and hasattr(node.executor, "description") and node.executor.description:
                    context_text += f" Agent description: {node.executor.description}"
                context_text += "\n"
            context_text += "\n"

        context_text += (
            "You have access to swarm coordination tools if you need help from other agents. "
            "If you don't hand off to another agent, the swarm will consider the task complete."
        )

        return context_text

    async def _execute_swarm(self) -> None:
        """Shared execution logic used by execute_async."""
        try:
            # Main execution loop
            while True:
                if self.state.completion_status != Status.EXECUTING:
                    reason = f"Completion status is: {self.state.completion_status}"
                    logger.debug("reason=<%s> | stopping execution", reason)
                    break

                should_continue, reason = self.state.should_continue(
                    max_handoffs=self.max_handoffs,
                    max_iterations=self.max_iterations,
                    execution_timeout=self.execution_timeout,
                    repetitive_handoff_detection_window=self.repetitive_handoff_detection_window,
                    repetitive_handoff_min_unique_agents=self.repetitive_handoff_min_unique_agents,
                )
                if not should_continue:
                    self.state.completion_status = Status.FAILED
                    logger.debug("reason=<%s> | stopping execution", reason)
                    break

                # Get current node
                current_node = self.state.current_node
                if not current_node or current_node.node_id not in self.nodes:
                    logger.error("node=<%s> | node not found", current_node.node_id if current_node else "None")
                    self.state.completion_status = Status.FAILED
                    break

                logger.debug(
                    "current_node=<%s>, iteration=<%d> | executing node",
                    current_node.node_id,
                    len(self.state.node_history) + 1,
                )

                # Execute node with timeout protection
                # TODO: Implement cancellation token to stop _execute_node from continuing
                try:
                    await asyncio.wait_for(
                        self._execute_node(current_node, self.state.task),
                        timeout=self.node_timeout,
                    )

                    self.state.node_history.append(current_node)

                    logger.debug("node=<%s> | node execution completed", current_node.node_id)

                    # Check if the current node is still the same after execution
                    # If it is, then no handoff occurred and we consider the swarm complete
                    if self.state.current_node == current_node:
                        logger.debug("node=<%s> | no handoff occurred, marking swarm as complete", current_node.node_id)
                        self.state.completion_status = Status.COMPLETED
                        break

                except asyncio.TimeoutError:
                    logger.exception(
                        "node=<%s>, timeout=<%s>s | node execution timed out after timeout",
                        current_node.node_id,
                        self.node_timeout,
                    )
                    self.state.completion_status = Status.FAILED
                    break

                except Exception:
                    logger.exception("node=<%s> | node execution failed", current_node.node_id)
                    self.state.completion_status = Status.FAILED
                    break

        except Exception:
            logger.exception("swarm execution failed")
            self.state.completion_status = Status.FAILED

        elapsed_time = time.time() - self.state.start_time
        logger.debug("status=<%s> | swarm execution completed", self.state.completion_status)
        logger.debug(
            "node_history_length=<%d>, time=<%s>s | metrics",
            len(self.state.node_history),
            f"{elapsed_time:.2f}",
        )

    async def _execute_node(self, node: SwarmNode, task: str | list[ContentBlock]) -> AgentResult:
        """Execute swarm node."""
        start_time = time.time()
        node_name = node.node_id

        try:
            # Prepare context for node
            context_text = self._build_node_input(node)
            node_input = [ContentBlock(text=f"Context:\n{context_text}\n\n")]

            # Clear handoff message after it's been included in context
            self.state.handoff_message = None

            if not isinstance(task, str):
                # Include additional ContentBlocks in node input
                node_input = node_input + task

            # Execute node
            result = None
            node.reset_executor_state()
            result = await node.executor.invoke_async(node_input)

            execution_time = round((time.time() - start_time) * 1000)

            # Create NodeResult
            usage = Usage(inputTokens=0, outputTokens=0, totalTokens=0)
            metrics = Metrics(latencyMs=execution_time)
            if hasattr(result, "metrics") and result.metrics:
                if hasattr(result.metrics, "accumulated_usage"):
                    usage = result.metrics.accumulated_usage
                if hasattr(result.metrics, "accumulated_metrics"):
                    metrics = result.metrics.accumulated_metrics

            node_result = NodeResult(
                result=result,
                execution_time=execution_time,
                status=Status.COMPLETED,
                accumulated_usage=usage,
                accumulated_metrics=metrics,
                execution_count=1,
            )

            # Store result in state
            self.state.results[node_name] = node_result

            # Accumulate metrics
            self._accumulate_metrics(node_result)

            return result

        except Exception as e:
            execution_time = round((time.time() - start_time) * 1000)
            logger.exception("node=<%s> | node execution failed", node_name)

            # Create a NodeResult for the failed node
            node_result = NodeResult(
                result=e,  # Store exception as result
                execution_time=execution_time,
                status=Status.FAILED,
                accumulated_usage=Usage(inputTokens=0, outputTokens=0, totalTokens=0),
                accumulated_metrics=Metrics(latencyMs=execution_time),
                execution_count=1,
            )

            # Store result in state
            self.state.results[node_name] = node_result

            raise

    def _accumulate_metrics(self, node_result: NodeResult) -> None:
        """Accumulate metrics from a node result."""
        self.state.accumulated_usage["inputTokens"] += node_result.accumulated_usage.get("inputTokens", 0)
        self.state.accumulated_usage["outputTokens"] += node_result.accumulated_usage.get("outputTokens", 0)
        self.state.accumulated_usage["totalTokens"] += node_result.accumulated_usage.get("totalTokens", 0)
        self.state.accumulated_metrics["latencyMs"] += node_result.accumulated_metrics.get("latencyMs", 0)

    def _build_result(self) -> SwarmResult:
        """Build swarm result from current state."""
        return SwarmResult(
            status=self.state.completion_status,
            results=self.state.results,
            accumulated_usage=self.state.accumulated_usage,
            accumulated_metrics=self.state.accumulated_metrics,
            execution_count=len(self.state.node_history),
            execution_time=self.state.execution_time,
            node_history=self.state.node_history,
        )
