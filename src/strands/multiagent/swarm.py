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
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Tuple, cast

from ..agent import Agent, AgentResult
from ..tools.decorator import tool
from ..types.event_loop import Metrics, Usage
from .base import MultiAgentBase, MultiAgentResult, NodeResult, Status

logger = logging.getLogger(__name__)


@dataclass
class SwarmNode:
    """Represents a node (e.g. Agent) in the swarm."""

    node_id: str
    executor: Agent

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


@dataclass
class SwarmMessage:
    """Message passed between nodes in swarm."""

    from_node: SwarmNode
    to_node: SwarmNode
    content: str
    context: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class SwarmConfig:
    """Configuration for swarm execution safety."""

    max_handoffs: int = 10
    max_iterations: int = 20
    execution_timeout: float = 900.0  # Total execution timeout (seconds)
    node_timeout: float = 300.0  # Individual node timeout (seconds)
    ping_pong_check_nodes: int = 8  # Number of recent nodes to check for ping-pong
    ping_pong_min_unique_nodes: int = 3  # Minimum unique nodes required in recent sequence


@dataclass
class SharedContext:
    """Shared context accessible via tools."""

    context: dict[str, dict[str, Any]] = field(default_factory=dict)
    node_history: list[SwarmNode] = field(default_factory=list)
    current_task: str | None = None
    available_nodes: list[SwarmNode] = field(default_factory=list)

    def set_task(self, task: str) -> None:
        """Set the current task."""
        self.current_task = task

    def set_available_nodes(self, nodes: list[SwarmNode]) -> None:
        """Set list of available agents."""
        self.available_nodes = nodes

    def add_context(self, node: SwarmNode, key: str, value: Any) -> None:
        """Add context."""
        if node.node_id not in self.context:
            self.context[node.node_id] = {}
        self.context[node.node_id][key] = value

    def get_relevant_context(self, target_node: SwarmNode) -> dict[str, Any]:
        """Get context relevant to specific node."""
        return {
            "task": self.current_task,
            "node_history": [node.node_id for node in self.node_history],
            "shared_context": {k: v for k, v in self.context.items() if v},
            "available_nodes": [node.node_id for node in self.available_nodes if node != target_node],
        }


@dataclass
class SwarmState:
    """Current state of swarm execution."""

    current_node: SwarmNode
    task: str
    completion_status: Status = Status.PENDING
    shared_context: SharedContext = field(default_factory=SharedContext)
    node_history: list[SwarmNode] = field(default_factory=list)
    message_history: list[SwarmMessage] = field(default_factory=list)
    iteration_count: int = 0
    start_time: float = field(default_factory=time.time)
    last_node_sequence: list[SwarmNode] = field(default_factory=list)
    final_result: str | None = None
    results: dict[str, NodeResult] = field(default_factory=dict)
    accumulated_usage: Usage = field(default_factory=lambda: Usage(inputTokens=0, outputTokens=0, totalTokens=0))
    accumulated_metrics: Metrics = field(default_factory=lambda: Metrics(latencyMs=0))
    execution_count: int = 0
    execution_time: int = 0

    def should_continue(self, config: SwarmConfig) -> Tuple[bool, str]:
        """Check if the swarm should continue.

        Returns: (should_continue, reason)
        """
        elapsed = time.time() - self.start_time

        # 1. Check completion status
        if self.completion_status != Status.EXECUTING:
            return False, f"completion_status_changed_to_{self.completion_status}"

        # 2. Check handoff limit
        if len(self.node_history) >= config.max_handoffs:
            self.completion_status = Status.FAILED
            return False, f"max_handoffs_reached_{config.max_handoffs}"

        # 3. Check iteration limit
        if self.iteration_count >= config.max_iterations:
            self.completion_status = Status.FAILED
            return False, f"max_iterations_reached_{config.max_iterations}"

        # 4. Check timeout
        if elapsed > config.execution_timeout:
            self.completion_status = Status.FAILED
            return False, f"execution_timeout_{config.execution_timeout}s"

        # 5. Check for node ping-pong (nodes passing back and forth)
        if len(self.last_node_sequence) >= config.ping_pong_check_nodes:
            recent = self.last_node_sequence[-config.ping_pong_check_nodes :]
            unique_nodes = len(set(recent))
            if unique_nodes < config.ping_pong_min_unique_nodes:
                self.completion_status = Status.FAILED
                return (
                    False,
                    f"node_ping_pong_detected_{unique_nodes}_unique_in_{config.ping_pong_check_nodes}_recent",
                )

        return True, "continuing"

    def increment_iteration(self, current_node: SwarmNode, config: SwarmConfig) -> None:
        """Increment iteration count and track node usage."""
        self.iteration_count += 1
        self.last_node_sequence.append(current_node)

        # Keep only the required number of nodes for ping-pong detection
        if len(self.last_node_sequence) > config.ping_pong_check_nodes:
            self.last_node_sequence = self.last_node_sequence[-config.ping_pong_check_nodes :]


@dataclass
class SwarmResult(MultiAgentResult):
    """Result from swarm execution - extends MultiAgentResult with swarm-specific details."""

    status: Status = Status.PENDING
    node_history: list[SwarmNode] = field(default_factory=list)
    message_history: list[SwarmMessage] = field(default_factory=list)
    iteration_count: int = 0
    final_result: str | None = None


class Swarm(MultiAgentBase):
    """Self-organizing collaborative agent teams with shared working memory."""

    def __init__(
        self,
        nodes: list[Agent],
        config: SwarmConfig | None = None,
    ) -> None:
        """Initialize Swarm with agents and configuration.

        Args:
            nodes: List of nodes (e.g. Agent) to include in the swarm
            config: Optional swarm execution configuration
        """
        super().__init__()

        self.config = config or SwarmConfig()
        self.shared_context = SharedContext()
        self.nodes: dict[str, SwarmNode] = {}
        self.state = SwarmState(
            current_node=SwarmNode("", Agent()),  # Placeholder, will be set properly
            task="",
            completion_status=Status.PENDING,
        )

        self._setup_swarm(nodes)
        self._inject_swarm_tools()

    def execute(self, task: str) -> SwarmResult:
        """Execute task synchronously."""

        def execute() -> SwarmResult:
            return asyncio.run(self.execute_async(task))

        with ThreadPoolExecutor() as executor:
            future = executor.submit(execute)
            return future.result()

    async def execute_async(self, task: str) -> SwarmResult:
        """Execute the swarm asynchronously."""
        logger.info("starting swarm execution")

        # Initialize swarm state with configuration
        initial_node = next(iter(self.nodes.values()))  # First SwarmNode
        self.state = SwarmState(
            current_node=initial_node,
            task=task,
            completion_status=Status.EXECUTING,
            shared_context=self.shared_context,
        )
        self.shared_context.set_task(task)

        start_time = time.time()
        try:
            logger.info("current_node=<%s> | starting swarm execution with node", self.state.current_node.node_id)
            logger.info(
                "max_handoffs=<%d>, max_iterations=<%d>, timeout=<%s>s | SwarmConfig",
                self.config.max_handoffs,
                self.config.max_iterations,
                self.config.execution_timeout,
            )

            await self._execute_swarm()

            if self.state.completion_status == Status.EXECUTING:
                self.state.completion_status = Status.COMPLETED

        except Exception:
            logger.exception("swarm execution failed")
            self.state.completion_status = Status.FAILED
            raise
        finally:
            self.state.execution_time = round((time.time() - start_time) * 1000)

        return self._build_result()

    def _setup_swarm(self, nodes: list[Agent]) -> None:
        """Initialize swarm configuration."""
        # Validate agents have names and create SwarmNode objects
        for i, node in enumerate(nodes):
            if not hasattr(node, "name") or not node.name:
                node_id = f"node_{i}"
                node.name = node_id
                logger.info("node_id=<%d> | agent has no name, dynamically generating one", node_id)

            node_id = str(node.name)
            self.nodes[node_id] = SwarmNode(node_id=node_id, executor=node)

        swarm_nodes = list(self.nodes.values())
        self.shared_context.set_available_nodes(swarm_nodes)
        logger.info("nodes=<%s> | initialized swarm with nodes", [node.node_id for node in swarm_nodes])

    def _inject_swarm_tools(self) -> None:
        """Add swarm coordination tools to each agent."""
        # Create tool functions with proper closures
        swarm_tools = [
            self._create_handoff_tool(),
            self._create_complete_tool(),
            self._create_context_tool(),
        ]

        for node in self.nodes.values():
            # Use the agent's tool registry to process and register the tools
            node.executor.tool_registry.process_tools(swarm_tools)

        logger.info(
            "tool_count=<%d>, node_count=<%d> | injected coordination tools into agents",
            len(swarm_tools),
            len(self.nodes),
        )

    def _create_handoff_tool(self) -> Callable[..., Any]:
        """Create handoff tool for agent coordination."""
        swarm_ref = self  # Capture swarm reference

        @tool
        def handoff_to_agent(  # noqa: D417
            agent: Agent, agent_name: str, message: str, context: dict[str, Any] | None = None
        ) -> dict[str, Any]:
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
                if not swarm_ref.nodes.get(agent_name):
                    return {"status": "error", "content": [{"text": f"Error: Agent '{agent_name}' not found in swarm"}]}

                # Execute handoff
                swarm_ref._handle_handoff(agent, agent_name, message, context)

                return {"status": "success", "content": [{"text": f"Handed off to {agent_name}: {message}"}]}

            except Exception as e:
                return {"status": "error", "content": [{"text": f"Error in handoff: {str(e)}"}]}

        return handoff_to_agent

    def _create_complete_tool(self) -> Callable[..., Any]:
        """Create completion tool for task completion."""
        swarm_ref = self  # Capture swarm reference

        @tool
        def complete_swarm_task(agent: Agent, result: str, summary: str | None = None) -> dict[str, Any]:  # noqa: D417
            """Mark the task as complete with final result. No more agents will be called.

            Args:
                result: The final result/answer
                summary: Optional summary of how the task was completed

            Returns:
                Task completion confirmation
            """
            try:
                # Mark swarm as complete
                swarm_ref._handle_completion(agent, result, summary or "")

                return {"status": "success", "content": [{"text": f"Task completed: {result}"}]}

            except Exception as e:
                return {"status": "error", "content": [{"text": f"Error completing task: {str(e)}"}]}

        return complete_swarm_task

    def _create_context_tool(self) -> Callable[..., Any]:
        """Create context tool for accessing shared context."""
        swarm_ref = self  # Capture swarm reference

        @tool
        def get_swarm_context() -> dict[str, Any]:
            """Get the current shared context and agent history.

            Returns:
                Current swarm state including shared facts, agent history, etc.
            """
            try:
                # Get context for current agent
                current_node = swarm_ref.state.current_node
                context = swarm_ref.shared_context.get_relevant_context(current_node)

                context_text = swarm_ref._format_context(context)

                return {"status": "success", "content": [{"text": context_text}]}

            except Exception as e:
                return {"status": "error", "content": [{"text": f"Error getting context: {str(e)}"}]}

        return get_swarm_context

    def _handle_handoff(
        self, agent: Agent, target_agent_name: str, message: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle handoff to another agent."""
        # If task is already completed, don't allow further handoffs
        if self.state.completion_status != Status.EXECUTING:
            logger.info(
                "task_status=<%s> | ignoring handoff request - task already completed",
                self.state.completion_status,
            )
            return {"status": "ignored", "reason": f"task_already_{self.state.completion_status}"}

        # Get target node
        target_node = self.nodes.get(target_agent_name)
        if not target_node:
            return {"status": "error", "reason": f"agent_{target_agent_name}_not_found"}

        # Update swarm state
        previous_agent = self.state.current_node
        self.state.current_node = target_node

        # Add handoff message
        handoff_msg = SwarmMessage(
            from_node=previous_agent,
            to_node=target_node,
            content=message,
            context=context,
        )
        self.state.message_history.append(handoff_msg)

        # Store handoff context as shared context
        if context:
            for key, value in context.items():
                self.shared_context.add_context(previous_agent, key, value)

        agent.stop_event_loop = True

        logger.info(
            "from_node=<%s>, to_node=<%s> | handed off from agent to agent",
            previous_agent.node_id,
            target_node.node_id,
        )
        return {"status": "success", "target_agent": target_agent_name}

    def _handle_completion(self, agent: Agent, result: str, summary: str = "") -> None:
        """Handle task completion."""
        self.state.completion_status = Status.COMPLETED
        self.state.final_result = result

        # Create a system node for completion message
        system_node = SwarmNode("system", Agent())

        # Add completion message
        completion_msg = SwarmMessage(
            from_node=self.state.current_node,
            to_node=system_node,
            content=result,
            context={"summary": summary},
        )
        self.state.message_history.append(completion_msg)

        agent.stop_event_loop = True

        logger.info("swarm task completed")

    def _format_context(self, context_info: dict[str, Any]) -> str:
        """Format task message with relevant context."""
        context_text = ""

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

        # Include available nodes
        if context_info.get("available_nodes"):
            context_text += (
                f"Other agents available for collaboration: {', '.join(context_info['available_nodes'])}\n\n"
            )

        context_text += (
            "You have access to swarm coordination tools if you need help from other agents "
            "or want to complete the task."
        )

        return context_text

    async def _execute_swarm(self) -> None:
        """Shared execution logic used by execute_async."""
        try:
            # Main execution loop
            while True:
                should_continue, reason = self.state.should_continue(self.config)
                if not should_continue:
                    logger.info("reason=<%s> | stopping execution", reason)
                    break

                self.state.increment_iteration(self.state.current_node, self.config)

                # Get current node
                current_node_node = self.state.current_node
                if not current_node_node or current_node_node.node_id not in self.nodes:
                    logger.error(
                        "node=<%s> | node not found", current_node_node.node_id if current_node_node else "None"
                    )
                    self.state.completion_status = Status.FAILED
                    break

                logger.info(
                    "current_node=<%s>, iteration=<%d> | executing node",
                    current_node_node.node_id,
                    self.state.iteration_count,
                )

                # Execute node with timeout protection
                try:
                    await asyncio.wait_for(
                        self._execute_node(current_node_node, self.state.task),
                        timeout=self.config.node_timeout,
                    )

                    self.state.node_history.append(current_node_node)
                    self.shared_context.node_history.append(current_node_node)

                    logger.info("node=<%s> | node execution completed", current_node_node.node_id)

                    # Immediate check for completion after node execution
                    if self.state.completion_status != Status.EXECUTING:
                        logger.info("status=<%s> | task completed with status", self.state.completion_status)
                        break

                except asyncio.TimeoutError:
                    logger.exception(
                        "node=<%s>, timeout=<%s>s | node execution timed out after timeout",
                        current_node_node.node_id,
                        self.config.node_timeout,
                    )
                    self.state.completion_status = Status.FAILED
                    break

                except Exception:
                    logger.exception("node=<%s> | node execution failed", current_node_node.node_id)
                    self.state.completion_status = Status.FAILED
                    break

        except Exception:
            logger.exception("swarm execution failed")
            self.state.completion_status = Status.FAILED

        elapsed_time = time.time() - self.state.start_time
        logger.info("status=<%s> | swarm execution completed", self.state.completion_status)
        logger.info(
            "iterations=<%d>, handoffs=<%d>, time=<%s>s | metrics",
            self.state.iteration_count,
            len(self.state.node_history),
            f"{elapsed_time:.2f}",
        )

    async def _execute_node(self, node: SwarmNode, task: str) -> AgentResult:
        """Execute swarm node."""
        start_time = time.time()
        node_name = node.node_id

        try:
            # Prepare context for node
            context_info = self.shared_context.get_relevant_context(node)

            # Create task message with context
            task_with_context = f"Task: {task}\n\n"
            task_with_context += self._format_context(context_info)

            # Execute node
            result = None
            node.executor.messages = []  # Reset agent's messages to avoid polluting context
            async for event in node.executor.stream_async(task_with_context):
                if "result" in event:
                    result = cast(AgentResult, event["result"])

            if not result:
                raise ValueError(f"Node '{node_name}' did not return a result")

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
        self.state.execution_count += node_result.execution_count

    def _build_result(self) -> SwarmResult:
        """Build swarm result from current state."""
        return SwarmResult(
            status=self.state.completion_status,
            results=self.state.results,
            accumulated_usage=self.state.accumulated_usage,
            accumulated_metrics=self.state.accumulated_metrics,
            execution_count=self.state.execution_count,
            execution_time=self.state.execution_time,
            node_history=self.state.node_history,
            message_history=self.state.message_history,
            iteration_count=self.state.iteration_count,
            final_result=self.state.final_result,
        )
