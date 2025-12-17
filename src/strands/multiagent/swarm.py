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
- Human input via user interrupts raised in BeforeNodeCallEvent hooks and agent nodes
"""

import asyncio
import copy
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Mapping, Optional, Tuple, cast

from opentelemetry import trace as trace_api

from .._async import run_async
from ..agent import Agent
from ..agent.state import AgentState
from ..experimental.hooks.multiagent import (
    AfterMultiAgentInvocationEvent,
    AfterNodeCallEvent,
    BeforeMultiAgentInvocationEvent,
    BeforeNodeCallEvent,
    MultiAgentInitializedEvent,
)
from ..hooks import HookProvider, HookRegistry
from ..interrupt import Interrupt, _InterruptState
from ..session import SessionManager
from ..telemetry import get_tracer
from ..tools.decorator import tool
from ..types._events import (
    MultiAgentHandoffEvent,
    MultiAgentNodeCancelEvent,
    MultiAgentNodeInterruptEvent,
    MultiAgentNodeStartEvent,
    MultiAgentNodeStopEvent,
    MultiAgentNodeStreamEvent,
    MultiAgentResultEvent,
)
from ..types.content import ContentBlock, Messages
from ..types.event_loop import Metrics, Usage
from ..types.multiagent import MultiAgentInput
from ..types.traces import AttributeValue
from .base import MultiAgentBase, MultiAgentResult, NodeResult, Status

logger = logging.getLogger(__name__)

_DEFAULT_SWARM_ID = "default_swarm"


@dataclass
class SwarmNode:
    """Represents a node (e.g. Agent) in the swarm."""

    node_id: str
    executor: Agent
    swarm: Optional["Swarm"] = None
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
        """Reset SwarmNode executor state to initial state when swarm was created.

        If Swarm is resuming from an interrupt, we reset the executor state from the interrupt context.
        """
        if self.swarm and self.swarm._interrupt_state.activated:
            context = self.swarm._interrupt_state.context[self.node_id]
            self.executor.messages = context["messages"]
            self.executor.state = AgentState(context["state"])
            self.executor._interrupt_state = _InterruptState.from_dict(context["interrupt_state"])
            return

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

    current_node: SwarmNode | None  # The agent currently executing
    task: MultiAgentInput  # The original task from the user that is being executed
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
    handoff_node: SwarmNode | None = None  # The agent to execute next
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
        entry_point: Agent | None = None,
        max_handoffs: int = 20,
        max_iterations: int = 20,
        execution_timeout: float = 900.0,
        node_timeout: float = 300.0,
        repetitive_handoff_detection_window: int = 0,
        repetitive_handoff_min_unique_agents: int = 0,
        session_manager: Optional[SessionManager] = None,
        hooks: Optional[list[HookProvider]] = None,
        id: str = _DEFAULT_SWARM_ID,
        trace_attributes: Optional[Mapping[str, AttributeValue]] = None,
    ) -> None:
        """Initialize Swarm with agents and configuration.

        Args:
            id: Unique swarm id (default: "default_swarm")
            nodes: List of nodes (e.g. Agent) to include in the swarm
            entry_point: Agent to start with. If None, uses the first agent (default: None)
            max_handoffs: Maximum handoffs to agents and users (default: 20)
            max_iterations: Maximum node executions within the swarm (default: 20)
            execution_timeout: Total execution timeout in seconds (default: 900.0)
            node_timeout: Individual node timeout in seconds (default: 300.0)
            repetitive_handoff_detection_window: Number of recent nodes to check for repetitive handoffs
                Disabled by default (default: 0)
            repetitive_handoff_min_unique_agents: Minimum unique agents required in recent sequence
                Disabled by default (default: 0)
            session_manager: Session manager for persisting graph state and execution history (default: None)
            hooks: List of hook providers for monitoring and extending graph execution behavior (default: None)
            trace_attributes: Custom trace attributes to apply to the agent's trace span (default: None)
        """
        super().__init__()
        self.id = id
        self.entry_point = entry_point
        self.max_handoffs = max_handoffs
        self.max_iterations = max_iterations
        self.execution_timeout = execution_timeout
        self.node_timeout = node_timeout
        self.repetitive_handoff_detection_window = repetitive_handoff_detection_window
        self.repetitive_handoff_min_unique_agents = repetitive_handoff_min_unique_agents

        self.shared_context = SharedContext()
        self.nodes: dict[str, SwarmNode] = {}

        self.state = SwarmState(
            current_node=None,  # Placeholder, will be set properly
            task="",
            completion_status=Status.PENDING,
        )
        self._interrupt_state = _InterruptState()

        self.tracer = get_tracer()
        self.trace_attributes: dict[str, AttributeValue] = self._parse_trace_attributes(trace_attributes)

        self.session_manager = session_manager
        self.hooks = HookRegistry()
        if hooks:
            for hook in hooks:
                self.hooks.add_hook(hook)
        if self.session_manager:
            self.hooks.add_hook(self.session_manager)

        self._resume_from_session = False

        self._setup_swarm(nodes)
        self._inject_swarm_tools()
        run_async(lambda: self.hooks.invoke_callbacks_async(MultiAgentInitializedEvent(self)))

    def __call__(
        self, task: MultiAgentInput, invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> SwarmResult:
        """Invoke the swarm synchronously.

        Args:
            task: The task to execute
            invocation_state: Additional state/context passed to underlying agents.
                Defaults to None to avoid mutable default argument issues.
            **kwargs: Keyword arguments allowing backward compatible future changes.
        """
        if invocation_state is None:
            invocation_state = {}
        return run_async(lambda: self.invoke_async(task, invocation_state))

    async def invoke_async(
        self, task: MultiAgentInput, invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> SwarmResult:
        """Invoke the swarm asynchronously.

        This method uses stream_async internally and consumes all events until completion,
        following the same pattern as the Agent class.

        Args:
            task: The task to execute
            invocation_state: Additional state/context passed to underlying agents.
                Defaults to None to avoid mutable default argument issues.
            **kwargs: Keyword arguments allowing backward compatible future changes.
        """
        events = self.stream_async(task, invocation_state, **kwargs)
        final_event = None
        async for event in events:
            final_event = event

        if final_event is None or "result" not in final_event:
            raise ValueError("Swarm streaming completed without producing a result event")

        return cast(SwarmResult, final_event["result"])

    async def stream_async(
        self, task: MultiAgentInput, invocation_state: dict[str, Any] | None = None, **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream events during swarm execution.

        Args:
            task: The task to execute
            invocation_state: Additional state/context passed to underlying agents.
                Defaults to None to avoid mutable default argument issues.
            **kwargs: Keyword arguments allowing backward compatible future changes.

        Yields:
            Dictionary events during swarm execution, such as:
            - multi_agent_node_start: When a node begins execution
            - multi_agent_node_stream: Forwarded agent events with node context
            - multi_agent_handoff: When control is handed off between agents
            - multi_agent_node_stop: When a node stops execution
            - result: Final swarm result
        """
        self._interrupt_state.resume(task)

        if invocation_state is None:
            invocation_state = {}

        await self.hooks.invoke_callbacks_async(BeforeMultiAgentInvocationEvent(self, invocation_state))

        logger.debug("starting swarm execution")

        if self._resume_from_session or self._interrupt_state.activated:
            self.state.completion_status = Status.EXECUTING
            self.state.start_time = time.time()
        else:
            # Initialize swarm state with configuration
            initial_node = self._initial_node()

            self.state = SwarmState(
                current_node=initial_node,
                task=task,
                completion_status=Status.EXECUTING,
                shared_context=self.shared_context,
            )

        span = self.tracer.start_multiagent_span(task, "swarm", custom_trace_attributes=self.trace_attributes)
        with trace_api.use_span(span, end_on_exit=True):
            interrupts = []

            try:
                current_node = cast(SwarmNode, self.state.current_node)
                logger.debug("current_node=<%s> | starting swarm execution with node", current_node.node_id)
                logger.debug(
                    "max_handoffs=<%d>, max_iterations=<%d>, timeout=<%s>s | swarm execution config",
                    self.max_handoffs,
                    self.max_iterations,
                    self.execution_timeout,
                )

                async for event in self._execute_swarm(invocation_state):
                    if isinstance(event, MultiAgentNodeInterruptEvent):
                        interrupts = event.interrupts

                    yield event.as_dict()

            except Exception:
                logger.exception("swarm execution failed")
                self.state.completion_status = Status.FAILED
                raise
            finally:
                self.state.execution_time = round((time.time() - self.state.start_time) * 1000)
                await self.hooks.invoke_callbacks_async(AfterMultiAgentInvocationEvent(self, invocation_state))
                self._resume_from_session = False

            # Yield final result after execution_time is set
            result = self._build_result(interrupts)
            yield MultiAgentResultEvent(result=result).as_dict()

    async def _stream_with_timeout(
        self, async_generator: AsyncIterator[Any], timeout: float | None, timeout_message: str
    ) -> AsyncIterator[Any]:
        """Wrap an async generator with timeout for total execution time.

        Tracks elapsed time from start and enforces timeout across all events.
        Each event wait uses remaining time from the total timeout budget.

        Args:
            async_generator: The generator to wrap
            timeout: Total timeout in seconds for entire stream, or None for no timeout
            timeout_message: Message to include in timeout exception

        Yields:
            Events from the wrapped generator as they arrive

        Raises:
            Exception: If total execution time exceeds timeout
        """
        if timeout is None:
            # No timeout - just pass through
            async for event in async_generator:
                yield event
        else:
            # Track start time for total timeout
            start_time = asyncio.get_event_loop().time()

            while True:
                # Calculate remaining time from total timeout budget
                elapsed = asyncio.get_event_loop().time() - start_time
                remaining = timeout - elapsed

                if remaining <= 0:
                    raise Exception(timeout_message)

                try:
                    event = await asyncio.wait_for(async_generator.__anext__(), timeout=remaining)
                    yield event
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as err:
                    raise Exception(timeout_message) from err

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

            self.nodes[node_id] = SwarmNode(node_id, node, swarm=self)

        # Validate entry point if specified
        if self.entry_point is not None:
            entry_point_node_id = str(self.entry_point.name)
            if (
                entry_point_node_id not in self.nodes
                or self.nodes[entry_point_node_id].executor is not self.entry_point
            ):
                available_agents = [
                    f"{node_id} ({type(node.executor).__name__})" for node_id, node in self.nodes.items()
                ]
                raise ValueError(f"Entry point agent not found in swarm nodes. Available agents: {available_agents}")

        swarm_nodes = list(self.nodes.values())
        logger.debug("nodes=<%s> | initialized swarm with nodes", [node.node_id for node in swarm_nodes])

        if self.entry_point:
            entry_point_name = getattr(self.entry_point, "name", "unnamed_agent")
            logger.debug("entry_point=<%s> | configured entry point", entry_point_name)
        else:
            first_node = next(iter(self.nodes.keys()))
            logger.debug("entry_point=<%s> | using first node as entry point", first_node)

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

                return {"status": "success", "content": [{"text": f"Handing off to {agent_name}: {message}"}]}
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

        current_node = cast(SwarmNode, self.state.current_node)

        self.state.handoff_node = target_node
        self.state.handoff_message = message

        # Store handoff context as shared context
        if context:
            for key, value in context.items():
                self.shared_context.add_context(current_node, key, value)

        logger.debug(
            "from_node=<%s>, to_node=<%s> | handing off from agent to agent",
            current_node.node_id,
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

    def _activate_interrupt(self, node: SwarmNode, interrupts: list[Interrupt]) -> MultiAgentNodeInterruptEvent:
        """Activate the interrupt state.

        Note, a Swarm may be interrupted either from a BeforeNodeCallEvent hook or from within an agent node. In either
        case, we must manage the interrupt state of both the Swarm and the individual agent nodes.

        Args:
            node: The interrupted node.
            interrupts: The interrupts raised by the user.

        Returns:
            MultiAgentNodeInterruptEvent
        """
        logger.debug("node=<%s> | node interrupted", node.node_id)
        self.state.completion_status = Status.INTERRUPTED

        self._interrupt_state.context[node.node_id] = {
            "activated": node.executor._interrupt_state.activated,
            "interrupt_state": node.executor._interrupt_state.to_dict(),
            "state": node.executor.state.get(),
            "messages": node.executor.messages,
        }

        self._interrupt_state.interrupts.update({interrupt.id: interrupt for interrupt in interrupts})
        self._interrupt_state.activate()

        return MultiAgentNodeInterruptEvent(node.node_id, interrupts)

    async def _execute_swarm(self, invocation_state: dict[str, Any]) -> AsyncIterator[Any]:
        """Execute swarm and yield TypedEvent objects."""
        try:
            # Main execution loop
            while True:
                if self.state.completion_status != Status.EXECUTING:
                    reason = f"Completion status is: {self.state.completion_status}"
                    logger.debug("reason=<%s> | stopping streaming execution", reason)
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

                before_event, interrupts = await self.hooks.invoke_callbacks_async(
                    BeforeNodeCallEvent(self, current_node.node_id, invocation_state)
                )

                # TODO: Implement cancellation token to stop _execute_node from continuing
                try:
                    if interrupts:
                        yield self._activate_interrupt(current_node, interrupts)
                        break

                    if before_event.cancel_node:
                        cancel_message = (
                            before_event.cancel_node
                            if isinstance(before_event.cancel_node, str)
                            else "node cancelled by user"
                        )
                        logger.debug("reason=<%s> | cancelling execution", cancel_message)
                        yield MultiAgentNodeCancelEvent(current_node.node_id, cancel_message)
                        self.state.completion_status = Status.FAILED
                        break

                    node_stream = self._stream_with_timeout(
                        self._execute_node(current_node, self.state.task, invocation_state),
                        self.node_timeout,
                        f"Node '{current_node.node_id}' execution timed out after {self.node_timeout}s",
                    )
                    async for event in node_stream:
                        yield event

                    stop_event = cast(MultiAgentNodeStopEvent, event)
                    node_result = stop_event["node_result"]
                    if node_result.status == Status.INTERRUPTED:
                        yield self._activate_interrupt(current_node, node_result.interrupts)
                        break

                    self._interrupt_state.deactivate()

                    self.state.node_history.append(current_node)

                except Exception:
                    logger.exception("node=<%s> | node execution failed", current_node.node_id)
                    self.state.completion_status = Status.FAILED
                    break

                finally:
                    await self.hooks.invoke_callbacks_async(
                        AfterNodeCallEvent(self, current_node.node_id, invocation_state)
                    )

                logger.debug("node=<%s> | node execution completed", current_node.node_id)

                # Check if handoff requested during execution
                if self.state.handoff_node:
                    previous_node = current_node
                    current_node = self.state.handoff_node

                    self.state.handoff_node = None
                    self.state.current_node = current_node

                    handoff_event = MultiAgentHandoffEvent(
                        from_node_ids=[previous_node.node_id],
                        to_node_ids=[current_node.node_id],
                        message=self.state.handoff_message or "Agent handoff occurred",
                    )
                    yield handoff_event
                    logger.debug(
                        "from_node=<%s>, to_node=<%s> | handoff detected",
                        previous_node.node_id,
                        current_node.node_id,
                    )

                else:
                    logger.debug("node=<%s> | no handoff occurred, marking swarm as complete", current_node.node_id)
                    self.state.completion_status = Status.COMPLETED
                    break

        except Exception:
            logger.exception("swarm execution failed")
            self.state.completion_status = Status.FAILED
        finally:
            elapsed_time = time.time() - self.state.start_time
            logger.debug("status=<%s> | swarm execution completed", self.state.completion_status)
            logger.debug(
                "node_history_length=<%d>, time=<%s>s | metrics",
                len(self.state.node_history),
                f"{elapsed_time:.2f}",
            )

    async def _execute_node(
        self, node: SwarmNode, task: MultiAgentInput, invocation_state: dict[str, Any]
    ) -> AsyncIterator[Any]:
        """Execute swarm node and yield TypedEvent objects."""
        start_time = time.time()
        node_name = node.node_id

        # Emit node start event
        start_event = MultiAgentNodeStartEvent(node_id=node_name, node_type="agent")
        yield start_event

        try:
            if self._interrupt_state.activated and self._interrupt_state.context[node_name]["activated"]:
                node_input = self._interrupt_state.context["responses"]

            else:
                # Prepare context for node
                context_text = self._build_node_input(node)
                node_input = [ContentBlock(text=f"Context:\n{context_text}\n\n")]

                # Clear handoff message after it's been included in context
                self.state.handoff_message = None

                if not isinstance(task, str):
                    # Include additional ContentBlocks in node input
                    node_input = node_input + cast(list[ContentBlock], task)

            # Execute node with streaming
            node.reset_executor_state()

            # Stream agent events with node context and capture final result
            result = None
            async for event in node.executor.stream_async(node_input, invocation_state=invocation_state):
                # Forward agent events with node context
                wrapped_event = MultiAgentNodeStreamEvent(node_name, event)
                yield wrapped_event
                # Capture the final result event
                if "result" in event:
                    result = event["result"]

            if result is None:
                raise ValueError(f"Node '{node_name}' did not produce a result event")

            execution_time = round((time.time() - start_time) * 1000)
            status = Status.INTERRUPTED if result.stop_reason == "interrupt" else Status.COMPLETED

            # Create NodeResult with extracted metrics
            result_metrics = getattr(result, "metrics", None)
            usage = getattr(result_metrics, "accumulated_usage", Usage(inputTokens=0, outputTokens=0, totalTokens=0))
            metrics = getattr(result_metrics, "accumulated_metrics", Metrics(latencyMs=execution_time))

            node_result = NodeResult(
                result=result,
                execution_time=execution_time,
                status=status,
                accumulated_usage=usage,
                accumulated_metrics=metrics,
                execution_count=1,
                interrupts=result.interrupts or [],
            )

            # Store result in state
            self.state.results[node_name] = node_result

            # Accumulate metrics
            self._accumulate_metrics(node_result)

            # Emit node stop event with full NodeResult
            complete_event = MultiAgentNodeStopEvent(
                node_id=node_name,
                node_result=node_result,
            )
            yield complete_event

        except Exception as e:
            execution_time = round((time.time() - start_time) * 1000)
            logger.exception("node=<%s> | node execution failed", node_name)

            # Create a NodeResult for the failed node
            node_result = NodeResult(
                result=e,
                execution_time=execution_time,
                status=Status.FAILED,
                accumulated_usage=Usage(inputTokens=0, outputTokens=0, totalTokens=0),
                accumulated_metrics=Metrics(latencyMs=execution_time),
                execution_count=1,
            )

            # Store result in state
            self.state.results[node_name] = node_result

            # Emit node stop event even for failures
            complete_event = MultiAgentNodeStopEvent(
                node_id=node_name,
                node_result=node_result,
            )
            yield complete_event

            raise

    def _accumulate_metrics(self, node_result: NodeResult) -> None:
        """Accumulate metrics from a node result."""
        self.state.accumulated_usage["inputTokens"] += node_result.accumulated_usage.get("inputTokens", 0)
        self.state.accumulated_usage["outputTokens"] += node_result.accumulated_usage.get("outputTokens", 0)
        self.state.accumulated_usage["totalTokens"] += node_result.accumulated_usage.get("totalTokens", 0)
        self.state.accumulated_metrics["latencyMs"] += node_result.accumulated_metrics.get("latencyMs", 0)

    def _build_result(self, interrupts: list[Interrupt]) -> SwarmResult:
        """Build swarm result from current state."""
        return SwarmResult(
            status=self.state.completion_status,
            results=self.state.results,
            accumulated_usage=self.state.accumulated_usage,
            accumulated_metrics=self.state.accumulated_metrics,
            execution_count=len(self.state.node_history),
            execution_time=self.state.execution_time,
            node_history=self.state.node_history,
            interrupts=interrupts,
        )

    def serialize_state(self) -> dict[str, Any]:
        """Serialize the current swarm state to a dictionary."""
        status_str = self.state.completion_status.value
        if self.state.completion_status == Status.EXECUTING and self.state.current_node:
            next_nodes = [self.state.current_node.node_id]
        elif self.state.completion_status == Status.INTERRUPTED and self.state.current_node:
            next_nodes = [self.state.current_node.node_id]
        elif self.state.handoff_node:
            next_nodes = [self.state.handoff_node.node_id]
        else:
            next_nodes = []

        return {
            "type": "swarm",
            "id": self.id,
            "status": status_str,
            "node_history": [n.node_id for n in self.state.node_history],
            "node_results": {k: v.to_dict() for k, v in self.state.results.items()},
            "next_nodes_to_execute": next_nodes,
            "current_task": self.state.task,
            "context": {
                "shared_context": getattr(self.state.shared_context, "context", {}) or {},
                "handoff_node": self.state.handoff_node.node_id if self.state.handoff_node else None,
                "handoff_message": self.state.handoff_message,
            },
            "_internal_state": {
                "interrupt_state": self._interrupt_state.to_dict(),
            },
        }

    def deserialize_state(self, payload: dict[str, Any]) -> None:
        """Restore swarm state from a session dict and prepare for execution.

        This method handles two scenarios:
        1. If the persisted status is COMPLETED, FAILED resets all nodes and graph state
           to allow re-execution from the beginning.
        2. Otherwise, restores the persisted state and prepares to resume execution
           from the next ready nodes.

        Args:
            payload: Dictionary containing persisted state data including status,
                    completed nodes, results, and next nodes to execute.
        """
        if "_internal_state" in payload:
            internal_state = payload["_internal_state"]
            self._interrupt_state = _InterruptState.from_dict(internal_state["interrupt_state"])

        self._resume_from_session = "next_nodes_to_execute" in payload
        if self._resume_from_session:
            self._from_dict(payload)
            return

        for node in self.nodes.values():
            node.reset_executor_state()

        self.state = SwarmState(
            current_node=SwarmNode("", Agent(), swarm=self),
            task="",
            completion_status=Status.PENDING,
        )

    def _from_dict(self, payload: dict[str, Any]) -> None:
        self.state.completion_status = Status(payload["status"])
        # Hydrate completed nodes & results
        context = payload["context"] or {}
        self.shared_context.context = context.get("shared_context") or {}
        self.state.handoff_message = context.get("handoff_message")
        self.state.handoff_node = self.nodes[context["handoff_node"]] if context.get("handoff_node") else None

        self.state.node_history = [self.nodes[nid] for nid in (payload.get("node_history") or []) if nid in self.nodes]

        raw_results = payload.get("node_results") or {}
        results: dict[str, NodeResult] = {}
        for node_id, entry in raw_results.items():
            if node_id not in self.nodes:
                continue
            try:
                results[node_id] = NodeResult.from_dict(entry)
            except Exception:
                logger.exception("Failed to hydrate NodeResult for node_id=%s; skipping.", node_id)
                raise
        self.state.results = results
        self.state.task = payload.get("current_task", self.state.task)

        next_node_ids = payload.get("next_nodes_to_execute") or []
        if next_node_ids:
            self.state.current_node = self.nodes[next_node_ids[0]] if next_node_ids[0] else self._initial_node()

    def _initial_node(self) -> SwarmNode:
        if self.entry_point:
            return self.nodes[str(self.entry_point.name)]
        return next(iter(self.nodes.values()))  # First SwarmNode
