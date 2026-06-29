"""Agent-as-tool adapter.

This module provides the _AgentAsTool class that wraps an Agent as a tool
so it can be passed to another agent's tool list.
"""

from __future__ import annotations

import copy
import logging
import threading
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

from ..agent.state import AgentState
from ..interrupt import _InterruptState
from ..types._events import AgentAsToolStreamEvent, ToolInterruptEvent, ToolResultEvent
from ..types.content import Messages
from ..types.interrupt import InterruptResponseContent
from ..types.tools import AgentTool, ToolGenerator, ToolSpec, ToolUse

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)


class _AgentAsTool(AgentTool):
    """Adapter that exposes an Agent as a tool for use by other agents.

    The tool accepts a single ``input`` string parameter, invokes the wrapped
    agent, and returns the text response.

    Example:
        ```python
        from strands import Agent

        researcher = Agent(name="researcher", description="Finds information")

        # Use via convenience method (default: fresh conversation each call)
        tool = researcher.as_tool()

        # Preserve context across invocations
        tool = researcher.as_tool(preserve_context=True)

        writer = Agent(name="writer", tools=[tool])
        writer("Write about AI agents")
        ```
    """

    def __init__(
        self,
        agent: Agent,
        *,
        name: str,
        description: str | None = None,
        preserve_context: bool = False,
    ) -> None:
        r"""Initialize the agent-as-tool adapter.

        Args:
            agent: The agent to wrap as a tool.
            name: Tool name. Must match the pattern ``[a-zA-Z0-9_\\-]{1,64}``.
            description: Tool description. Defaults to the agent's description, or a
                generic description if the agent has no description set.
            preserve_context: Whether to preserve the agent's conversation history across
                invocations. When False, the agent's messages and state are reset to the
                values they had at construction time before each call, ensuring every
                invocation starts from the same baseline regardless of any external
                interactions with the agent. Defaults to False.
        """
        super().__init__()
        self._agent = agent
        self._tool_name = name
        self._description = (
            description or agent.description or f"Use the {name} agent as a tool by providing a natural language input"
        )
        self._preserve_context = preserve_context

        # When preserve_context=False, we snapshot the agent's initial state so we can
        # restore it before each invocation. This mirrors GraphNode.reset_executor_state().
        self._initial_messages: Messages = []
        self._initial_state: AgentState = AgentState()
        # Serialize access so _reset_agent_state + stream_async are atomic.
        # threading.Lock (not asyncio.Lock) because run_async() may create
        # separate event loops in different threads.
        self._lock = threading.Lock()

        if not preserve_context:
            if getattr(agent, "_session_manager", None) is not None:
                raise ValueError(
                    "preserve_context=False cannot be used with an agent that has a session manager. "
                    "The session manager persists conversation history externally, which conflicts with "
                    "resetting the agent's state between invocations."
                )
            self._initial_messages = copy.deepcopy(agent.messages)
            self._initial_state = AgentState(agent.state.get())

    @property
    def agent(self) -> Agent:
        """The wrapped agent instance."""
        return self._agent

    @property
    def tool_name(self) -> str:
        """Get the tool name."""
        return self._tool_name

    @property
    def tool_spec(self) -> ToolSpec:
        """Get the tool specification."""
        return {
            "name": self._tool_name,
            "description": self._description,
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": "The input to send to the agent tool.",
                        },
                    },
                    "required": ["input"],
                }
            },
        }

    @property
    def tool_type(self) -> str:
        """Get the tool type."""
        return "agent"

    @override
    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        """Invoke the wrapped agent via streaming and yield events.

        Intermediate agent events are wrapped in AgentAsToolStreamEvent so the caller
        can distinguish sub-agent progress from regular tool events. The final
        AgentResult is yielded as a ToolResultEvent.

        When the sub-agent encounters a hook interrupt (e.g. from BeforeToolCallEvent),
        the interrupts are propagated to the parent agent via ToolInterruptEvent. On
        resume, interrupt responses are forwarded to the sub-agent automatically.

        Args:
            tool_use: The tool use request containing the input parameter.
            invocation_state: Context for the tool invocation.
            **kwargs: Additional keyword arguments.

        Yields:
            AgentAsToolStreamEvent for intermediate events, ToolInterruptEvent if the
            sub-agent is interrupted, or ToolResultEvent with the final response.
        """
        tool_input = tool_use["input"]
        if isinstance(tool_input, dict):
            prompt = tool_input.get("input", "")
        elif isinstance(tool_input, str):
            prompt = tool_input
        else:
            logger.warning("tool_name=<%s> | unexpected input type: %s", self._tool_name, type(tool_input))
            prompt = str(tool_input)

        tool_use_id = tool_use["toolUseId"]

        # Serialize access to the underlying agent. _reset_agent_state() mutates
        # the agent before stream_async acquires its own lock, so a concurrent
        # call would corrupt an in-flight invocation.
        if not self._lock.acquire(blocking=False):
            logger.warning(
                "tool_name=<%s>, tool_use_id=<%s> | agent is already processing a request",
                self._tool_name,
                tool_use_id,
            )
            yield ToolResultEvent(
                {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": f"Agent '{self._tool_name}' is already processing a request"}],
                }
            )
            return

        try:
            # Determine if we are resuming the sub-agent from an interrupt.
            sub_agent_snapshot = self._get_sub_agent_snapshot(invocation_state, tool_use_id)
            if sub_agent_snapshot is not None:
                # Resume routes both the interrupt response and the interrupted turn as data,
                # carried in the parent's persisted interrupt record. It does not depend on the
                # orchestrator and sub-agent sharing an in-memory Interrupt object, so it works
                # even when both have been independently rebuilt from storage (e.g. a stateless
                # Lambda that recreates every agent each invocation).
                prompt = self._restore_from_snapshot(sub_agent_snapshot, invocation_state)
                logger.debug(
                    "tool_name=<%s>, tool_use_id=<%s> | resuming sub-agent from serialized interrupt snapshot",
                    self._tool_name,
                    tool_use_id,
                )
            elif not self._preserve_context:
                self._reset_agent_state(tool_use_id)

            logger.debug("tool_name=<%s>, tool_use_id=<%s> | invoking agent", self._tool_name, tool_use_id)

            result = None
            async for event in self._agent.stream_async(prompt):
                if "result" in event:
                    result = event["result"]
                else:
                    yield AgentAsToolStreamEvent(tool_use, event, self)

            if result is None:
                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [{"text": "Agent did not produce a result"}],
                    }
                )
                return

            # Propagate sub-agent interrupts to the parent agent. The snapshot lets the parent
            # persist everything required to resume this exact invocation, so the sub-agent does
            # not need its own session and does not accumulate history across turns.
            if result.stop_reason == "interrupt" and result.interrupts:
                yield ToolInterruptEvent(
                    tool_use, list(result.interrupts), sub_agent_snapshot=self._build_sub_agent_snapshot()
                )
                return

            if result.structured_output:
                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use_id,
                        "status": "success",
                        "content": [{"json": result.structured_output.model_dump()}],
                    }
                )
            else:
                yield ToolResultEvent(
                    {
                        "toolUseId": tool_use_id,
                        "status": "success",
                        "content": [{"text": str(result)}],
                    }
                )

        except Exception as e:
            logger.warning(
                "tool_name=<%s>, tool_use_id=<%s> | agent invocation failed: %s",
                self._tool_name,
                tool_use_id,
                e,
            )
            yield ToolResultEvent(
                {
                    "toolUseId": tool_use_id,
                    "status": "error",
                    "content": [{"text": f"Agent error: {e}"}],
                }
            )
        finally:
            self._lock.release()

    def _reset_agent_state(self, tool_use_id: str) -> None:
        """Reset the wrapped agent to its initial state.

        Restores messages and state to the values captured at construction time.
        This mirrors the pattern used by ``GraphNode.reset_executor_state()``.

        Args:
            tool_use_id: Tool use ID for logging context.
        """
        logger.debug(
            "tool_name=<%s>, tool_use_id=<%s> | resetting agent to initial state",
            self._tool_name,
            tool_use_id,
        )
        self._agent.messages = copy.deepcopy(self._initial_messages)
        self._agent.state = AgentState(self._initial_state.get())

    def _build_sub_agent_snapshot(self) -> dict[str, Any]:
        """Capture the minimal serializable state needed to resume this invocation.

        The snapshot is scoped to the single interrupted turn: the sub-agent's messages, its
        key/value state, and its interrupt state (which carries the original ``tool_use_message``
        and therefore the original ``toolUseId``). Restoring it reproduces the exact pending tool
        call so the recomputed interrupt id matches, without persisting the sub-agent's session or
        retaining history beyond this turn.

        Returns:
            Serializable snapshot of the interrupted invocation.
        """
        return {
            "messages": copy.deepcopy(self._agent.messages),
            "state": copy.deepcopy(self._agent.state.get()),
            "interrupt_state": self._agent._interrupt_state.to_dict(),
        }

    def _get_sub_agent_snapshot(self, invocation_state: dict[str, Any], tool_use_id: str) -> dict[str, Any] | None:
        """Return the sub-agent snapshot for this invocation, if the parent is resuming it.

        Args:
            invocation_state: Tool invocation context, populated by the event loop with
                ``_sub_agent_interrupt_resume`` when resuming from an interrupt.
            tool_use_id: The toolUseId of this agent-as-tool call, used to key the snapshot.

        Returns:
            The snapshot for this invocation, or ``None`` if this is not a sub-agent resume.
        """
        sub_agent_interrupt_resume = invocation_state.get("_sub_agent_interrupt_resume")
        if not sub_agent_interrupt_resume:
            return None
        snapshots = sub_agent_interrupt_resume.get("snapshots") or {}
        return cast("dict[str, Any] | None", snapshots.get(tool_use_id))

    def _restore_from_snapshot(
        self, snapshot: dict[str, Any], invocation_state: dict[str, Any]
    ) -> list[InterruptResponseContent]:
        """Overlay the interrupted turn onto a fresh sub-agent and build its resume prompt.

        Args:
            snapshot: Snapshot produced by ``_build_sub_agent_snapshot`` (possibly round-tripped
                through session serialization).
            invocation_state: Tool invocation context carrying the parent's resume responses
                under ``_sub_agent_interrupt_resume``.

        Returns:
            The interrupt responses destined for this sub-agent, ready to pass to ``stream_async``.
        """
        self._agent.messages = snapshot["messages"]
        self._agent.state = AgentState(snapshot["state"])
        self._agent._interrupt_state = _InterruptState.from_dict(snapshot["interrupt_state"])

        sub_agent_interrupt_resume = invocation_state.get("_sub_agent_interrupt_resume") or {}
        responses = sub_agent_interrupt_resume.get("responses") or []
        return [
            response
            for response in responses
            if response["interruptResponse"]["interruptId"] in self._agent._interrupt_state.interrupts
        ]

    @override
    def get_display_properties(self) -> dict[str, str]:
        """Get properties for UI display."""
        properties = super().get_display_properties()
        properties["Agent"] = getattr(self._agent, "name", "unknown")
        return properties
