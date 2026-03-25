"""Agent-as-tool adapter.

This module provides the AgentAsTool class that wraps an Agent (or any AgentBase) as a tool
so it can be passed to another agent's tool list.
"""

import copy
import logging
import threading
from typing import Any

from typing_extensions import override

from ..agent.state import AgentState
from ..types._events import AgentAsToolStreamEvent, ToolInterruptEvent, ToolResultEvent
from ..types.content import Messages
from ..types.interrupt import InterruptResponseContent
from ..types.tools import AgentTool, ToolGenerator, ToolSpec, ToolUse
from .base import AgentBase

logger = logging.getLogger(__name__)


class AgentAsTool(AgentTool):
    """Adapter that exposes an Agent as a tool for use by other agents.

    The tool accepts a single ``input`` string parameter, invokes the wrapped
    agent, and returns the text response.

    Example:
        ```python
        from strands import Agent
        from strands.agent import AgentAsTool

        researcher = Agent(name="researcher", description="Finds information")

        # Use directly
        tool = AgentAsTool(researcher, name="researcher", description="Finds information")

        # Or via convenience method
        tool = researcher.as_tool()

        # Start each invocation with a fresh conversation
        tool = researcher.as_tool(preserve_context=False)

        writer = Agent(name="writer", tools=[tool])
        writer("Write about AI agents")
        ```
    """

    def __init__(
        self,
        agent: AgentBase,
        *,
        name: str,
        description: str,
        preserve_context: bool = False,
    ) -> None:
        r"""Initialize the agent-as-tool adapter.

        Args:
            agent: The agent to wrap as a tool.
            name: Tool name. Must match the pattern ``[a-zA-Z0-9_\\-]{1,64}``.
            description: Tool description.
            preserve_context: Whether to preserve the agent's conversation history across
                invocations. When False, the agent's messages and state are reset to the
                values they had at construction time before each call, ensuring every
                invocation starts from the same baseline regardless of any external
                interactions with the agent. Defaults to False. Only effective when the
                wrapped agent exposes a mutable ``messages`` list and/or an ``AgentState``
                (e.g. ``strands.agent.Agent``).
        """
        super().__init__()
        self._agent = agent
        self._tool_name = name
        self._description = description
        self._preserve_context = preserve_context

        # When preserve_context=False, we snapshot the agent's initial state so we can
        # restore it before each invocation. This mirrors GraphNode.reset_executor_state().
        # We require an Agent instance for this since AgentBase doesn't guarantee
        # messages/state attributes.
        self._initial_messages: Messages = []
        self._initial_state: AgentState = AgentState()
        # Serialize access so _reset_agent_state + stream_async are atomic.
        # threading.Lock (not asyncio.Lock) because run_async() may create
        # separate event loops in different threads.
        self._lock = threading.Lock()

        if not preserve_context:
            from .agent import Agent

            if not isinstance(agent, Agent):
                raise TypeError(f"preserve_context=False requires an Agent instance, got {type(agent).__name__}")
            self._initial_messages = copy.deepcopy(agent.messages)
            self._initial_state = AgentState(agent.state.get())

    @property
    def agent(self) -> AgentBase:
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
            if self._is_sub_agent_interrupted():
                prompt = self._build_interrupt_responses()
                logger.debug(
                    "tool_name=<%s>, tool_use_id=<%s> | resuming sub-agent from interrupt",
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

            # Propagate sub-agent interrupts to the parent agent.
            if result.stop_reason == "interrupt" and result.interrupts:
                yield ToolInterruptEvent(tool_use, list(result.interrupts))
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
        from .agent import Agent

        # isinstance narrows the type for mypy; __init__ guarantees this when preserve_context=False
        if not isinstance(self._agent, Agent):
            return

        logger.debug(
            "tool_name=<%s>, tool_use_id=<%s> | resetting agent to initial state",
            self._tool_name,
            tool_use_id,
        )
        self._agent.messages = copy.deepcopy(self._initial_messages)
        self._agent.state = AgentState(self._initial_state.get())

    def _is_sub_agent_interrupted(self) -> bool:
        """Check whether the wrapped agent is in an activated interrupt state."""
        interrupt_state = getattr(self._agent, "_interrupt_state", None)
        return interrupt_state is not None and interrupt_state.activated

    def _build_interrupt_responses(self) -> list[InterruptResponseContent]:
        """Build interrupt response payloads from the sub-agent's interrupt state.

        The parent agent's ``_interrupt_state.resume()`` sets ``.response`` on the shared
        ``Interrupt`` objects (registered by the executor), so we re-package them in the
        format expected by ``Agent.stream_async``.

        Returns:
            List of interrupt response content blocks for resuming the sub-agent.
        """
        interrupt_state = getattr(self._agent, "_interrupt_state", None)
        if interrupt_state is None:
            return []

        return [
            {"interruptResponse": {"interruptId": interrupt.id, "response": interrupt.response}}
            for interrupt in interrupt_state.interrupts.values()
            if interrupt.response is not None
        ]

    @override
    def get_display_properties(self) -> dict[str, str]:
        """Get properties for UI display."""
        properties = super().get_display_properties()
        properties["Agent"] = getattr(self._agent, "name", "unknown")
        return properties
