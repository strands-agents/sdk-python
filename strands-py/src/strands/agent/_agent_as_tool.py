"""Agent-as-tool adapter.

This module provides the _AgentAsTool class that wraps an Agent as a tool
so it can be passed to another agent's tool list.
"""

from __future__ import annotations

import copy
import logging
import threading
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from typing_extensions import override

from ..agent.state import AgentState
from ..interrupt import Interrupt
from ..types._events import AgentAsToolStreamEvent, ToolInterruptEvent, ToolResultEvent
from ..types._snapshot import Snapshot
from ..types.content import Messages
from ..types.interrupt import InterruptResponseContent
from ..types.tools import AgentTool, ToolGenerator, ToolSpec, ToolUse

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)

_CONTINUATIONS_KEY = "sub_agent_continuations"
"""Key under which an orchestrator's interrupt context holds interrupted sub-agent turns, keyed by tool use id."""


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
            if self._parent_awaiting_resume(invocation_state, tool_use_id):
                restored = self._restore_continuation(invocation_state, tool_use_id)
                if not restored and not self._is_sub_agent_interrupted():
                    logger.error(
                        "tool_name=<%s>, tool_use_id=<%s> | cannot resume: the sub-agent's interrupted turn "
                        "is not available, so the interrupt response cannot be applied",
                        self._tool_name,
                        tool_use_id,
                    )
                    yield ToolResultEvent(
                        {
                            "toolUseId": tool_use_id,
                            "status": "error",
                            "content": [{"text": self._unresumable_message(invocation_state, tool_use_id)}],
                        }
                    )
                    return

                prompt = self._interrupt_responses(invocation_state, tool_use_id)
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
                interrupts = list(result.interrupts)
                self._stash_continuation(invocation_state, tool_use_id)
                yield ToolInterruptEvent(tool_use, self._namespace_interrupts(tool_use_id, interrupts))
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

        The reset is unconditional, so exposing one ``Agent`` instance as a tool to several
        orchestrators at once lets one orchestrator's call clear a turn another one has parked on an
        interrupt. The parked orchestrator still resumes correctly, because it reinstates the turn
        from its own interrupt record, but the other call fails; give each orchestrator its own
        sub-agent instance instead of sharing one.

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

    def _is_sub_agent_interrupted(self) -> bool:
        """Check whether the wrapped agent is in an activated interrupt state."""
        return self._agent._interrupt_state.activated

    @staticmethod
    def _namespace_prefix(tool_use_id: str) -> str:
        """Build the prefix that namespaces a sub-agent interrupt id to this tool call.

        The tool use ID is percent-encoded so the prefix cannot contain the separator. Tool use IDs and
        interrupt IDs are both model-derived strings: with a raw prefix, a tool use ID ending in the
        separator plus the start of another call's interrupt IDs would match that call's answers and
        route them to the wrong sub-agent.

        Args:
            tool_use_id: Tool use ID of this agent-as-tool call.

        Returns:
            Prefix, separator included, for interrupt IDs belonging to this call.
        """
        return f"{quote(tool_use_id, safe='')}:"

    @staticmethod
    def _namespace_interrupts(tool_use_id: str, interrupts: list[Interrupt]) -> list[Interrupt]:
        """Copy sub-agent interrupts with their ids namespaced by this tool call.

        A sub-agent derives an interrupt id from its own toolUseId, so two sub-agents invoked in the
        same turn can raise the same id. Namespacing keeps them distinct at the orchestrator level and
        keeps the sub-agent-local id recoverable by stripping the prefix.

        Args:
            tool_use_id: Tool use ID of this agent-as-tool call.
            interrupts: Interrupts raised inside the sub-agent.

        Returns:
            Orchestrator-visible copies carrying namespaced ids.
        """
        prefix = _AgentAsTool._namespace_prefix(tool_use_id)
        return [
            Interrupt(id=f"{prefix}{interrupt.id}", name=interrupt.name, reason=interrupt.reason)
            for interrupt in interrupts
        ]

    @staticmethod
    def _parent_agent(invocation_state: dict[str, Any]) -> Agent | None:
        """Get the orchestrator running this tool call, if the invocation state carries one."""
        parent: Agent | None = invocation_state.get("agent")
        return parent

    @staticmethod
    def _parent_awaiting_resume(invocation_state: dict[str, Any], tool_use_id: str) -> bool:
        """Check whether the orchestrator is waiting on an interrupt raised by this tool call.

        Keyed on the orchestrator rather than on the sub-agent so a sub-agent that shares a session
        with another caller cannot adopt a pending turn that belongs to someone else.

        Args:
            invocation_state: Context for the tool invocation.
            tool_use_id: Tool use ID of this agent-as-tool call.

        Returns:
            True if the orchestrator holds an interrupt raised by this call.
        """
        parent = _AgentAsTool._parent_agent(invocation_state)
        if parent is None or not parent._interrupt_state.activated:
            return False

        prefix = _AgentAsTool._namespace_prefix(tool_use_id)
        return any(interrupt_id.startswith(prefix) for interrupt_id in parent._interrupt_state.interrupts)

    @staticmethod
    def _interrupt_responses(invocation_state: dict[str, Any], tool_use_id: str) -> list[InterruptResponseContent]:
        """Map the orchestrator's interrupt responses back to sub-agent-local ids.

        The orchestrator persists the human's answers as data, so this survives a rehydration
        boundary where orchestrator and sub-agent no longer share an in-memory ``Interrupt``. An
        empty list means none of the answers belong to this call, and the sub-agent re-raises its
        interrupt and stays pending.

        Args:
            invocation_state: Context for the tool invocation.
            tool_use_id: Tool use ID of this agent-as-tool call.

        Returns:
            Interrupt response content blocks addressed to the sub-agent.
        """
        parent = _AgentAsTool._parent_agent(invocation_state)
        if parent is None:
            return []

        prefix = _AgentAsTool._namespace_prefix(tool_use_id)
        responses: list[InterruptResponseContent] = []
        for response in parent._interrupt_state.context.get("responses") or []:
            interrupt_id = response["interruptResponse"]["interruptId"]
            if interrupt_id.startswith(prefix):
                responses.append(
                    {
                        "interruptResponse": {
                            "interruptId": interrupt_id[len(prefix) :],
                            "response": response["interruptResponse"]["response"],
                        }
                    }
                )

        return responses

    @staticmethod
    def _stored_continuation(invocation_state: dict[str, Any], tool_use_id: str) -> dict[str, Any] | None:
        """Get the interrupted turn the orchestrator holds for this tool call, if there is one."""
        parent = _AgentAsTool._parent_agent(invocation_state)
        if parent is None:
            return None

        continuations: dict[str, Any] = parent._interrupt_state.context.get(_CONTINUATIONS_KEY) or {}
        continuation: dict[str, Any] | None = continuations.get(tool_use_id)
        return continuation

    def _unresumable_message(self, invocation_state: dict[str, Any], tool_use_id: str) -> str:
        """Explain why the interrupted turn could not be reinstated, for the failed tool result."""
        if self._stored_continuation(invocation_state, tool_use_id) is not None:
            return (
                f"Agent '{self._tool_name}' could not resume its interrupted turn: the stored turn failed to "
                "load, so the interrupt response was not applied. See the logged error for the cause."
            )

        return (
            f"Agent '{self._tool_name}' could not resume its interrupted turn. A sub-agent used with "
            "preserve_context=True keeps its own state, so it needs a session manager for that state to "
            "survive rehydration."
        )

    def _stash_continuation(self, invocation_state: dict[str, Any], tool_use_id: str) -> None:
        """Store the sub-agent's interrupted turn in the orchestrator's interrupt context.

        Only for ``preserve_context=False`` sub-agents: they are ephemeral by contract and are rejected
        at construction if they have a session manager, so the orchestrator holds the turn for them and
        Strands persists it
        with the orchestrator's own interrupt record. A sub-agent that preserves context owns its
        state and keeps it in its own session.

        Args:
            invocation_state: Context for the tool invocation.
            tool_use_id: Tool use ID of this agent-as-tool call.
        """
        if self._preserve_context:
            return

        parent = self._parent_agent(invocation_state)
        if parent is None:
            return

        continuations: dict[str, Any] = parent._interrupt_state.context.setdefault(_CONTINUATIONS_KEY, {})
        continuations[tool_use_id] = self._agent.take_snapshot(preset="session").to_dict()
        logger.debug(
            "tool_name=<%s>, tool_use_id=<%s> | stored interrupted sub-agent turn for resume",
            self._tool_name,
            tool_use_id,
        )

    def _restore_continuation(self, invocation_state: dict[str, Any], tool_use_id: str) -> bool:
        """Rebuild the sub-agent's interrupted turn from the orchestrator's interrupt context.

        The turn is consumed: a sub-agent that interrupts again stores a fresh one.

        Args:
            invocation_state: Context for the tool invocation.
            tool_use_id: Tool use ID of this agent-as-tool call.

        Returns:
            True if the interrupted turn was restored onto the sub-agent.
        """
        parent = self._parent_agent(invocation_state)
        if parent is None:
            return False

        continuation = self._stored_continuation(invocation_state, tool_use_id)
        if continuation is None:
            return False

        try:
            self._agent.load_snapshot(Snapshot.from_dict(continuation))
        except Exception as error:
            # Leave the stored turn in place: a load can fail for a reason a later attempt survives,
            # such as a schema version the running SDK does not accept yet, and dropping it here would
            # destroy the only copy of the turn the human already answered. A failure after the first
            # field has been applied leaves the sub-agent itself mismatched; the caller reports the
            # error without invoking it, and an ephemeral sub-agent is reset on its next fresh call.
            logger.error(
                "tool_name=<%s>, tool_use_id=<%s> | failed to restore interrupted sub-agent turn: %s",
                self._tool_name,
                tool_use_id,
                error,
            )
            return False

        parent._interrupt_state.context[_CONTINUATIONS_KEY].pop(tool_use_id, None)
        return True

    @override
    def get_display_properties(self) -> dict[str, str]:
        """Get properties for UI display."""
        properties = super().get_display_properties()
        properties["Agent"] = getattr(self._agent, "name", "unknown")
        return properties
