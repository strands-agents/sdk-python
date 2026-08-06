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

_PARKED_TURNS_KEY = "sub_agent_continuations"
"""Key under which an orchestrator's interrupt context holds interrupted sub-agent turns, keyed by tool use id."""

_NAMESPACE_TAG = "v1:agent_as_tool:"
"""Reserved marker opening every namespaced sub-agent interrupt id.

Written by the SDK and never derived from model output, so a namespace prefix cannot collide with an
interrupt id the SDK generates itself - all of which open with the ``v1:`` scheme marker.
"""


def _namespace_prefix(tool_use_id: str) -> str:
    """Build the prefix that namespaces a sub-agent interrupt id to one agent-as-tool call.

    Two sub-agents invoked in the same turn can raise the same interrupt id, because each derives it
    from its own tool use id. The prefix keeps them distinct in the orchestrator's interrupt record and
    keeps the sub-agent-local id recoverable by stripping it back off.

    Tool use ids and interrupt ids are both model-derived, so the prefix defends against both ways one
    call's prefix can match another id: the tool use id is percent-encoded, because otherwise one
    containing the separator matches another call's ids, and the prefix opens with a reserved marker,
    because otherwise a tool use id of ``v1`` matches every interrupt the orchestrator raised itself.

    Args:
        tool_use_id: Tool use ID of the agent-as-tool call.

    Returns:
        Prefix, separator included, for interrupt IDs belonging to that call.
    """
    return f"{_NAMESPACE_TAG}{quote(tool_use_id, safe='')}:"


def _namespace_interrupts(tool_use_id: str, interrupts: list[Interrupt]) -> list[Interrupt]:
    """Copy sub-agent interrupts with their ids namespaced to one agent-as-tool call.

    Only the id changes: ``name`` and ``reason`` are what a human or an approval UI reads, so they are
    passed through untouched.

    Args:
        tool_use_id: Tool use ID of the agent-as-tool call.
        interrupts: Interrupts raised inside the sub-agent.

    Returns:
        Orchestrator-visible copies carrying namespaced ids.
    """
    prefix = _namespace_prefix(tool_use_id)
    return [
        Interrupt(id=f"{prefix}{interrupt.id}", name=interrupt.name, reason=interrupt.reason)
        for interrupt in interrupts
    ]


class _ParentCall:
    """The orchestrator's record of one agent-as-tool call.

    Binds the orchestrator running a call to that call's interrupt id namespace, so propagation and
    resume both read and write the orchestrator's persisted interrupt state in terms of *this* call:
    the interrupts it has pending, the answers addressed to it, and the sub-agent turn it parked.
    """

    def __init__(self, parent: Agent, tool_use_id: str) -> None:
        """Bind to an orchestrator and one of its tool calls."""
        self._parent = parent
        self._tool_use_id = tool_use_id
        self._prefix = _namespace_prefix(tool_use_id)

    @classmethod
    def resolve(cls, invocation_state: dict[str, Any], tool_use_id: str) -> _ParentCall | None:
        """Bind to the orchestrator running this tool call, if the invocation state carries one.

        A tool invoked directly rather than by an agent has no orchestrator, and cannot interrupt.
        """
        parent: Agent | None = invocation_state.get("agent")
        return cls(parent, tool_use_id) if parent is not None else None

    @property
    def tool_use_id(self) -> str:
        """Tool use ID of the call."""
        return self._tool_use_id

    @property
    def awaiting_resume(self) -> bool:
        """Whether the orchestrator is parked on an interrupt raised by this call.

        Keyed on the orchestrator rather than on the sub-agent, so a sub-agent shared with another
        caller cannot adopt a pending turn belonging to someone else.
        """
        if not self._parent._interrupt_state.activated:
            return False

        return any(interrupt_id.startswith(self._prefix) for interrupt_id in self._parent._interrupt_state.interrupts)

    def pending_interrupts(self) -> list[Interrupt]:
        """Get the namespaced interrupts the orchestrator holds for this call."""
        return [
            interrupt
            for interrupt_id, interrupt in self._parent._interrupt_state.interrupts.items()
            if interrupt_id.startswith(self._prefix)
        ]

    def responses(self) -> list[InterruptResponseContent]:
        """Map the answers addressed to this call back to sub-agent-local interrupt ids.

        The orchestrator persists the human's answers as data, so this survives a rehydration boundary
        where orchestrator and sub-agent no longer share an in-memory ``Interrupt``. An empty list means
        no answer belongs to this call, and the sub-agent re-raises its interrupt and stays pending.

        Returns:
            Interrupt response content blocks addressed to the sub-agent.
        """
        responses: list[InterruptResponseContent] = []
        for response in self._parent._interrupt_state.context.get("responses") or []:
            interrupt_id = response["interruptResponse"]["interruptId"]
            if interrupt_id.startswith(self._prefix):
                responses.append(
                    {
                        "interruptResponse": {
                            "interruptId": interrupt_id[len(self._prefix) :],
                            "response": response["interruptResponse"]["response"],
                        }
                    }
                )

        return responses

    def parked_turn(self) -> dict[str, Any] | None:
        """Get the interrupted sub-agent turn parked for this call, if there is one."""
        parked: dict[str, Any] = self._parent._interrupt_state.context.get(_PARKED_TURNS_KEY) or {}
        turn: dict[str, Any] | None = parked.get(self._tool_use_id)
        return turn

    def park_turn(self, turn: dict[str, Any]) -> None:
        """Park an interrupted sub-agent turn on the orchestrator's own interrupt record."""
        parked: dict[str, Any] = self._parent._interrupt_state.context.setdefault(_PARKED_TURNS_KEY, {})
        parked[self._tool_use_id] = turn

    def drop_parked_turn(self) -> None:
        """Free the parked turn once it has been reinstated."""
        parked: dict[str, Any] = self._parent._interrupt_state.context.get(_PARKED_TURNS_KEY) or {}
        parked.pop(self._tool_use_id, None)


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

                Interrupts raised inside the sub-agent resume automatically. When False, the
                orchestrator carries the sub-agent's interrupted turn, so the resume survives a
                process restart. When True the sub-agent owns its state: give it its own session
                manager if the resume has to survive a restart, otherwise the interrupt resumes
                only within the same process.
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
        parent_call = _ParentCall.resolve(invocation_state, tool_use_id)

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
            if parent_call is not None and parent_call.awaiting_resume:
                if not self._reinstate_turn(parent_call) and not self._agent._interrupt_state.activated:
                    pending = parent_call.pending_interrupts()
                    if parent_call.parked_turn() is not None and pending:
                        # The answer cannot be applied yet, but the turn is still parked, so raise the
                        # interrupt again instead of failing the call: the orchestrator parks it once more,
                        # keeping both the turn and the pending interrupt for another attempt. Failing here
                        # would end the orchestrator's turn, and the event loop clears the whole interrupt
                        # record when a turn ends - taking the parked turn and the human's answer with it.
                        logger.error(
                            "tool_name=<%s>, agent_name=<%s>, tool_use_id=<%s> | cannot apply the interrupt "
                            "response yet, raising the interrupt again so it survives to be answered once "
                            "more",
                            self._tool_name,
                            self._agent_name,
                            tool_use_id,
                        )
                        yield ToolInterruptEvent(tool_use, pending)
                        return

                    logger.error(
                        "tool_name=<%s>, agent_name=<%s>, tool_use_id=<%s> | cannot resume: the sub-agent's "
                        "interrupted turn is not available, so the interrupt response cannot be applied",
                        self._tool_name,
                        self._agent_name,
                        tool_use_id,
                    )
                    yield ToolResultEvent(
                        {
                            "toolUseId": tool_use_id,
                            "status": "error",
                            "content": [{"text": self._unresumable_message(parent_call)}],
                        }
                    )
                    return

                prompt = parent_call.responses()
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
                if parent_call is not None:
                    self._park_turn(parent_call)
                yield ToolInterruptEvent(tool_use, _namespace_interrupts(tool_use_id, list(result.interrupts)))
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

    @property
    def _agent_name(self) -> str:
        """Name of the wrapped agent, for logs and UI display."""
        return getattr(self._agent, "name", "unknown")

    def _unresumable_message(self, parent_call: _ParentCall) -> str:
        """Explain to the orchestrator's model why the interrupted turn could not be reinstated.

        The model decides what the user is told, and the human's approval is gone either way, so the
        message has to rule out reporting the guarded action as done.

        Args:
            parent_call: The orchestrator's record of this call.

        Returns:
            Text for the failed tool result.
        """
        if parent_call.parked_turn() is not None:
            return (
                f"Agent '{self._tool_name}' did NOT run and the human's response was NOT applied: its "
                "interrupted turn failed to load. Do not report the requested action as completed or "
                "successful. Tell the user it failed and ask them to respond again. See the logged error "
                "for the cause."
            )

        return (
            f"Agent '{self._tool_name}' did NOT run and the human's response was NOT applied: its "
            "interrupted turn did not survive the restart. A sub-agent used with preserve_context=True "
            "keeps its own state, so it needs its own session manager for that state to survive. Do not "
            "report the requested action as completed or successful; tell the user it failed."
        )

    def _park_turn(self, parent_call: _ParentCall) -> None:
        """Park the sub-agent's interrupted turn on the orchestrator so a later request can reinstate it.

        Only for ``preserve_context=False`` sub-agents: they are ephemeral by contract and are rejected at
        construction if they have a session manager, so the orchestrator carries the turn for them and
        Strands persists it with the orchestrator's own interrupt record. A sub-agent that preserves
        context owns its state and keeps its turn in its own session; without a session manager that turn
        is held only in memory, which is worth warning about now rather than when the answer arrives and
        cannot be applied.

        Args:
            parent_call: The orchestrator's record of this call.
        """
        if self._preserve_context:
            if getattr(self._agent, "_session_manager", None) is None:
                logger.warning(
                    "tool_name=<%s>, agent_name=<%s>, tool_use_id=<%s> | interrupted sub-agent uses "
                    "preserve_context=True with no session manager, so its interrupted turn is held only in "
                    "memory: the interrupt resumes in this process but not after a restart",
                    self._tool_name,
                    self._agent_name,
                    parent_call.tool_use_id,
                )
            return

        # Copied because the snapshot carries the sub-agent's live interrupt context by reference, and the
        # sub-agent keeps being used after this.
        parent_call.park_turn(copy.deepcopy(self._agent.take_snapshot(preset="session").to_dict()))
        logger.debug(
            "tool_name=<%s>, tool_use_id=<%s> | parked interrupted sub-agent turn for resume",
            self._tool_name,
            parent_call.tool_use_id,
        )

    def _reinstate_turn(self, parent_call: _ParentCall) -> bool:
        """Rebuild the sub-agent's interrupted turn from the turn the orchestrator parked.

        The turn is freed once it loads: a sub-agent that interrupts again parks a fresh one.

        Args:
            parent_call: The orchestrator's record of this call.

        Returns:
            True if the interrupted turn was reinstated onto the sub-agent.
        """
        turn = parent_call.parked_turn()
        if turn is None:
            return False

        try:
            self._agent.load_snapshot(Snapshot.from_dict(turn))
        except Exception as error:
            # Keep the parked turn: a load can fail for a reason a later attempt survives, such as a schema
            # version the running SDK does not accept yet, and this is the only copy of the turn the human
            # already answered. The caller re-raises the interrupt so the turn stays parked with it. A
            # failure after the first field was applied leaves the sub-agent mismatched; it is not invoked,
            # and an ephemeral sub-agent is reset on its next fresh call.
            logger.error(
                "tool_name=<%s>, agent_name=<%s>, tool_use_id=<%s> | failed to reinstate interrupted "
                "sub-agent turn: %s",
                self._tool_name,
                self._agent_name,
                parent_call.tool_use_id,
                error,
            )
            return False

        parent_call.drop_parked_turn()
        return True

    @override
    def get_display_properties(self) -> dict[str, str]:
        """Get properties for UI display."""
        properties = super().get_display_properties()
        properties["Agent"] = self._agent_name
        return properties
