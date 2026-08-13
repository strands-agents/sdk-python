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
from ..interrupt import Interrupt, _InterruptState
from ..types._events import AgentAsToolStreamEvent, ToolInterruptEvent, ToolResultEvent
from ..types._snapshot import Snapshot
from ..types.content import Messages
from ..types.exceptions import SnapshotException
from ..types.interrupt import InterruptResponseContent
from ..types.tools import AgentTool, ToolGenerator, ToolSpec, ToolUse

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)

_INTERRUPTED_TURNS_KEY = "sub_agent_interrupted_turns"
"""Key under which an orchestrator's interrupt context holds interrupted sub-agent turns, keyed by tool use ID."""

_NAMESPACE_TAG = "v1:agent_as_tool:"
"""Reserved marker opening every namespaced sub-agent interrupt ID.

Written by the SDK and never derived from model output, so a namespace prefix cannot collide with an
interrupt ID the SDK generates itself - all of which open with the ``v1:`` scheme marker.
"""


class _ParentCall:
    """The orchestrator's record of one agent-as-tool call.

    An interrupt raised inside a sub-agent has to reach the orchestrator's caller, and the human's
    response has to travel back down - across a boundary where either agent may have been rebuilt from
    storage, so the two no longer share an in-memory ``Interrupt``. Both directions therefore go through
    the orchestrator's own persisted interrupt state, and everything here is scoped to one tool call
    within it: the interrupts that call has pending, the responses addressed to it, and the interrupted
    sub-agent turn stored for it.

    Interrupt IDs are namespaced with a prefix derived from the tool use ID, because two sub-agents
    invoked in the same turn can raise the same ID - each derives it from its own tool use ID. Both are
    model-derived, so the prefix guards against either one matching another call's IDs: the tool use ID
    is percent-encoded, or one containing the separator would match, and the prefix opens with a
    reserved marker, or a tool use ID of ``v1`` would match every interrupt the orchestrator itself
    raised.
    """

    def __init__(self, parent: Agent, tool_use_id: str) -> None:
        """Bind to an orchestrator and one of its tool calls."""
        self._parent = parent
        self._tool_use_id = tool_use_id
        self._prefix = f"{_NAMESPACE_TAG}{quote(tool_use_id, safe='')}:"

    @property
    def tool_use_id(self) -> str:
        """Tool use ID of the call."""
        return self._tool_use_id

    @property
    def is_resuming(self) -> bool:
        """Whether the orchestrator is holding an interrupt raised by this call.

        Keyed on the orchestrator rather than on the sub-agent, so a sub-agent shared with another
        caller cannot adopt an interrupted turn belonging to someone else.
        """
        if not self._parent._interrupt_state.activated:
            return False

        return any(interrupt_id.startswith(self._prefix) for interrupt_id in self._parent._interrupt_state.interrupts)

    def namespace(self, interrupts: list[Interrupt]) -> list[Interrupt]:
        """Copy sub-agent interrupts with their IDs namespaced to this call.

        Only the ID changes: ``name`` and ``reason`` are what a human or an approval UI reads.

        Args:
            interrupts: Interrupts raised inside the sub-agent.

        Returns:
            Orchestrator-visible copies carrying namespaced IDs.
        """
        return [
            Interrupt(id=f"{self._prefix}{interrupt.id}", name=interrupt.name, reason=interrupt.reason)
            for interrupt in interrupts
        ]

    def pending_interrupts(self) -> list[Interrupt]:
        """Get the interrupts this call's stored turn is still waiting on.

        The orchestrator keeps every ID it has been handed until its own turn ends, including ones the
        sub-agent has since finished with, so the stored turn decides which are still live. Empty when
        no turn is stored, because then there is nothing to raise again.

        Returns:
            Interrupts that have to outlive a failed restore.
        """
        turn = self.interrupted_turn() or {}
        interrupt_state: dict[str, Any] = (turn.get("data") or {}).get("interrupt_state") or {}
        awaited_ids = set(interrupt_state.get("interrupts") or {})
        return [
            interrupt
            for interrupt_id, interrupt in self._parent._interrupt_state.interrupts.items()
            if interrupt_id.startswith(self._prefix) and interrupt_id[len(self._prefix) :] in awaited_ids
        ]

    def responses(self) -> list[InterruptResponseContent]:
        """Map the responses addressed to this call back to sub-agent-local interrupt IDs.

        The orchestrator persists the human's responses as data, so this survives a rehydration boundary
        where orchestrator and sub-agent no longer share an in-memory ``Interrupt``. An empty list means
        no response belongs to this call, and the sub-agent re-raises its interrupt and stays pending.

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

    def interrupted_turn(self) -> dict[str, Any] | None:
        """Get the interrupted sub-agent turn stored for this call, if there is one."""
        stored: dict[str, Any] = self._parent._interrupt_state.context.get(_INTERRUPTED_TURNS_KEY) or {}
        turn: dict[str, Any] | None = stored.get(self._tool_use_id)
        return turn

    def store_interrupted_turn(self, turn: dict[str, Any]) -> None:
        """Store an interrupted sub-agent turn on the orchestrator's own interrupt record."""
        stored: dict[str, Any] = self._parent._interrupt_state.context.setdefault(_INTERRUPTED_TURNS_KEY, {})
        stored[self._tool_use_id] = turn

    def clear_interrupted_turn(self) -> None:
        """Free the stored turn once it has been restored."""
        stored: dict[str, Any] = self._parent._interrupt_state.context.get(_INTERRUPTED_TURNS_KEY) or {}
        stored.pop(self._tool_use_id, None)


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
        # A tool invoked directly rather than by an agent has no orchestrator, and cannot interrupt.
        parent = invocation_state.get("agent")
        parent_call = _ParentCall(parent, tool_use_id) if parent is not None else None

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
            if parent_call is not None and parent_call.is_resuming:
                # The interrupted turn comes from one of two places: an ephemeral sub-agent's turn is
                # stored on the orchestrator, so restoring it reads the orchestrator's interrupt record,
                # while a preserve_context=True sub-agent restores its own from its session manager and
                # so arrives already activated.
                restored = self._restore_interrupted_turn(parent_call) or self._agent._interrupt_state.activated

                if not restored:
                    pending = parent_call.pending_interrupts()
                    if pending:
                        # The response cannot be applied yet, but the turn is still stored, so raise the
                        # interrupt again instead of failing the call: the orchestrator holds it once more,
                        # keeping both the turn and the pending interrupt for another attempt. Failing here
                        # would end the orchestrator's turn, and the event loop clears the whole interrupt
                        # record when a turn ends - taking the stored turn and the response with it.
                        logger.error(
                            "tool_name=<%s>, agent_name=<%s>, tool_use_id=<%s>, interrupt_ids=<%s> | the "
                            "interrupted turn could not be restored, so the response cannot be applied yet: "
                            "raising the interrupt again so it survives to be answered once more",
                            self._tool_name,
                            self._agent_name,
                            tool_use_id,
                            [interrupt.id for interrupt in pending],
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
                # Namespacing and storing both need the orchestrator the response will come back
                # through. A tool invoked directly has none, and cannot interrupt in the first place.
                interrupts = list(result.interrupts)
                if parent_call is not None:
                    self._store_interrupted_turn(parent_call)
                    interrupts = parent_call.namespace(interrupts)
                yield ToolInterruptEvent(tool_use, interrupts)
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

        Restores messages, state and interrupt state to the values captured at construction time. This
        mirrors the pattern used by ``GraphNode.reset_executor_state()``. Only reached on a fresh call,
        so any interrupt state the agent is still carrying belongs to a turn nobody is resuming.

        The reset is unconditional, so exposing one ``Agent`` instance as a tool to several
        orchestrators at once lets one orchestrator's call clear a turn another one is holding an
        interrupt for. That orchestrator still resumes correctly, because it restores the turn from its
        own interrupt record, but give each orchestrator its own sub-agent instance rather than sharing
        one.

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
        self._agent._interrupt_state = _InterruptState()

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
        if parent_call.interrupted_turn() is not None:
            cause = "its interrupted turn failed to load. See the logged error for the cause"
        elif self._preserve_context:
            cause = (
                "its interrupted turn did not survive the restart. A sub-agent used with "
                "preserve_context=True keeps its own state, so it needs its own session manager for that "
                "state to survive"
            )
        else:
            cause = "its interrupted turn is no longer available"

        return (
            f"Agent '{self._tool_name}' did NOT run and the human's response was NOT applied: {cause}. "
            "Do not report the requested action as completed or successful. Tell the user it failed and "
            "ask them to respond again."
        )

    def _store_interrupted_turn(self, parent_call: _ParentCall) -> None:
        """Store the sub-agent's interrupted turn on the orchestrator so a later request can restore it.

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
        parent_call.store_interrupted_turn(copy.deepcopy(self._agent.take_snapshot(preset="session").to_dict()))
        logger.debug(
            "tool_name=<%s>, tool_use_id=<%s> | stored interrupted sub-agent turn for resume",
            self._tool_name,
            parent_call.tool_use_id,
        )

    def _restore_interrupted_turn(self, parent_call: _ParentCall) -> bool:
        """Rebuild the sub-agent's interrupted turn from the turn the orchestrator stored.

        The turn is freed once it loads: a sub-agent that interrupts again stores a fresh one.

        Args:
            parent_call: The orchestrator's record of this call.

        Returns:
            True if the interrupted turn was restored onto the sub-agent.
        """
        turn = parent_call.interrupted_turn()
        if turn is None:
            return False

        try:
            self._agent.load_snapshot(Snapshot.from_dict(turn))
        except (SnapshotException, ValueError, KeyError, TypeError) as error:
            # The ways a turn written by another build of the SDK fails to load: an unsupported schema
            # version or scope, a state a component rejects, or a field that has since been renamed or
            # retyped. Anything else is a bug rather than stale data, and is left to propagate.
            #
            # Keep the stored turn: a load can fail for a reason a later attempt survives, and this is
            # the only copy of the turn the human already answered. The caller raises the interrupt again
            # so the turn stays stored with it. A failure after the first field was applied leaves the
            # sub-agent mismatched. It is not invoked unless it is itself still activated from an earlier
            # turn in this process, and an ephemeral sub-agent is reset on its next fresh call.
            logger.error(
                "tool_name=<%s>, agent_name=<%s>, tool_use_id=<%s> | failed to restore interrupted sub-agent turn: %s",
                self._tool_name,
                self._agent_name,
                parent_call.tool_use_id,
                error,
            )
            return False

        parent_call.clear_interrupted_turn()
        return True

    @override
    def get_display_properties(self) -> dict[str, str]:
        """Get properties for UI display."""
        properties = super().get_display_properties()
        properties["Agent"] = self._agent_name
        return properties
