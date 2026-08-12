"""AgentDelegation plugin — enforces delegation semantics for tool routing.

When a tool is configured with ``delegate=True``, this plugin ensures:

1. The delegation tool is the only tool called in the turn (single-call constraint)
2. The agent loop exits immediately after a successful delegation (via end_turn)
3. The AgentResult is transformed with the delegation tool's content as the last message
4. Streaming events from the delegate agent are surfaced natively in the parent stream

The final delegation message is produced in middleware after the core loop exits.
``MessageAddedEvent`` subscribers will see the end_turn placeholder followed by a second
event with the real delegation content (no session manager), or no event for the real
content at all (session manager attached).
"""

from __future__ import annotations

import json as json_module
import logging
import weakref
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._middleware.stages import AgentStreamContext, AgentStreamStage, ExecuteToolContext, ExecuteToolStage
from .._middleware.types import MiddlewareNext
from ..hooks import (
    AfterToolCallEvent,
    AfterToolsEvent,
    BeforeModelCallEvent,
    BeforeToolsEvent,
    HookOrder,
    MessageAddedEvent,
)
from ..plugins import Plugin
from ..types._events import AgentAsToolStreamEvent, EventLoopStopEvent, ToolResultEvent, TypedEvent
from ..types.content import ContentBlock, _ensure_tracking_id
from ..types.content import Message as MessageType
from ._agent_as_tool import _AgentAsTool

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)


@dataclass
class _DelegationState:
    """Per-agent state tracked across the delegation lifecycle within a single invocation."""

    tool_use_count: int = 0
    """Number of tool use blocks in the current batch."""

    tool_use_id: str | None = None
    """Tool use ID of the delegation tool that succeeded."""

    end_turn_via_delegation: bool = False
    """Whether this plugin ended turn."""


def _is_delegation_tool(agent: Agent, tool_name: str) -> bool:
    """Check whether a tool registered on the agent is a delegation AgentAsTool."""
    tool = agent.tool_registry.registry.get(tool_name) or agent.tool_registry.dynamic_tools.get(tool_name)
    return isinstance(tool, _AgentAsTool) and tool.delegate


def _to_content_blocks(tool_result: dict) -> list[ContentBlock]:
    """Convert ToolResult content into ContentBlocks for an assistant message.

    JSON blocks are serialized to text (matching TS's ``toContentBlocks``).
    """
    raw_content = tool_result.get("content", [])
    result: list[ContentBlock] = []
    for block in raw_content:
        if "text" in block:
            result.append({"text": block["text"]})
        elif "json" in block:
            result.append({"text": json_module.dumps(block["json"])})
        else:
            result.append(block)
    return result


def _find_delegation_result(messages: list, tool_use_id: str) -> dict | None:
    """Find the delegation tool's successful result in the agent messages.

    Searches backwards for the last user message containing a successful tool
    result matching the given tool_use_id. Returns None if not found or errored.
    """
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "user":
            continue
        for block in msg.get("content", []):
            if not isinstance(block, dict) or "toolResult" not in block:
                continue
            tool_result = block["toolResult"]
            if tool_result.get("toolUseId") == tool_use_id:
                if tool_result.get("status") == "error":
                    return None
                return dict(tool_result)
    return None


class AgentDelegation(Plugin):
    """Plugin that enforces delegation semantics for tool routing.

    Automatically registered on every agent. Acts as a no-op when no delegation tools fire.
    Implements single-call constraint, early loop exit, and result transformation.

    Example:
        ```python
        from strands import Agent

        specialist = Agent(name="specialist", description="Handles billing")
        orchestrator = Agent(
            tools=[specialist.as_tool(delegate=True)],
            # AgentDelegation is auto-registered — no manual setup needed
        )
        ```
    """

    name = "strands:agent-delegation"

    def __init__(self) -> None:
        super().__init__()
        self._state: weakref.WeakKeyDictionary[Agent, _DelegationState] = weakref.WeakKeyDictionary()

    def init_agent(self, agent: Agent) -> None:
        """Register hooks and middleware for delegation enforcement."""
        # Delegation is incompatible with stateful models — early exit would leave
        # an unclosed function call on the server.
        if agent.model.stateful:
            has_delegation_tool = any(
                isinstance(tool, _AgentAsTool) and tool.delegate for tool in agent.tool_registry.registry.values()
            )
            if has_delegation_tool:
                raise ValueError(
                    "Delegation tools (delegate=True) are not supported with stateful models. "
                    "Stateful models manage conversation state server-side, and delegation's early loop exit "
                    "would leave unclosed function calls on the server."
                )

        agent.add_hook(self._on_before_tools, BeforeToolsEvent)
        agent.add_hook(self._on_after_tool_call, AfterToolCallEvent)
        agent.add_hook(self._on_after_tools, AfterToolsEvent, order=HookOrder.SDK_LAST)
        agent.add_hook(self._on_before_model_call, BeforeModelCallEvent)

        agent._middleware_registry.add_middleware(ExecuteToolStage, self._handle_tool_execution)
        agent._middleware_registry.add_middleware(AgentStreamStage, self._handle_stream)

    # --- Hooks ---

    def _on_before_tools(self, event: BeforeToolsEvent) -> None:
        """Initialize batch state; cancel if a delegate is mixed with other tools."""
        agent = event.agent
        if agent.model.stateful:
            return

        message = event.message
        content = message.get("content", [])
        tool_use_blocks = [block for block in content if isinstance(block, dict) and "toolUse" in block]

        self._state[agent] = _DelegationState(tool_use_count=len(tool_use_blocks))

        has_delegation = any(_is_delegation_tool(agent, block["toolUse"]["name"]) for block in tool_use_blocks)
        if has_delegation and len(tool_use_blocks) > 1:
            event.cancel = (
                "This tool call was not executed. A delegation tool must be the only "
                "tool called in a turn. Retry with a single delegation tool call or "
                "use only non-delegation tools."
            )

    def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        """Clear stale delegation state when the loop continues past a delegation batch."""
        self._state.pop(event.agent, None)

    def _on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        """Mark tool_use_id on delegation success; clear on error or retry-swap."""
        if event.agent.model.stateful:
            return

        if not isinstance(event.selected_tool, _AgentAsTool) or not event.selected_tool.delegate:
            state = self._state.get(event.agent)
            if state is not None and state.tool_use_id == event.tool_use.get("toolUseId"):
                state.tool_use_id = None
            return

        state = self._state.get(event.agent)
        if state is None:
            return

        if event.result.get("status") == "error":
            state.tool_use_id = None
            return

        state.tool_use_id = event.tool_use.get("toolUseId")

    def _on_after_tools(self, event: AfterToolsEvent) -> None:
        """Set end_turn when delegation succeeded and the result is still valid.

        This hook runs at ``HookOrder.SDK_LAST`` (100). If a hook with a higher numeric order
        mutates a tool result's status from success to error, the end_turn signal has already
        been committed and the loop will still exit; ``_handle_stream`` re-verifies the result
        as defence in depth. No SDK or vended plugin registers AfterToolsEvent hooks above
        SDK_LAST, and mutating a committed tool result's status there is not a supported pattern.
        """
        if event.agent.model.stateful:
            return

        state = self._state.get(event.agent)
        if state is None or not state.tool_use_id:
            return

        # Verify the tool result is still successful in the committed message.
        message = event.message
        content = message.get("content", [])
        result_block = None
        for block in content:
            if not isinstance(block, dict) or "toolResult" not in block:
                continue
            tool_result = block["toolResult"]
            if tool_result.get("toolUseId") == state.tool_use_id:
                result_block = tool_result
                break

        if result_block is None or result_block.get("status") == "error":
            state.tool_use_id = None
            return

        # Skip delegation when the parent expects structured output.
        has_structured_output = any(
            getattr(t, "tool_type", None) == "structured_output"
            for t in event.agent.tool_registry.dynamic_tools.values()
        )
        if has_structured_output:
            logger.debug(
                "tool_use_id=<%s> | parent requires structured output, skipping delegation",
                state.tool_use_id,
            )
            return

        event.end_turn = True
        state.end_turn_via_delegation = True

    # --- Middleware ---

    async def _handle_tool_execution(
        self,
        context: ExecuteToolContext,
        next_fn: MiddlewareNext,
    ) -> AsyncGenerator[TypedEvent, None]:
        """ExecuteToolStage middleware: enforce delegation constraints and unwrap events."""
        # Non-delegation tools pass through unchanged.
        if not isinstance(context.tool, _AgentAsTool) or not context.tool.delegate:
            async for event in next_fn(context):
                yield event
            return

        # Stateful model: skip delegation, execute as a normal tool.
        if context.agent.model.stateful:
            logger.debug(
                "tool_use_id=<%s> | stateful model, skipping delegation and running as a normal tool",
                context.tool_use["toolUseId"],
            )
            async for event in next_fn(context):
                yield event
            return

        # Enforce single-call constraint.
        state = self._state.get(context.agent)
        if state and state.tool_use_count > 1:
            yield ToolResultEvent(
                {
                    "toolUseId": context.tool_use["toolUseId"],
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                "Delegation failed: a delegation tool must be the only tool "
                                "called in a turn. Retry with a single delegation tool call "
                                "or use only non-delegation tools."
                            )
                        }
                    ],
                }
            )
            return

        # Stream the delegation tool, unwrapping inner agent events as native events.
        async for event in next_fn(context):
            if isinstance(event, ToolResultEvent):
                yield event
            elif isinstance(event, AgentAsToolStreamEvent):
                inner_data = event.get("tool_stream_event", {}).get("data")
                if inner_data is not None:
                    yield TypedEvent(inner_data)
                else:
                    yield event
            else:
                yield event

    async def _handle_stream(
        self,
        context: AgentStreamContext,
        next_fn: MiddlewareNext,
    ) -> AsyncGenerator[TypedEvent, None]:
        """AgentStreamStage middleware: replace the terminal stop with delegation content.

        Events before the first stop are yielded immediately. Once a stop appears,
        the tail is buffered and replayed with the stop replaced (or unchanged).
        """
        self._state.pop(context.agent, None)

        tail_buffer: list[TypedEvent] = []
        buffering = False

        try:
            async for event in next_fn(context):
                if buffering:
                    tail_buffer.append(event)
                elif isinstance(event, EventLoopStopEvent):
                    buffering = True
                    tail_buffer.append(event)
                else:
                    yield event
        except Exception:
            self._state.pop(context.agent, None)
            for event in tail_buffer:
                yield event
            raise

        state = self._state.pop(context.agent, None)

        if state is None or not state.tool_use_id or not state.end_turn_via_delegation:
            for event in tail_buffer:
                yield event
            return

        # Re-verify the delegation result is still successful in committed history.
        messages = context.agent.messages
        tool_result = _find_delegation_result(messages, state.tool_use_id)
        if tool_result is None:
            logger.debug(
                "tool_use_id=<%s> | delegation result no longer successful in history, "
                "skipping delegation transformation",
                state.tool_use_id,
            )
            for event in tail_buffer:
                yield event
            return

        delegation_content = _to_content_blocks(tool_result)
        delegation_message: MessageType = {"role": "assistant", "content": delegation_content}
        _ensure_tracking_id(delegation_message)

        # Replace the placeholder message the event loop persisted.
        session_manager = getattr(context.agent, "_session_manager", None)
        if messages and messages[-1].get("role") == "assistant":
            messages[-1] = delegation_message
            if session_manager is not None:
                session_manager.redact_latest_message(delegation_message, context.agent)
            else:
                await context.agent.hooks.invoke_callbacks_async(
                    MessageAddedEvent(agent=context.agent, message=delegation_message)
                )
        else:
            messages.append(delegation_message)
            await context.agent.hooks.invoke_callbacks_async(
                MessageAddedEvent(agent=context.agent, message=delegation_message)
            )

        # Replay tail with stop replaced by delegation terminal.
        for event in tail_buffer:
            if isinstance(event, EventLoopStopEvent):
                yield EventLoopStopEvent(
                    "end_turn",
                    delegation_message,
                    context.agent.event_loop_metrics,
                    context.invocation_state.get("request_state", {}),
                )
            else:
                yield event
