"""Experimental governed child-agent tool."""

from __future__ import annotations

import asyncio
import weakref
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import NotRequired, TypedDict, override

from ....hooks import BeforeToolCallEvent, BeforeToolsEvent
from ....interrupt import Interrupt
from ....types._events import ToolInterruptEvent, ToolResultEvent
from ....types.interrupt import InterruptResponseContent
from ....types.tools import AgentTool, ToolGenerator, ToolResult, ToolSpec, ToolUse

if TYPE_CHECKING:
    from ....agent import Agent
    from ....agent.agent_result import AgentResult

_USE_AGENT_TOOL_NAME = "use_agent"
_INTERNAL_INVOCATION_STATE_KEYS = (
    "agent",
    "event_loop_cycle_id",
    "event_loop_cycle_span",
    "event_loop_cycle_trace",
    "event_loop_parent_cycle_id",
    "messages",
    "model",
    "request_state",
    "system_prompt",
    "tool_config",
)


class _UseAgentInput(TypedDict):
    task: str
    instructions: NotRequired[str]
    tools: NotRequired[list[str]]


class UseAgentLimits(TypedDict, total=False):
    """Developer-controlled child execution limits."""

    turns: int
    total_tokens: int
    depth: int
    timeout_seconds: int


_DEFAULT_LIMITS: dict[str, int] = {
    "turns": 50,
    "total_tokens": 100_000,
    "depth": 3,
    "timeout_seconds": 300,
}


@dataclass
class _PendingChild:
    child: Agent
    task: str
    parent_interrupt_state: object
    remaining_seconds: float
    remaining_turns: int
    remaining_total_tokens: int
    child_invocation_state: dict[str, Any]
    interrupt_generation: int = 0
    interrupts: tuple[tuple[str, str], ...] = ()


class _UseAgentTool(AgentTool):
    """Runs bounded tasks in fresh child agents."""

    def __init__(
        self,
        *,
        limits: UseAgentLimits | None,
    ) -> None:
        super().__init__()
        self._limits = _resolve_limits(limits)
        self._pending: weakref.WeakKeyDictionary[Agent, dict[str, _PendingChild]] = weakref.WeakKeyDictionary()
        self._depths: weakref.WeakKeyDictionary[Agent, int] = weakref.WeakKeyDictionary()

    @property
    @override
    def tool_name(self) -> str:
        """Get the tool name."""
        return _USE_AGENT_TOOL_NAME

    @property
    @override
    def tool_spec(self) -> ToolSpec:
        """Get the model-facing tool specification."""
        return {
            "name": self.tool_name,
            "description": (
                "Runs a task in a fresh child agent. The child receives only the exact parent tools named in tools; "
                "omit tools for no child tools."
            ),
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Task for the child agent.",
                        },
                        "instructions": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Optional child-specific instructions.",
                        },
                        "tools": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "description": "Exact parent tool names to grant. Omit for no tools.",
                        },
                    },
                    "required": ["task"],
                    "additionalProperties": False,
                }
            },
        }

    @property
    @override
    def tool_type(self) -> str:
        """Get the tool type."""
        return "function"

    @override
    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> ToolGenerator:
        """Run or resume a child agent."""
        from ....agent import Agent

        tool_use_id = tool_use["toolUseId"]
        parent = invocation_state.get("agent")
        if not isinstance(parent, Agent):
            yield _result_event(tool_use_id, {"status": "failed", "error": "use_agent requires a local Agent context"})
            return

        pending = self._pending.get(parent, {}).get(tool_use_id)
        try:
            interrupt_state = parent._interrupt_state
            restored = (
                pending is not None
                and pending.parent_interrupt_state is not interrupt_state
                or pending is None
                and any(interrupt_id.startswith(f"{tool_use_id}:") for interrupt_id in interrupt_state.interrupts)
            )
            if restored:
                raise RuntimeError(
                    "use_agent cannot resume an interrupted child after the parent or tool instance was restored"
                )
            deadline = monotonic() + (
                pending.remaining_seconds if pending is not None else self._limits["timeout_seconds"]
            )
            child_state = pending or self._create_child(
                parent,
                tool_use["input"],
                {key: value for key, value in invocation_state.items() if key not in _INTERNAL_INVOCATION_STATE_KEYS},
            )
            stop_reason = None
            if pending is not None:
                if child_state.remaining_turns <= 0:
                    stop_reason = "limit_turns"
                elif child_state.remaining_total_tokens <= 0:
                    stop_reason = "limit_total_tokens"
            if stop_reason is not None:
                self._pending.get(parent, {}).pop(tool_use_id, None)
                yield _result_event(tool_use_id, _stopped_result(stop_reason))
                return

            if pending is None:
                prompt: str | list[InterruptResponseContent] = child_state.task
            else:
                prompt = _interrupt_responses(parent, child_state.interrupts)

            result = await self._run_child(
                parent,
                child_state,
                prompt,
                deadline,
            )
            if result.stop_reason == "interrupt" and result.interrupts:
                invocation = result.metrics.latest_agent_invocation
                if invocation is not None:
                    child_state.remaining_turns -= len(invocation.cycles)
                    child_state.remaining_total_tokens -= invocation.usage.get("totalTokens", 0)
                child_state.remaining_seconds = max(0, deadline - monotonic())
                child_state.interrupt_generation += 1
                child_state.interrupts = tuple(
                    (
                        interrupt.id,
                        f"{tool_use_id}:{child_state.interrupt_generation}:{interrupt.id}",
                    )
                    for interrupt in result.interrupts
                )
                self._pending.setdefault(parent, {})[tool_use_id] = child_state
                yield ToolInterruptEvent(
                    tool_use,
                    [
                        Interrupt(
                            id=outward_id,
                            name=interrupt.name,
                            reason=interrupt.reason,
                        )
                        for interrupt, (_, outward_id) in zip(result.interrupts, child_state.interrupts, strict=True)
                    ],
                )
                return

            self._pending.get(parent, {}).pop(tool_use_id, None)
            yield _result_event(tool_use_id, _terminal_result(result))
        except Exception as error:
            self._pending.get(parent, {}).pop(tool_use_id, None)
            status = "cancelled" if parent._cancel_signal.is_set() else "failed"
            yield _result_event(tool_use_id, {"status": status, "error": str(error)})

    async def _run_child(
        self,
        parent: Agent,
        child_state: _PendingChild,
        prompt: str | list[InterruptResponseContent],
        deadline: float,
    ) -> AgentResult:
        invocation = asyncio.create_task(
            child_state.child.invoke_async(
                prompt,
                invocation_state=child_state.child_invocation_state,
                limits={
                    "turns": child_state.remaining_turns,
                    "total_tokens": child_state.remaining_total_tokens,
                },
            )
        )
        cancellation = asyncio.create_task(_wait_for_parent_cancellation(parent))
        try:
            done, _ = await asyncio.wait(
                {invocation, cancellation},
                timeout=max(0, deadline - monotonic()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if invocation in done:
                return invocation.result()
            if cancellation in done:
                raise RuntimeError("use_agent child was cancelled")
            raise TimeoutError("use_agent child exceeded its execution timeout")
        finally:
            cancellation.cancel()
            await asyncio.gather(cancellation, return_exceptions=True)
            if not invocation.done():
                child_state.child.cancel()
                invocation.cancel()
                invocation.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)

    def _create_child(
        self,
        parent: Agent,
        raw_input: Any,
        parent_state: dict[str, Any],
    ) -> _PendingChild:
        from ....agent import Agent

        input_data = _parse_input(raw_input)
        depth = self._depths.get(parent, 0)
        if depth >= self._limits["depth"]:
            raise ValueError(f"use_agent recursion depth limit of {self._limits['depth']} reached")
        if _has_provider_native_tools(parent):
            raise ValueError(
                "use_agent is not supported with provider-native model tools; "
                "register governed SDK tools on the parent agent instead"
            )

        tools = _resolve_tools(parent, input_data.get("tools", []), self)
        child = Agent(
            model=parent.model,
            system_prompt=input_data.get("instructions"),
            sandbox=parent.sandbox,
            callback_handler=None,
        )
        self._depths[child] = depth + 1
        child.hooks._inherit_callbacks_from(parent.hooks, [BeforeToolsEvent, BeforeToolCallEvent])
        child.tool_registry.registry.clear()
        child.tool_registry.dynamic_tools.clear()
        for selected_tool in tools:
            child.tool_registry.register_tool(selected_tool)
        allowed_tool_ids = {id(tool) for tool in tools}
        child.add_hook(
            lambda event: _enforce_tool_grants(event, allowed_tool_ids),
            BeforeToolCallEvent,
            order=float("inf"),
        )
        return _PendingChild(
            child=child,
            task=input_data["task"],
            parent_interrupt_state=parent._interrupt_state,
            remaining_seconds=self._limits["timeout_seconds"],
            remaining_turns=self._limits["turns"],
            remaining_total_tokens=self._limits["total_tokens"],
            child_invocation_state=parent_state,
        )


def make_use_agent(
    *,
    limits: UseAgentLimits | None = None,
) -> AgentTool:
    """Create an experimental governed ``use_agent`` tool.

    Args:
        limits: Child execution limits.

    Returns:
        A streaming tool that creates bounded child agents.

    Raises:
        TypeError: If a child execution limit is outside its supported range.
    """
    return _UseAgentTool(limits=limits)


def _stopped_result(stop_reason: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "error": f"use_agent child stopped before completion: {stop_reason}",
    }


def _terminal_result(result: AgentResult) -> dict[str, Any]:
    if result.stop_reason == "cancelled":
        return {"status": "cancelled", "error": "use_agent child was cancelled"}
    if result.stop_reason not in ("end_turn", "stop_sequence"):
        return _stopped_result(result.stop_reason)
    return {
        "status": "completed",
        "output": str(result),
    }


def _parse_input(raw_input: Any) -> _UseAgentInput:
    if not isinstance(raw_input, dict):
        raise TypeError("use_agent input must be an object")
    unknown_fields = raw_input.keys() - _UseAgentInput.__annotations__.keys()
    if unknown_fields:
        raise ValueError(f"use_agent input contains unknown fields: {', '.join(sorted(unknown_fields))}")
    task = raw_input.get("task")
    instructions = raw_input.get("instructions")
    tools = raw_input.get("tools")
    if not isinstance(task, str) or not task:
        raise ValueError("use_agent task must be a non-empty string")
    if instructions is not None and (not isinstance(instructions, str) or not instructions):
        raise ValueError("use_agent instructions must be a non-empty string")
    if tools is not None:
        if not isinstance(tools, list) or any(not isinstance(name, str) or not name for name in tools):
            raise ValueError("use_agent tools must be a list of non-empty strings")
    return cast(_UseAgentInput, raw_input)


def _resolve_tools(
    parent: Agent,
    requested_names: list[str],
    executing_tool: AgentTool,
) -> list[AgentTool]:
    selected: list[AgentTool] = []
    for name in dict.fromkeys(requested_names):
        tool = parent.tool_registry.dynamic_tools.get(name) or parent.tool_registry.registry.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' was not found on the parent agent")
        if name == _USE_AGENT_TOOL_NAME and tool is not executing_tool:
            raise ValueError("A child can receive only the currently executing use_agent tool")
        selected.append(tool)
    return selected


def _enforce_tool_grants(event: BeforeToolCallEvent, allowed_tool_ids: set[int]) -> None:
    if event.cancel_tool or event.selected_tool is None:
        return
    if id(event.selected_tool) not in allowed_tool_ids:
        event.cancel_tool = "use_agent blocked a tool outside the child grant set"


def _resolve_limits(limits: UseAgentLimits | None) -> dict[str, int]:
    resolved = _DEFAULT_LIMITS.copy()
    if limits is not None:
        resolved.update(cast(dict[str, int], limits))
    for name, maximum in _DEFAULT_LIMITS.items():
        value = resolved[name]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise TypeError(f"limits.{name} must be an integer between 1 and {maximum}, got {value!r}")
    return resolved


def _interrupt_responses(
    parent: Agent,
    interrupts: tuple[tuple[str, str], ...],
) -> list[InterruptResponseContent]:
    responses: list[InterruptResponseContent] = []
    parent_interrupts = parent._interrupt_state.interrupts
    for child_id, outward_id in interrupts:
        interrupt = parent_interrupts.get(outward_id)
        if interrupt is None or interrupt.response is None:
            raise ValueError(f"use_agent interrupt '{outward_id}' has no response")
        responses.append({"interruptResponse": {"interruptId": child_id, "response": interrupt.response}})
    return responses


def _has_provider_native_tools(parent: Agent) -> bool:
    config = parent.model.get_config()
    if not isinstance(config, dict):
        return False
    if config.get("gemini_tools"):
        return True
    params = config.get("params")
    return isinstance(params, dict) and bool(params.get("tools"))


async def _wait_for_parent_cancellation(parent: Agent) -> None:
    while not parent._cancel_signal.is_set():
        await asyncio.sleep(0.05)


def _result_event(tool_use_id: str, result: dict[str, Any]) -> ToolResultEvent:
    tool_result: ToolResult = {
        "toolUseId": tool_use_id,
        "status": "success" if result["status"] == "completed" else "error",
        "content": [{"json": cast(Any, result)}],
    }
    return ToolResultEvent(tool_result)


use_agent = make_use_agent()
