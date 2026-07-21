"""Vended ``use_agent`` tool: delegate a task to a nested agent.

The calling agent constructs a fresh :class:`~strands.Agent` with the given
system prompt and an allowlisted subset of the parent's tools, then hands
it a single task. The nested agent's final text response is returned to the
parent.

This is a shim over :class:`~strands.Agent` and its normal construction path;
it does not reinvent agent lifecycle. It only enforces the security surface
of runtime delegation (allowlist, size caps, recursion cap, cancellation
propagation).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import time
from typing import TYPE_CHECKING, Any

from ...tools.decorator import tool
from ...types.tools import ToolContext

if TYPE_CHECKING:
    from ...tools.decorator import DecoratedFunctionTool

logger = logging.getLogger(__name__)

_MAX_SYSTEM_PROMPT_BYTES = 8 * 1024
_MAX_TASK_BYTES = 32 * 1024
_MAX_TOOL_ALLOWLIST = 64

_DEPTH_KEY = "multiagent_depth"
_MAX_DEPTH = 3

_WILDCARD_TOOL_NAMES = frozenset({"*", "**", "all", "any", ""})

# Defense-in-depth: child agents cannot invoke multi-agent tools directly. A
# developer who wants that must register the variants at construction time.
_MULTIAGENT_TOOL_NAMES = frozenset({"use_agent", "swarm", "graph", "a2a_client"})


USE_AGENT_DESCRIPTION = (
    "Delegate a single task to a nested agent that you construct at call time. "
    "You provide the child agent's system_prompt, an explicit allowlist of tool "
    "names to expose (drawn from your own tools), and the task itself. The child "
    "runs with the same model as you and a fresh conversation, and returns its "
    "final text response. Prefer this for scoped sub-tasks that benefit from a "
    "different system prompt or a narrower tool surface than your own."
)


class MultiagentDepthExceeded(ValueError):
    """Raised when the shared multi-agent recursion cap is reached."""


def _validate_positive_int_cap(value: Any, name: str) -> int:
    """Validate a factory-time positive integer cap.

    Rejects bool (a Python ``int`` subclass), non-integers, non-finite floats,
    and non-positive values. Keeps the cap-bypass surface tight even when a
    caller wires up the tool with something exotic.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive int, got bool")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite positive int, got {value}")
        if not value.is_integer():
            raise ValueError(f"{name} must be an int, got {value}")
        value = int(value)
    if not isinstance(value, int):
        raise ValueError(f"{name} must be a positive int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_bounded_string(value: Any, name: str, max_bytes: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    size = len(value.encode("utf-8"))
    if size > max_bytes:
        raise ValueError(f"{name} exceeds size cap: {size} bytes > {max_bytes} bytes")
    return value


def _validate_tool_allowlist(tools: Any, parent_tool_names: set[str], max_entries: int) -> list[str]:
    """Validate ``tools`` is an exact-name allowlist drawn from the parent registry.

    Rejects wildcards, inherit-all, non-list values, non-string entries,
    over-long lists, names not in the parent's registry, and multi-agent tool
    names. Deduplicates while preserving order.
    """
    if not isinstance(tools, list):
        raise ValueError(f"tools must be a list of tool names, got {type(tools).__name__}")
    if len(tools) > max_entries:
        raise ValueError(f"tools allowlist exceeds cap of {max_entries} entries")

    seen: set[str] = set()
    resolved: list[str] = []
    for entry in tools:
        if not isinstance(entry, str):
            raise ValueError(f"tools entries must be strings, got {type(entry).__name__}")
        stripped = entry.strip()
        if stripped.lower() in _WILDCARD_TOOL_NAMES:
            raise ValueError(f"tools entry {entry!r} is a wildcard; every child tool must be named explicitly")
        if stripped in _MULTIAGENT_TOOL_NAMES:
            raise ValueError(
                f"tools entry {stripped!r} is a multi-agent tool and cannot be nested inside a child agent"
            )
        if stripped in seen:
            continue
        if stripped not in parent_tool_names:
            raise ValueError(f"tools entry {stripped!r} is not present in the parent agent's tool registry")
        seen.add(stripped)
        resolved.append(stripped)
    return resolved


def _current_depth(invocation_state: dict[str, Any]) -> int:
    """Read the current multi-agent recursion depth from the parent's state.

    ``bool`` is a Python ``int`` subclass; reject it explicitly so ``True``
    doesn't silently count as depth 1. Non-integers and negatives fall through
    to zero: a hostile ``invocation_state`` shouldn't be able to confuse the
    counter into refusing at a lower depth than configured, and a misbehaving
    parent shouldn't crash the tool either.
    """
    raw = invocation_state.get(_DEPTH_KEY, 0)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return 0
    return int(raw)


def _resolve_parent_tools(agent: Any, names: list[str]) -> list[Any]:
    registry = agent.tool_registry.registry
    return [registry[n] for n in names]


async def _watch_parent_cancel(parent_agent: Any, sub_agent: Any) -> None:
    """Poll the parent's cancel signal and forward it to ``sub_agent``.

    The parent's ``_cancel_signal`` is a ``threading.Event``; sub-agents get
    their own. Cross-agent sharing of the underlying event is not supported,
    so we observe the parent's flag on a short interval and call ``cancel()``
    on the child. Cancel is idempotent.
    """
    while not parent_agent._cancel_signal.is_set():
        await asyncio.sleep(0.05)
    sub_agent.cancel()


def make_use_agent(
    *,
    name: str = "use_agent",
    description: str = USE_AGENT_DESCRIPTION,
    max_depth: int = _MAX_DEPTH,
    max_system_prompt_bytes: int = _MAX_SYSTEM_PROMPT_BYTES,
    max_task_bytes: int = _MAX_TASK_BYTES,
    max_tool_allowlist: int = _MAX_TOOL_ALLOWLIST,
) -> DecoratedFunctionTool:
    """Create the use_agent vended tool.

    Every cap is fixed at the tool boundary and not reachable by the model at
    call time; the parent chooses them at tool construction, exactly like
    ``max_iterations`` on a plain ``Agent``.

    Args:
        name: Tool name. Defaults to ``"use_agent"``.
        description: Description shown to the model.
        max_depth: Cap on the shared multi-agent recursion counter before the
            tool refuses (see ``_multiagent_conventions.md``).
        max_system_prompt_bytes: UTF-8 byte cap on the child's system prompt.
        max_task_bytes: UTF-8 byte cap on the task string.
        max_tool_allowlist: Cap on the number of entries in the tool allowlist.

    Returns:
        A decorated tool that constructs and invokes a nested child agent.
    """
    max_depth = _validate_positive_int_cap(max_depth, "max_depth")
    max_system_prompt_bytes = _validate_positive_int_cap(max_system_prompt_bytes, "max_system_prompt_bytes")
    max_task_bytes = _validate_positive_int_cap(max_task_bytes, "max_task_bytes")
    max_tool_allowlist = _validate_positive_int_cap(max_tool_allowlist, "max_tool_allowlist")

    @tool(name=name, description=description, context="tool_context")
    async def use_agent_tool(
        system_prompt: str,
        task: str,
        tool_context: ToolContext,
        tools: list[str] | None = None,
    ) -> dict[str, Any]:
        """Delegate a single task to a freshly constructed nested agent.

        The nested agent inherits the parent's model instance, runs on a fresh
        conversation, and is exposed only to the parent tools you name in ``tools``.

        Providers shipped with Strands do not hold per-invocation state on
        ``self`` today, so sharing the model instance is safe; if a future
        provider does, this may need to be revisited.

        Args:
            system_prompt: System prompt for the nested agent. Non-empty, capped
                at ``max_system_prompt_bytes`` UTF-8 bytes.
            task: The task to hand the nested agent. Non-empty, capped at
                ``max_task_bytes`` UTF-8 bytes.
            tool_context: Injected by the framework. Not user-facing.
            tools: Exact-name allowlist of tools to expose to the nested agent.
                Every entry must be a tool that exists in your tool registry.
                Wildcards and multi-agent tool names are rejected.
        """
        from ...agent.agent import Agent

        parent_agent = tool_context.agent
        invocation_state = tool_context.invocation_state or {}

        depth = _current_depth(invocation_state)
        if depth >= max_depth:
            raise MultiagentDepthExceeded(
                f"use_agent refused: recursion depth cap of {max_depth} reached (current depth {depth})"
            )

        system_prompt_v = _validate_bounded_string(system_prompt, "system_prompt", max_system_prompt_bytes)
        task_v = _validate_bounded_string(task, "task", max_task_bytes)

        parent_tool_names = set(parent_agent.tool_registry.registry.keys())
        tool_names = _validate_tool_allowlist(tools or [], parent_tool_names, max_tool_allowlist)
        child_tools = _resolve_parent_tools(parent_agent, tool_names)

        sub_agent = Agent(
            model=parent_agent.model,
            system_prompt=system_prompt_v,
            tools=child_tools,
            name=f"{getattr(parent_agent, 'name', 'agent')}::use_agent",
            callback_handler=None,
        )

        # Preserve the parent's invocation_state so tracing / telemetry /
        # per-run keys flow through to the child. Only override the shared
        # depth counter with the incremented value.
        child_invocation_state: dict[str, Any] = {
            **invocation_state,
            _DEPTH_KEY: depth + 1,
        }

        start = time.monotonic()
        watcher = asyncio.create_task(_watch_parent_cancel(parent_agent, sub_agent))
        try:
            result = await sub_agent.invoke_async(task_v, invocation_state=child_invocation_state)
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher

        elapsed_ms = int((time.monotonic() - start) * 1000)

        return {
            "status": _map_stop_reason(result.stop_reason),
            "output": str(result),
            "execution_time_ms": elapsed_ms,
        }

    return use_agent_tool


def _map_stop_reason(stop_reason: str) -> str:
    """Map an ``AgentResult.stop_reason`` into the shared multi-agent result vocabulary.

    The shared dialect (see ``_multiagent_conventions.md``) uses the lower-cased
    values of the SDK's multi-agent ``Status`` enum: completed, failed,
    cancelled, interrupted. Any non-``end_turn`` non-``cancelled`` non-``interrupt``
    stop reason (``limit_turns``, ``content_filtered``, ``max_tokens``,
    ``guardrail_intervened``, ...) surfaces as ``failed`` so the parent can
    distinguish a delivered delegation from one that hit a policy or limit.
    """
    if stop_reason == "end_turn":
        return "completed"
    if stop_reason == "cancelled":
        return "cancelled"
    if stop_reason == "interrupt":
        return "interrupted"
    return "failed"


use_agent = make_use_agent()
"""Default use_agent tool with the built-in safety caps."""
