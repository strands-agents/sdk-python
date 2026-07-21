"""Swarm vended tool.

Vends a ``swarm`` tool that lets an agent spin up a handoff-based team of
sub-agents at call time. The tool is a thin shim over
:class:`~strands.multiagent.swarm.Swarm`: it validates the spec, constructs
child :class:`~strands.agent.Agent` instances (all inheriting the parent's
model), builds the swarm with fixed safety caps, executes it, and maps the
result into the shared multi-agent result dialect (see
``_multiagent_conventions.md``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, TypedDict

from ...agent import Agent
from ...hooks import BeforeNodeCallEvent
from ...multiagent.base import Status
from ...multiagent.swarm import Swarm
from ...tools.decorator import tool
from ...types.event_loop import Usage
from ...types.tools import ToolContext

if TYPE_CHECKING:
    from ...tools.decorator import DecoratedFunctionTool

_MAX_AGENTS = 5
"""Hard cap on the number of child agents in a single swarm invocation.

The SDK's `Swarm` doesn't cap agent count itself. We pick 5 as the point past
which a swarm is almost certainly the wrong tool — the parent agent should
model that many collaborators as a Graph or a hand-authored orchestration.
"""

_EXECUTION_TIMEOUT_SECONDS = 300.0
"""Total wall-clock ceiling passed to Swarm.execution_timeout."""

_NODE_TIMEOUT_SECONDS = 120.0
"""Per-node wall-clock ceiling passed to Swarm.node_timeout."""

_MAX_ITERATIONS = 10
"""Cap on how many node executions the swarm can perform. Also bounds handoffs."""

# Shared with the sibling multi-agent tools. See _multiagent_conventions.md.
_MULTIAGENT_DEPTH_KEY = "multiagent_depth"
_MAX_MULTIAGENT_DEPTH = 3

# Size caps from the shared multi-agent dialect (bytes, UTF-8).
_MAX_INITIAL_INPUT_BYTES = 32 * 1024
_MAX_SYSTEM_PROMPT_BYTES = 8 * 1024

# `name` regex + char cap from the shared multi-agent dialect.
_NAME_REGEX = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,63}$")
_MAX_NAME_CHARS = 64

# Max entries in a child spec's `tools` allowlist (shared multi-agent dialect).
_MAX_TOOLS_PER_SPEC = 64

# Names of the multi-agent tools themselves. Children may not list any of these
# in their `tools` allowlist — that would let a compromised model bypass the
# shared depth counter by having a child re-invoke `swarm`/`graph`/`use_agent`/etc.
_MULTIAGENT_TOOL_NAMES = frozenset({"use_agent", "swarm", "graph", "a2a_client"})


SWARM_TOOL_DESCRIPTION = (
    "Spin up a small team of sub-agents that hand off to each other to complete a task. "
    "Use when the task splits cleanly into specialized roles and you want the sub-agents "
    "to decide amongst themselves who does what. "
    "Each sub-agent is defined by a name, system_prompt, and a (possibly empty) tools list. "
    "Sub-agents inherit your model. Child sub-agents may only use tools you allowlist from "
    "your own tool registry. Returns the final response plus which sub-agents ran."
)


class SwarmToolResult(TypedDict):
    """Shape returned by the swarm tool. Matches the shared multi-agent dialect."""

    status: str
    output: str
    node_history: list[str]
    execution_count: int
    execution_time_ms: int
    usage: Usage


class MultiagentDepthExceeded(ValueError):
    """Raised when the caller has already hit the shared multi-agent recursion cap."""


def make_swarm(
    *,
    name: str = "swarm",
    description: str = SWARM_TOOL_DESCRIPTION,
    max_agents: int = _MAX_AGENTS,
    execution_timeout: float = _EXECUTION_TIMEOUT_SECONDS,
    node_timeout: float = _NODE_TIMEOUT_SECONDS,
    max_iterations: int = _MAX_ITERATIONS,
    max_multiagent_depth: int = _MAX_MULTIAGENT_DEPTH,
    max_initial_input_bytes: int = _MAX_INITIAL_INPUT_BYTES,
    max_system_prompt_bytes: int = _MAX_SYSTEM_PROMPT_BYTES,
) -> DecoratedFunctionTool:
    """Create the swarm vended tool.

    Every cap is fixed at the tool boundary and not reachable by the model at
    call time — the parent chooses them at tool construction, exactly like
    ``max_iterations`` on a plain ``Agent``.

    Args:
        name: Tool name. Defaults to ``"swarm"``.
        description: Description shown to the model.
        max_agents: Maximum number of child agents allowed in a single invocation.
        execution_timeout: Total wall-clock timeout in seconds for the whole swarm.
        node_timeout: Per-node wall-clock timeout in seconds.
        max_iterations: Cap on total node executions (also caps handoffs).
        max_multiagent_depth: Cap on how deep the shared multi-agent recursion
            counter may go before the tool refuses (see
            ``_multiagent_conventions.md``).
        max_initial_input_bytes: UTF-8 byte cap on ``initial_input``.
        max_system_prompt_bytes: UTF-8 byte cap on each spec's ``system_prompt``.

    Returns:
        A decorated tool that runs a Swarm and returns a normalized result dict.
    """

    @tool(name=name, description=description, context="tool_context")
    async def swarm_tool(
        agents: list[dict[str, Any]],
        initial_input: str,
        tool_context: ToolContext,
        entry_agent: str | None = None,
    ) -> SwarmToolResult:
        """Spin up a handoff-based sub-agent swarm to complete a task.

        Args:
            agents: List of sub-agent specs. Each spec is a dict with:

                - ``name`` (str, required): unique identifier for the sub-agent.
                  Must match ``[a-zA-Z_][a-zA-Z0-9_]{0,63}``.
                - ``system_prompt`` (str, required): sub-agent's system prompt
                  (<= 8 KiB).
                - ``tools`` (list[str], required, may be empty): names of tools
                  to expose to this sub-agent. Must be a subset of the parent
                  agent's registered tools. Max 64 entries. No wildcards.
                - ``description`` (str, optional): shown to sibling agents in
                  handoff context.
            initial_input: Task to hand to the entry agent (<= 32 KiB).
            tool_context: Injected by the framework. Not user-facing.
            entry_agent: Name of the agent to start with. Defaults to the first
                agent in ``agents``.
        """
        parent_agent = tool_context.agent
        invocation_state = tool_context.invocation_state

        # Depth check first — cheapest way to shut down runaway delegation chains,
        # and shared across every multi-agent tool via `invocation_state`.
        depth = _current_depth(invocation_state)
        if depth >= max_multiagent_depth:
            raise MultiagentDepthExceeded(
                f"swarm refused: multi-agent recursion depth cap of {max_multiagent_depth} reached "
                f"(current depth {depth})"
            )

        _validate_initial_input(initial_input, max_initial_input_bytes=max_initial_input_bytes)
        specs = _validate_specs(
            agents, max_agents=max_agents, max_system_prompt_bytes=max_system_prompt_bytes
        )

        parent_tools = _parent_tool_registry(parent_agent)
        child_agents = [_build_child_agent(spec, parent_agent, parent_tools) for spec in specs]

        entry_point: Agent | None = None
        if entry_agent is not None:
            if not isinstance(entry_agent, str):
                raise ValueError("entry_agent must be a string")
            match = next((a for a in child_agents if a.name == entry_agent), None)
            if match is None:
                available = [a.name for a in child_agents]
                raise ValueError(f"entry_agent '{entry_agent}' not in agents list. Available: {available}")
            entry_point = match

        cancel_hook = _ParentCancelHook(parent_agent)
        child_swarm = Swarm(
            child_agents,
            entry_point=entry_point,
            max_handoffs=max_iterations,
            max_iterations=max_iterations,
            execution_timeout=execution_timeout,
            node_timeout=node_timeout,
        )
        child_swarm.add_hook(cancel_hook.on_before_node_call, BeforeNodeCallEvent)

        # Preserve the parent's invocation_state so tracing / telemetry /
        # per-run keys flow through to the child. Only override the shared
        # depth counter with the incremented value.
        child_invocation_state: dict[str, Any] = {
            **invocation_state,
            _MULTIAGENT_DEPTH_KEY: depth + 1,
        }
        result = await child_swarm.invoke_async(initial_input, invocation_state=child_invocation_state)
        # The hook fires when the parent's cancel signal is set between nodes.
        # The SDK's Swarm surfaces that as `Status.FAILED`, not `INTERRUPTED`,
        # so we detect it here explicitly rather than depending on the enum.
        return _map_result(result, cancelled=cancel_hook.fired)

    return swarm_tool


swarm = make_swarm()
"""Default swarm tool with the built-in safety caps."""


class _ChildSpec(TypedDict, total=False):
    # `name`, `system_prompt`, and `tools` are always populated by
    # `_validate_specs` (they're required per the shared spec). `description`
    # is genuinely optional. `total=False` here reflects that `description`
    # may be absent — required fields are enforced at validation time.
    name: str
    system_prompt: str
    description: str
    tools: list[str]


def _current_depth(invocation_state: dict[str, Any]) -> int:
    """Read the current multi-agent recursion depth from the parent's state.

    Non-integer / negative values are treated as zero: the model cannot corrupt
    the counter by writing garbage into invocation_state earlier in the run,
    and a misbehaving parent shouldn't crash the tool either.
    """
    raw = invocation_state.get(_MULTIAGENT_DEPTH_KEY, 0)
    if not isinstance(raw, int) or raw < 0:
        return 0
    return raw


def _validate_initial_input(value: Any, *, max_initial_input_bytes: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("initial_input must be a non-empty string")
    size = len(value.encode("utf-8"))
    if size > max_initial_input_bytes:
        raise ValueError(f"initial_input exceeds size cap: {size} bytes > {max_initial_input_bytes} bytes")


def _validate_specs(
    agents: Any, *, max_agents: int, max_system_prompt_bytes: int
) -> list[_ChildSpec]:
    """Validate the raw ``agents`` payload from the model.

    We do this by hand rather than with Pydantic to keep error messages tight
    and to avoid an extra schema layer the model would then have to internalize.
    """
    if not isinstance(agents, list):
        raise ValueError("agents must be a list of agent spec objects")
    if len(agents) == 0:
        raise ValueError("agents list is empty; provide at least one agent spec")
    if len(agents) > max_agents:
        raise ValueError(f"too many agents: got {len(agents)}, max is {max_agents}")

    validated: list[_ChildSpec] = []
    seen_names: set[str] = set()
    for i, raw in enumerate(agents):
        if not isinstance(raw, dict):
            raise ValueError(f"agents[{i}] must be an object, got {type(raw).__name__}")

        # `name` is required; must match the shared dialect's regex and length cap.
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"agents[{i}].name must be a non-empty string")
        if len(name) > _MAX_NAME_CHARS:
            raise ValueError(
                f"agents[{i}].name exceeds {_MAX_NAME_CHARS}-char cap: {len(name)} chars"
            )
        if not _NAME_REGEX.match(name):
            raise ValueError(
                f"agents[{i}].name '{name}' must match [a-zA-Z_][a-zA-Z0-9_]{{0,63}}"
            )
        if name in seen_names:
            raise ValueError(f"agents[{i}].name '{name}' is a duplicate; names must be unique")
        seen_names.add(name)

        # `system_prompt` is required per the shared dialect.
        system_prompt = raw.get("system_prompt")
        if not isinstance(system_prompt, str):
            raise ValueError(f"agents[{i}].system_prompt is required and must be a string")
        size = len(system_prompt.encode("utf-8"))
        if size > max_system_prompt_bytes:
            raise ValueError(
                f"agents[{i}].system_prompt exceeds size cap: {size} bytes > {max_system_prompt_bytes} bytes"
            )

        # `tools` is required per the shared dialect (may be empty for a no-tool child).
        tools = raw.get("tools")
        if not isinstance(tools, list) or not all(isinstance(t, str) for t in tools):
            raise ValueError(f"agents[{i}].tools is required and must be a list of strings")
        if len(tools) > _MAX_TOOLS_PER_SPEC:
            raise ValueError(
                f"agents[{i}].tools exceeds cap: {len(tools)} entries > {_MAX_TOOLS_PER_SPEC}"
            )

        spec: _ChildSpec = {
            "name": name,
            "system_prompt": system_prompt,
            "tools": list(tools),
        }

        description = raw.get("description")
        if description is not None:
            if not isinstance(description, str):
                raise ValueError(f"agents[{i}].description must be a string")
            spec["description"] = description

        known = {"name", "system_prompt", "description", "tools"}
        extras = set(raw.keys()) - known
        if extras:
            raise ValueError(f"agents[{i}] has unsupported fields: {sorted(extras)}. Allowed: {sorted(known)}")

        validated.append(spec)

    return validated


def _parent_tool_registry(parent_agent: Any) -> dict[str, Any]:
    """Return the parent agent's tool registry mapping, or empty dict.

    Kept lenient so callers that construct minimal test agents (bare
    ``SimpleNamespace``s) don't crash on the ``tool_registry`` attribute.
    """
    tool_registry = getattr(parent_agent, "tool_registry", None)
    if tool_registry is None:
        return {}
    registry = getattr(tool_registry, "registry", None)
    if not isinstance(registry, dict):
        return {}
    return registry


def _build_child_agent(spec: _ChildSpec, parent_agent: Any, parent_tools: dict[str, Any]) -> Agent:
    """Construct one child Agent from a validated spec.

    Child agents inherit the parent's model. Tools are name-resolved against the
    parent's registered tools; unknown names raise so the model can't request
    tools we didn't hand it.
    """
    # `tools` is always populated by `_validate_specs` (required per the
    # shared spec, may be an empty list).
    child_tools: list[Any] = []
    for tool_name in spec["tools"]:
        # Defense-in-depth: even if the parent has a multi-agent tool registered,
        # the model cannot grant it to a child. Otherwise a child could re-enter
        # `swarm`/`graph`/etc. and bypass the shared depth counter.
        if tool_name in _MULTIAGENT_TOOL_NAMES:
            raise ValueError(
                f"agent '{spec['name']}' requested multi-agent tool '{tool_name}'; "
                "multi-agent tools may not be listed in a child spec's tools"
            )
        if tool_name not in parent_tools:
            available = sorted(parent_tools.keys())
            raise ValueError(
                f"agent '{spec['name']}' requested unknown tool '{tool_name}'. Available parent tools: {available}"
            )
        child_tools.append(parent_tools[tool_name])

    kwargs: dict[str, Any] = {
        "name": spec["name"],
        "model": parent_agent.model,
        "tools": child_tools,
        "system_prompt": spec["system_prompt"],
        # Output belongs to the parent's stream — silence the child's callback handler.
        "callback_handler": None,
    }
    if "description" in spec:
        kwargs["description"] = spec["description"]

    return Agent(**kwargs)


def _map_result(result: Any, *, cancelled: bool = False) -> SwarmToolResult:
    """Map a SwarmResult (or MultiAgentResult) into the shared dialect.

    Translates the SDK's `Status` enum (`completed`/`failed`/`interrupted`) into
    the shared dialect's `success`/`error`/`cancelled` vocabulary so downstream
    models get a consistent contract across every multi-agent tool.

    ``cancelled`` overrides the SDK's status when the parent-cancel hook fired.
    The SDK maps a hook-set ``cancel_node`` to ``Status.FAILED``, which without
    this override would surface as ``error`` rather than ``cancelled``.
    """
    status_value = _map_status(getattr(result, "status", None), cancelled=cancelled)
    node_history = [str(n) for n in getattr(result, "node_history", []) or []]
    output = _extract_output(result, node_history)

    usage_dict = dict(getattr(result, "accumulated_usage", {}) or {})
    usage: Usage = {
        "inputTokens": int(usage_dict.get("inputTokens", 0)),
        "outputTokens": int(usage_dict.get("outputTokens", 0)),
        "totalTokens": int(usage_dict.get("totalTokens", 0)),
    }

    return {
        "status": status_value,
        "output": output,
        "node_history": node_history,
        "execution_count": int(getattr(result, "execution_count", 0)),
        "execution_time_ms": int(getattr(result, "execution_time", 0)),
        "usage": usage,
    }


def _map_status(status: Any, *, cancelled: bool = False) -> str:
    """Map the SDK's `Status` enum onto the shared multi-agent result dialect.

    The SDK's execution vocabulary (`completed`/`failed`/`interrupted`) is
    orthogonal to what the tool boundary exposes. Callers on the other side of
    the tool see a stable `success` / `error` / `cancelled`. Anything unknown
    falls through to `error` — better to surface an unrecognized state as an
    error than to silently paper over it as success.

    When ``cancelled`` is True the parent-cancel hook fired during execution,
    which the SDK reports as ``Status.FAILED``; we override to ``cancelled`` so
    downstream models can distinguish parent cancellation from a real failure.
    """
    if cancelled:
        return "cancelled"
    if status is Status.COMPLETED:
        return "success"
    if status is Status.INTERRUPTED:
        return "cancelled"
    return "error"


def _extract_output(result: Any, node_history: list[str]) -> str:
    """Pull the final text from the terminal agent's AgentResult.

    Falls back to an empty string if the swarm never produced a result event
    (e.g. failed before the first agent got to speak).
    """
    if not node_history:
        return ""
    last_node_id = node_history[-1]
    results = getattr(result, "results", None) or {}
    node_result = results.get(last_node_id)
    if node_result is None:
        return ""

    inner = getattr(node_result, "result", None)
    if inner is None:
        return ""
    # Exceptions are stored on NodeResult.result for FAILED nodes.
    if isinstance(inner, Exception):
        return ""

    message = getattr(inner, "message", None)
    if not isinstance(message, dict):
        return ""

    content = message.get("content") or []
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


class _ParentCancelHook:
    """Cancel the child swarm before each node if the parent agent was cancelled.

    The Swarm exposes a `BeforeNodeCallEvent.cancel_node` field that stops
    execution cleanly. We flip it whenever the parent's cancel signal is set.
    This is the same pattern the SDK uses for orchestrator-level cancellation.

    This is a coarse-grained boundary: parent cancellation is observed only
    between nodes, not mid-node. Fine-grained cancellation would require the
    swarm to expose an in-flight `cancel()` API; today it does not.
    """

    def __init__(self, parent_agent: Any) -> None:
        # We reach into `_cancel_signal` because `Agent` doesn't expose a public
        # accessor. If the SDK renames or removes this attribute, we want to fail
        # loudly rather than silently drop cancellations on the floor.
        if not hasattr(parent_agent, "_cancel_signal"):
            raise AttributeError(
                "swarm cancellation hook requires parent_agent._cancel_signal; "
                "the SDK may have renamed or removed this attribute"
            )
        self._parent = parent_agent
        self.fired = False
        """True once ``on_before_node_call`` has cancelled a node.

        The wrapper reads this after ``invoke_async`` returns to override the
        SDK's ``Status.FAILED`` (which is what a hook-set ``cancel_node``
        actually produces) into the shared dialect's ``cancelled``.
        """

    def on_before_node_call(self, event: BeforeNodeCallEvent) -> None:
        cancel_signal = self._parent._cancel_signal
        if cancel_signal is not None and cancel_signal.is_set():
            event.cancel_node = "cancelled by parent agent"
            self.fired = True
