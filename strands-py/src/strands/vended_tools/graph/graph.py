"""Graph tool: deterministic DAG orchestration over the SDK's Graph primitive.

Shim over :class:`~strands.multiagent.graph.GraphBuilder` /
:class:`~strands.multiagent.graph.Graph`. The caller agent describes a DAG of
sub-agents inline; this tool constructs the ``Graph`` and invokes it, returning
per-node text results.

Design notes:
    * The tool is a **shim**. All orchestration lives in ``strands.multiagent.graph``.
      This module only validates the model-supplied topology, builds sub-agents,
      wires them into a ``Graph``, and formats the result.
    * **No edge conditions**. Conditional routing would require executing
      model-supplied code at every edge; that is not appropriate for a vended tool.
      Users who want conditional routing should build a ``Graph`` directly.
    * **Tools are name-allow-listed** from the parent agent's registry. The model
      cannot conjure new tools inside a node — only reference tools the caller
      already has. This mirrors the ``use_agent`` / ``swarm`` convention.
    * **Cycles are rejected** at validation time. ``Graph`` supports cycles;
      the tool does not.
    * **Recursion depth** participates in the shared ``multiagent_depth`` counter
      threaded through ``invocation_state``. See ``_multiagent_conventions.md``.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Any, cast

from ...agent import Agent
from ...hooks import BeforeNodeCallEvent
from ...multiagent.base import Status
from ...multiagent.graph import GraphBuilder
from ...tools.decorator import tool
from ...types.tools import ToolContext
from .types import (
    DEFAULT_EXECUTION_TIMEOUT_SECONDS,
    DEFAULT_NODE_TIMEOUT_SECONDS,
    GRAPH_DESCRIPTION,
    MAX_EDGES,
    MAX_ID_LENGTH,
    MAX_INITIAL_INPUT_LENGTH,
    MAX_NODE_EXECUTIONS,
    MAX_NODES,
    MAX_SYSTEM_PROMPT_LENGTH,
    MAX_TOOLS_PER_NODE,
    GraphNodeResult,
    GraphOutput,
)

# Shared multi-agent depth counter — see ``_multiagent_conventions.md``. All
# multi-agent vended tools (`use_agent`, `swarm`, `graph`, `a2a_client`)
# increment the same key so a chain that crosses tool boundaries is still
# capped.
_DEPTH_KEY = "multiagent_depth"
_DEFAULT_MAX_DEPTH = 3

# Multi-agent tool names that must never appear in a child node's allow-list.
# Depth-cap participation already blocks unbounded recursion, but the shared
# convention (``_multiagent_conventions.md``) also mandates an explicit reject
# at the tool boundary as defense-in-depth — a compromised parent registry could
# expose these under aliased names, but a reject on the canonical name still
# catches the common case.
_MULTIAGENT_TOOL_NAMES = frozenset({"use_agent", "swarm", "graph", "a2a_client"})

if TYPE_CHECKING:
    from ...tools.decorator import DecoratedFunctionTool
    from ...types.tools import AgentTool


class MultiagentDepthExceeded(RuntimeError):
    """Raised when the shared multi-agent recursion cap is reached."""


# Sentinel embedded in the ``cancel_node`` message. The SDK's ``Graph`` re-raises
# that message as a ``RuntimeError``, so we need to distinguish "cancelled by
# our hook" from "some other runtime failure that mentioned cancellation".
_CANCEL_SENTINEL = "graph-tool-parent-cancel"
_CANCEL_MESSAGE = f"cancelled by parent agent [{_CANCEL_SENTINEL}]"


class _ParentCancelHook:
    """Cancel each graph node when the parent agent's cancel signal is set.

    The SDK exposes ``BeforeNodeCallEvent.cancel_node``; setting that field
    stops execution cleanly at the next node boundary. Polling the parent's
    ``threading.Event`` inside a hook keeps propagation on the same task as
    the graph itself, which is the only correct place for it — the previous
    finally-block approach only ran after ``invoke_async`` had already returned.
    """

    def __init__(self, parent_agent: Any) -> None:
        self._parent = parent_agent

    def on_before_node_call(self, event: BeforeNodeCallEvent) -> None:
        cancel_signal = getattr(self._parent, "_cancel_signal", None)
        if cancel_signal is not None and cancel_signal.is_set():
            event.cancel_node = _CANCEL_MESSAGE


def _current_depth(invocation_state: dict[str, Any]) -> int:
    raw = invocation_state.get(_DEPTH_KEY, 0)
    # ``bool`` is a subclass of ``int``; reject it explicitly so ``True`` doesn't
    # silently count as depth 1. Not exploitable, but a hostile ``invocation_state``
    # shouldn't be able to confuse the counter.
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        return 0
    return int(raw)


def _cancelled_output(execution_time_ms: int, partial: str = "") -> GraphOutput:
    """Build a ``cancelled`` result.

    Spec (see ``_multiagent_conventions.md``) forbids raising past the tool
    boundary; a cancelled result reaching the loop is a signal to the parent,
    not an exception to propagate.
    """
    return GraphOutput(
        status="cancelled",
        output=partial,
        execution_order=[],
        results={},
        execution_time_ms=execution_time_ms,
    )


def _validate_node_id(node_id: Any) -> str:
    if not isinstance(node_id, str) or not node_id:
        raise ValueError("Each node must have a non-empty string 'id'.")
    if len(node_id) > MAX_ID_LENGTH:
        raise ValueError(f"Node id '{node_id[:16]}...' exceeds {MAX_ID_LENGTH} characters.")
    if not all(ch.isalnum() or ch in "_-" for ch in node_id):
        raise ValueError(
            f"Node id '{node_id}' contains characters outside [A-Za-z0-9_-]. "
            "Use short identifiers for readability and to avoid collisions."
        )
    return node_id


def _resolve_tools(names: list[Any] | None, parent_registry: dict[str, AgentTool], node_id: str) -> list[AgentTool]:
    """Resolve a node's tool allow-list from the parent registry.

    Args:
        names: List of tool names the model asked for on this node. ``None`` or
            an empty list means the node runs with no tools.
        parent_registry: The caller agent's tool registry (name -> tool).
        node_id: Node id, for error messages.

    Raises:
        ValueError: If ``names`` is not a list of strings, contains a wildcard,
            or references a tool the parent registry does not expose.
    """
    if names is None:
        return []
    if not isinstance(names, list):
        raise ValueError(f"Node '{node_id}': 'tools' must be a list of tool names.")
    if len(names) > MAX_TOOLS_PER_NODE:
        raise ValueError(
            f"Node '{node_id}': 'tools' has {len(names)} entries (max {MAX_TOOLS_PER_NODE})."
        )

    resolved: list[AgentTool] = []
    seen: set[str] = set()
    for raw in names:
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"Node '{node_id}': tool entries must be non-empty strings.")
        # Reject wildcards explicitly — a name-allow-list, never "all parent tools".
        if raw == "*" or "*" in raw:
            raise ValueError(
                f"Node '{node_id}': tool '{raw}' looks like a wildcard. "
                "List each tool by name; wildcards are not allowed."
            )
        # Reject multi-agent tools by name — defense-in-depth on top of the
        # shared depth cap. See ``_multiagent_conventions.md``.
        if raw in _MULTIAGENT_TOOL_NAMES:
            raise ValueError(
                f"Node '{node_id}': tool '{raw}' is a multi-agent tool and may not be "
                "used inside a graph node's allow-list."
            )
        if raw in seen:
            continue
        seen.add(raw)
        tool_instance = parent_registry.get(raw)
        if tool_instance is None:
            raise ValueError(f"Node '{node_id}': tool '{raw}' is not registered on the parent agent.")
        resolved.append(tool_instance)
    return resolved


def _validate_nodes_edges(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    """Structural checks that must pass before we build any sub-agents.

    Verifies node/edge counts, unique ids, that edges reference known nodes,
    and that the resulting graph has no cycles. The SDK ``GraphBuilder`` also
    accepts cycles (they're valid for cyclic multi-agent patterns), so cycle
    detection is the tool's responsibility.
    """
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("'nodes' must be a non-empty list.")
    if len(nodes) > MAX_NODES:
        raise ValueError(f"Too many nodes: {len(nodes)} (max {MAX_NODES}).")

    if not isinstance(edges, list):
        raise ValueError("'edges' must be a list (may be empty).")
    if len(edges) > MAX_EDGES:
        raise ValueError(f"Too many edges: {len(edges)} (max {MAX_EDGES}).")

    ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Each node must be an object.")
        node_id = _validate_node_id(node.get("id"))
        if node_id in ids:
            raise ValueError(f"Duplicate node id '{node_id}'.")
        ids.add(node_id)

    adjacency: dict[str, list[str]] = defaultdict(list)
    indegree: dict[str, int] = {node_id: 0 for node_id in ids}
    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Each edge must be an object with 'from_id' and 'to_id'.")
        from_id = edge.get("from_id")
        to_id = edge.get("to_id")
        if not isinstance(from_id, str) or not isinstance(to_id, str):
            raise ValueError("Edge 'from_id' and 'to_id' must be strings.")
        if from_id not in ids:
            raise ValueError(f"Edge references unknown source node '{from_id}'.")
        if to_id not in ids:
            raise ValueError(f"Edge references unknown target node '{to_id}'.")
        if from_id == to_id:
            raise ValueError(f"Self-loop on node '{from_id}' is not allowed; the graph must be a DAG.")
        adjacency[from_id].append(to_id)
        indegree[to_id] += 1

    # Kahn's algorithm — if we can't peel all nodes off, the remainder is a cycle.
    queue = deque([node_id for node_id, deg in indegree.items() if deg == 0])
    processed = 0
    while queue:
        current = queue.popleft()
        processed += 1
        for neighbor in adjacency[current]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    if processed != len(ids):
        raise ValueError("Graph contains a cycle; graphs must be acyclic (DAG).")


def _build_agent_for_node(
    node: dict[str, Any],
    parent_agent: Any,
    parent_registry: dict[str, AgentTool],
) -> Agent:
    """Build one sub-agent for a graph node.

    * ``model`` defaults to the parent agent's model — this keeps sub-agents on a
      known, working provider without letting the model pick arbitrary strings.
    * ``system_prompt`` is optional and length-capped.
    * ``tools`` is a strict allow-list against the parent's registry.
    """
    node_id = node["id"]

    system_prompt = node.get("system_prompt")
    if system_prompt is not None:
        if not isinstance(system_prompt, str):
            raise ValueError(f"Node '{node_id}': 'system_prompt' must be a string.")
        if len(system_prompt) > MAX_SYSTEM_PROMPT_LENGTH:
            raise ValueError(f"Node '{node_id}': 'system_prompt' exceeds {MAX_SYSTEM_PROMPT_LENGTH} characters.")

    tools = _resolve_tools(node.get("tools"), parent_registry, node_id)

    return Agent(
        model=parent_agent.model,
        name=node_id,
        system_prompt=system_prompt,
        tools=list(tools),
    )


def _format_result(result: Any) -> tuple[str, int]:
    """Turn a NodeResult into ``(text, execution_time_ms)``.

    Extracts text blocks explicitly from the underlying ``AgentResult`` rather
    than delegating to ``AgentResult.__str__``, which stringifies interrupts
    and structured output ahead of the text content. Explicit extraction keeps
    the tool byte-parity with the TypeScript side's ``contentToText`` and
    means a future change to ``__str__`` cannot silently alter tool output.

    Falls back to a short error description when the node failed. The wording
    matches the TypeScript side ("error: <message>") so a caller consuming
    either SDK sees byte-identical output for the same failure.
    """
    execution_time = int(getattr(result, "execution_time", 0) or 0)
    inner = getattr(result, "result", None)

    if isinstance(inner, Exception):
        return f"error: {inner}", execution_time

    if inner is None:
        return "", execution_time

    # ``AgentResult`` exposes ``.message`` (a ``Message`` dict with ``content``,
    # a list of content blocks). Extract only text blocks; images, tool uses,
    # reasoning, and other structured blocks are dropped so the tool result
    # cannot smuggle non-text bytes back through the graph.
    message = getattr(inner, "message", None)
    if isinstance(message, dict):
        content_blocks = message.get("content", [])
        parts: list[str] = []
        if isinstance(content_blocks, list):
            for block in content_blocks:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "\n".join(parts).strip(), execution_time

    # Nested ``MultiAgentResult`` or an unrecognised shape: fall back to str().
    return str(inner), execution_time


def make_graph(
    *,
    name: str = "graph",
    description: str = GRAPH_DESCRIPTION,
    max_depth: int = _DEFAULT_MAX_DEPTH,
) -> DecoratedFunctionTool:
    """Create a graph tool bound to no specific parent.

    Args:
        name: Tool name shown to the model. Defaults to ``"graph"``.
        description: Tool description shown to the model.
        max_depth: Shared multi-agent recursion cap. Factory-only; never
            surfaced to the model.

    Returns:
        A decorated tool. Register it on an :class:`~strands.agent.Agent` like
        any other tool.
    """

    @tool(name=name, description=description, context="tool_context")
    async def graph_tool(
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        initial_input: str,
        tool_context: ToolContext,
    ) -> GraphOutput:
        """Execute a directed acyclic graph of sub-agents.

        Args:
            nodes: List of node specs. Each entry has:

                * ``id`` (required): unique short identifier (``[A-Za-z0-9_-]``, up to 64 chars).
                * ``system_prompt`` (optional): sub-agent system prompt.
                * ``tools`` (optional): list of tool names to expose to this node,
                  drawn from the parent agent's tool registry. No wildcards.
            edges: List of ``{"from_id": ..., "to_id": ...}`` entries defining the
                dependency order. The graph must be acyclic.
            initial_input: The task passed to entry-point nodes (nodes with no
                incoming edges).
            tool_context: Injected by the framework. Not user-facing.

        Raises:
            ValueError: If the topology is invalid (non-list ``nodes`` / ``edges``,
                duplicate or malformed node ids, self-loops, cycles, references to
                unknown nodes), if ``initial_input`` is not a string or exceeds the
                size cap, if a node's ``system_prompt`` exceeds the size cap, or if
                its ``tools`` field references an unknown tool, a wildcard, a
                multi-agent tool name, or exceeds the per-node tool cap.
            MultiagentDepthExceeded: If the shared multi-agent recursion depth
                counter (``invocation_state['multiagent_depth']``) is at or above
                the factory-configured cap on entry.
        """
        # Validation (bad-input) errors still raise past the tool boundary —
        # they signal a bug the model should see. Only cancellation is caught
        # and mapped to a ``{status: cancelled}`` result per spec.
        invocation_state = tool_context.invocation_state or {}
        depth = _current_depth(invocation_state)
        if depth >= max_depth:
            raise MultiagentDepthExceeded(
                f"graph refused: multi-agent recursion depth cap of {max_depth} reached (current depth {depth})"
            )

        if not isinstance(initial_input, str):
            raise ValueError("'initial_input' must be a string.")
        if len(initial_input) > MAX_INITIAL_INPUT_LENGTH:
            raise ValueError(f"'initial_input' exceeds {MAX_INITIAL_INPUT_LENGTH} characters.")

        _validate_nodes_edges(nodes, edges)

        parent_agent = tool_context.agent
        parent_registry_container = getattr(parent_agent, "tool_registry", None)
        parent_registry: dict[str, AgentTool] = (
            getattr(parent_registry_container, "registry", {}) if parent_registry_container else {}
        )

        parent_cancel_signal = getattr(parent_agent, "_cancel_signal", None)
        start_time = time.monotonic()
        if parent_cancel_signal is not None and parent_cancel_signal.is_set():
            # Pre-flight cancellation: return the cancelled sentinel instead of
            # raising. Spec (``_multiagent_conventions.md`` line 80): the tool
            # returns ``{status: cancelled}``, does not raise.
            return _cancelled_output(execution_time_ms=int((time.monotonic() - start_time) * 1000))

        builder = GraphBuilder()
        builder.set_max_node_executions(MAX_NODE_EXECUTIONS)
        builder.set_execution_timeout(DEFAULT_EXECUTION_TIMEOUT_SECONDS)
        builder.set_node_timeout(DEFAULT_NODE_TIMEOUT_SECONDS)

        for node in nodes:
            sub_agent = _build_agent_for_node(node, parent_agent, parent_registry)
            builder.add_node(sub_agent, node_id=node["id"])

        for edge in edges:
            builder.add_edge(edge["from_id"], edge["to_id"])

        graph_instance = builder.build()
        graph_instance.add_hook(_ParentCancelHook(parent_agent).on_before_node_call, BeforeNodeCallEvent)

        child_invocation_state: dict[str, Any] = {_DEPTH_KEY: depth + 1}

        # Hard wall-clock ceiling. ``builder.set_execution_timeout`` is checked
        # at node boundaries, so an N-node chain can silently exceed the cap.
        # ``asyncio.wait_for`` gives us a strict upper bound at exactly the
        # documented cap so the tool's guarantee matches its documentation.
        hard_ceiling = DEFAULT_EXECUTION_TIMEOUT_SECONDS

        try:
            result = await asyncio.wait_for(
                graph_instance.invoke_async(initial_input, invocation_state=child_invocation_state),
                timeout=hard_ceiling,
            )
        except (asyncio.TimeoutError, TimeoutError):
            # Hard-ceiling exceeded. Surface as cancelled — the tool boundary
            # doesn't distinguish "cancelled by parent" from "cancelled by the
            # tool's own hard timeout"; both mean "no useful result".
            return _cancelled_output(execution_time_ms=int((time.monotonic() - start_time) * 1000))
        except RuntimeError as exc:
            # ``Graph.invoke_async`` re-raises ``RuntimeError`` when a hook sets
            # ``event.cancel_node``. Match on the sentinel embedded in the hook's
            # message so an unrelated runtime failure that happens to coincide
            # with parent cancellation still propagates rather than being
            # silently mapped to a cancelled result.
            if _CANCEL_SENTINEL not in str(exc):
                raise
            return _cancelled_output(execution_time_ms=int((time.monotonic() - start_time) * 1000))

        results: dict[str, GraphNodeResult] = {}
        for node_id, node_result in result.results.items():
            text, execution_time_ms = _format_result(node_result)
            results[node_id] = GraphNodeResult(
                status=node_result.status.value,
                output=text,
                execution_time_ms=execution_time_ms,
            )

        top_output = _aggregate_terminal_output(nodes, edges, results)

        return GraphOutput(
            status=result.status.value if isinstance(result.status, Status) else str(result.status),
            output=top_output,
            execution_order=[node.node_id for node in result.execution_order],
            results=results,
            execution_time_ms=int(result.execution_time),
        )

    return cast("DecoratedFunctionTool", graph_tool)


def _aggregate_terminal_output(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    results: dict[str, GraphNodeResult],
) -> str:
    """Return the top-level ``output`` string per the shared multi-agent dialect.

    Rule: concatenate every terminal (leaf) node's ``output`` text, joined by a
    blank line, in graph declaration order. Terminal nodes are those with no
    outgoing edges — a graph without a single named sink typically returns
    multiple final outputs, and dropping any of them would silently truncate
    the tool's answer.
    """
    with_outgoing: set[str] = {edge["from_id"] for edge in edges if isinstance(edge.get("from_id"), str)}
    parts: list[str] = []
    for node in nodes:
        node_id = node["id"]
        if node_id in with_outgoing:
            continue
        node_result = results.get(node_id)
        if node_result is None:
            continue
        text = node_result["output"]
        if text:
            parts.append(text)
    return "\n\n".join(parts)


graph = make_graph()
