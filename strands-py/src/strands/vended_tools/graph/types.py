"""Shared types, constants, and validation helpers for the graph tool."""

from __future__ import annotations

from typing import TypedDict

# Caps sized to keep the DAG a coordination surface, not a job runner.
# Anything meaningfully larger belongs in a user-authored ``Graph`` outside the tool.
MAX_NODES = 20
"""Maximum number of nodes the tool will accept in one call."""

MAX_EDGES = 40
"""Maximum number of edges the tool will accept in one call."""

MAX_ID_LENGTH = 64
"""Maximum length for a node id."""

MAX_SYSTEM_PROMPT_LENGTH = 8_000
"""Maximum length (chars) of any node's ``system_prompt``."""

MAX_INITIAL_INPUT_LENGTH = 32_000
"""Maximum length (chars) of the ``initial_input`` passed to the graph."""

MAX_NODE_EXECUTIONS = 40
"""Total node-execution budget passed to the underlying :class:`~strands.multiagent.graph.Graph`."""

MAX_TOOLS_PER_NODE = 64
"""Maximum number of tool names a single node may list in its allow-list."""

DEFAULT_EXECUTION_TIMEOUT_SECONDS = 300.0
"""Wall-clock budget for the whole graph invocation, in seconds."""

DEFAULT_NODE_TIMEOUT_SECONDS = 120.0
"""Wall-clock budget for each node in the graph, in seconds."""

GRAPH_DESCRIPTION = (
    "Runs a deterministic directed acyclic graph (DAG) of sub-agents. "
    "You describe the nodes (each with an optional system prompt and tool "
    "allow-list) and the edges (dependency order); the tool executes the "
    "graph, feeding each node the outputs of its dependencies, and returns "
    "the results keyed by node id. Use when a task has a fixed pipeline shape "
    "with clear dependencies; use plain tool calls for single-step work."
)
"""Description shown to the model."""


class GraphNodeResult(TypedDict):
    """One node's result in the aggregated graph output.

    Attributes:
        status: Node status (``completed``, ``failed``, ``interrupted``, etc.).
        output: Text output the node produced. Empty when the node failed
            before producing content.
        execution_time_ms: Node wall-clock time in milliseconds.
    """

    status: str
    output: str
    execution_time_ms: int


class GraphOutput(TypedDict):
    """Aggregated result of a graph invocation.

    Shape follows the shared multi-agent dialect: ``status``, ``output``, and
    ``execution_time_ms`` are the common contract; ``execution_order`` and
    ``results`` are the graph-specific extensions.

    ``output`` is the concatenation of every terminal (leaf) node's text, joined
    by a blank line — see ``_multiagent_conventions.md`` for the rationale.
    """

    status: str
    output: str
    execution_order: list[str]
    results: dict[str, GraphNodeResult]
    execution_time_ms: int
