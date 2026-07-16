"""Graph tool for deterministic DAG-based multi-agent orchestration.

The tool lets a caller agent describe a directed acyclic graph of sub-agents
inline (nodes with optional system prompts and per-node tool allow-lists;
edges giving the dependency order) and executes it via the SDK's
:class:`~strands.multiagent.graph.Graph` primitive. Sub-agents inherit the
parent agent's model instance — per-node model overrides are not supported.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import graph

    agent = Agent(tools=[graph])
    ```
"""

from .graph import graph, make_graph

__all__ = [
    "graph",
    "make_graph",
]
