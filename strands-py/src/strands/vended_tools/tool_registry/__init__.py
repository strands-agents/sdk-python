"""Runtime tool-registry-management tool (CRUD over the agent's own tools).

Only tools hosted on pre-approved MCP servers may be registered dynamically.
Loading tools from a file path or inline source is intentionally not supported.

Example Usage:
    ```python
    from strands import Agent
    from strands.tools.mcp import MCPClient
    from strands.vended_tools.tool_registry import make_tool_registry

    with MCPClient(...) as weather:
        registry_tool = make_tool_registry(mcp_clients={"weather": weather})
        agent = Agent(tools=[registry_tool])
    ```
"""

from .tool_registry import make_tool_registry
from .types import (
    ListResult,
    MutationResult,
    RegisteredTool,
    ToolRegistryError,
)

__all__ = [
    "ListResult",
    "MutationResult",
    "RegisteredTool",
    "ToolRegistryError",
    "make_tool_registry",
]
