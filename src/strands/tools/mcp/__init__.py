"""Deprecated: MCP integration has moved to ``strands.mcp``.

This module re-exports the public API from its new location and emits a
``DeprecationWarning`` at import time. Update imports to ``strands.mcp``.
"""

import warnings

from ...mcp import MCPAgentTool, MCPClient, MCPTransport, TasksConfig, ToolFilters

warnings.warn(
    "strands.tools.mcp has moved to strands.mcp. "
    "Import from strands.mcp instead; strands.tools.mcp will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["MCPAgentTool", "MCPClient", "MCPTransport", "TasksConfig", "ToolFilters"]
