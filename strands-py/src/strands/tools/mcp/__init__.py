"""Deprecated: MCP integration has moved to ``strands.mcp``.

This module re-exports the public API from its new location and emits a
``DeprecationWarning`` at import time. Legacy submodule paths such as
``strands.tools.mcp.mcp_client`` continue to resolve to the canonical
modules under ``strands.mcp`` via ``sys.modules`` aliasing. Update
imports to ``strands.mcp``; this alias will be removed in a future
release.
"""

import sys as _sys
import warnings as _warnings

from strands import mcp as _mcp
from strands.mcp import MCPAgentTool, MCPClient, MCPTransport, TasksConfig, ToolFilters

for _submod_name in ("mcp_agent_tool", "mcp_client", "mcp_instrumentation", "mcp_tasks", "mcp_types"):
    _sys.modules[f"strands.tools.mcp.{_submod_name}"] = getattr(_mcp, _submod_name)

_warnings.warn(
    "strands.tools.mcp has moved to strands.mcp. "
    "Import from strands.mcp instead; strands.tools.mcp will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["MCPAgentTool", "MCPClient", "MCPTransport", "TasksConfig", "ToolFilters"]
