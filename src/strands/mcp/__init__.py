"""Canonical import path for the Model Context Protocol (MCP) integration.

Model Context Protocol functionality is being promoted from
``strands.tools.mcp`` to this top-level ``strands.mcp`` package because MCP
now spans tools, prompts, resources, tasks, and elicitation -- concepts
that extend beyond ``tools``.

For now this package is a thin re-export of ``strands.tools.mcp``. A
follow-up change will invert the relationship: the implementation will
live here and ``strands.tools.mcp`` will become a deprecated alias.
Users can safely migrate imports to ``strands.mcp`` today; the public
API is identical and object identity is preserved.
"""

from ..tools.mcp import MCPAgentTool, MCPClient, MCPTransport, TasksConfig, ToolFilters

__all__ = ["MCPAgentTool", "MCPClient", "MCPTransport", "TasksConfig", "ToolFilters"]
