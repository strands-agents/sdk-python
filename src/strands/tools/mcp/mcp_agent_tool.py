"""Deprecated: moved to ``strands.mcp.mcp_agent_tool``."""

import warnings

from ...mcp.mcp_agent_tool import *  # noqa: F401, F403
from ...mcp.mcp_agent_tool import MCPAgentTool  # noqa: F401

warnings.warn(
    "strands.tools.mcp.mcp_agent_tool has moved to strands.mcp.mcp_agent_tool. "
    "Import from strands.mcp.mcp_agent_tool instead; strands.tools.mcp will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
