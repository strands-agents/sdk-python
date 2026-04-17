"""Deprecated: moved to ``strands.mcp.mcp_client``."""

import warnings

from ...mcp.mcp_client import *  # noqa: F401, F403
from ...mcp.mcp_client import MCPClient, ToolFilters  # noqa: F401

warnings.warn(
    "strands.tools.mcp.mcp_client has moved to strands.mcp.mcp_client. "
    "Import from strands.mcp.mcp_client instead; strands.tools.mcp will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
