"""Deprecated: moved to ``strands.mcp.mcp_types``."""

import warnings

from ...mcp.mcp_types import *  # noqa: F401, F403
from ...mcp.mcp_types import MCPToolResult, MCPTransport  # noqa: F401

warnings.warn(
    "strands.tools.mcp.mcp_types has moved to strands.mcp.mcp_types. "
    "Import from strands.mcp.mcp_types instead; strands.tools.mcp will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
