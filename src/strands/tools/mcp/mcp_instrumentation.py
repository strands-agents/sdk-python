"""Deprecated: moved to ``strands.mcp.mcp_instrumentation``."""

import warnings

from ...mcp.mcp_instrumentation import *  # noqa: F401, F403
from ...mcp.mcp_instrumentation import mcp_instrumentation  # noqa: F401

warnings.warn(
    "strands.tools.mcp.mcp_instrumentation has moved to strands.mcp.mcp_instrumentation. "
    "Import from strands.mcp.mcp_instrumentation instead; strands.tools.mcp will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
