"""Deprecated: moved to ``strands.mcp.mcp_tasks``."""

import warnings

from ...mcp.mcp_tasks import *  # noqa: F401, F403
from ...mcp.mcp_tasks import (  # noqa: F401
    DEFAULT_TASK_CONFIG,
    DEFAULT_TASK_POLL_TIMEOUT,
    DEFAULT_TASK_TTL,
    TasksConfig,
)

warnings.warn(
    "strands.tools.mcp.mcp_tasks has moved to strands.mcp.mcp_tasks. "
    "Import from strands.mcp.mcp_tasks instead; strands.tools.mcp will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)
