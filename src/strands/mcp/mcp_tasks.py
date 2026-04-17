"""Task-augmented tool execution configuration for MCP.

This module provides configuration types and defaults for the experimental MCP Tasks feature.
"""

from datetime import timedelta

from typing_extensions import TypedDict


class TasksConfig(TypedDict, total=False):
    """Configuration for MCP Tasks (task-augmented tool execution).

    When enabled, supported tool calls use the MCP task workflow:
    create task -> poll for completion -> get result.

    Warning:
        This is an experimental feature in the 2025-11-25 MCP specification and
        both the specification and the Strands Agents implementation of this
        feature are subject to change.

    Attributes:
        ttl: Task time-to-live. Defaults to 1 minute.
        poll_timeout: Timeout for polling task completion. Defaults to 5 minutes.
    """

    ttl: timedelta
    poll_timeout: timedelta


DEFAULT_TASK_TTL = timedelta(minutes=1)
DEFAULT_TASK_POLL_TIMEOUT = timedelta(minutes=5)
DEFAULT_TASK_CONFIG = TasksConfig(ttl=DEFAULT_TASK_TTL, poll_timeout=DEFAULT_TASK_POLL_TIMEOUT)
