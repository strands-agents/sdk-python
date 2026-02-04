"""Task-augmented tool execution configuration for MCP.

This module provides configuration types and defaults for the experimental MCP Tasks feature.
"""

from datetime import timedelta

from typing_extensions import TypedDict


class TasksConfig(TypedDict, total=False):
    """Configuration for MCP Tasks (task-augmented tool execution).

    If this config is provided (not None), task-augmented execution is enabled.
    When enabled, supported tool calls use the MCP task workflow:
    create task -> poll for completion -> get result.

    Warning:
        This feature is experimental and subject to change in future revisions without notice.

    Attributes:
        ttl: Task time-to-live. Defaults to 1 minute.
        poll_timeout: Timeout for polling task completion. Defaults to 5 minutes.
    """

    ttl: timedelta
    poll_timeout: timedelta


DEFAULT_TASK_TTL = timedelta(minutes=1)
DEFAULT_TASK_POLL_TIMEOUT = timedelta(minutes=5)
DEFAULT_TASK_CONFIG = TasksConfig(ttl=DEFAULT_TASK_TTL, poll_timeout=DEFAULT_TASK_POLL_TIMEOUT)
