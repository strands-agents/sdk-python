"""Internal background task snapshots."""

from __future__ import annotations

from typing import Literal, TypeAlias

from typing_extensions import NotRequired, TypedDict

from strands.types.tools import ToolResultContent

BackgroundTaskStatus: TypeAlias = Literal[
    "queued",
    "working",
    "input_required",
    "completed",
    "failed",
    "cancelled",
]
BackgroundTaskFailureType: TypeAlias = Literal["tool_error", "execution_error", "timeout"]


class BackgroundTaskResult(TypedDict):
    """Visible background task result."""

    content: list[ToolResultContent]


class BackgroundTaskError(TypedDict):
    """Background task failure details."""

    type: BackgroundTaskFailureType
    message: str


class BackgroundTaskInterrupt(TypedDict):
    """Visible unanswered background task interrupt."""

    id: str
    name: str
    reason: object
    source: Literal["middleware", "tool"]


class BackgroundTask(TypedDict):
    """Snapshot of one background task."""

    task_id: str
    tool_use_id: str
    tool_name: str
    status: BackgroundTaskStatus
    created_at: str
    last_updated_at: str
    result: NotRequired[BackgroundTaskResult]
    error: NotRequired[BackgroundTaskError]
    interrupts: NotRequired[list[BackgroundTaskInterrupt]]


def is_task_status_terminal(status: BackgroundTaskStatus) -> bool:
    """Return whether a task status is terminal."""
    return status in {"completed", "failed", "cancelled"}
