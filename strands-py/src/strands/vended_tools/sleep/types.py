"""Shared types and constants for the sleep tool."""

DEFAULT_MAX_DURATION = 60.0
"""Default upper bound on ``duration`` (seconds) accepted by :func:`make_sleep`."""

SLEEP_DESCRIPTION = (
    "Pauses execution for a specified number of seconds. Cooperative and cancellable: "
    "the sleep aborts immediately when the agent invocation is cancelled. "
    "Rejects negative, NaN, infinite, or non-numeric durations, and durations "
    "above the tool's configured maximum."
)
"""Description for the sleep tool."""
