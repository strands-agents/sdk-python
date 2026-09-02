"""Shared types and constants for the notebook tool."""

from typing import TypedDict


class NotebookState(TypedDict):
    """State structure for notebook storage."""

    notebooks: dict[str, str]


DEFAULT_NOTEBOOK_DESCRIPTION = (
    "Manages text notebooks for note-taking and documentation. Supports create, list, read, "
    "write (replace or insert), and clear operations. Notebooks persist across invocations "
    "within a session, and across sessions when the agent has a durable state store."
)
"""Description for the notebook tool."""
