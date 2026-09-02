"""Shared types and constants for the notebook tool."""

from typing import TypedDict


class NotebookState(TypedDict):
    """State structure for notebook storage.

    Notebooks are stored in the agent's state under the ``notebooks`` key.
    Each notebook stores plain text content with newline-separated lines.
    """

    notebooks: dict[str, str]


DEFAULT_NOTEBOOK_DESCRIPTION = (
    "Manages text notebooks for note-taking and documentation. Supports create, list, read, "
    "write (replace or insert), and clear operations. Notebooks persist across invocations "
    "within a session, and across sessions when the agent has a durable state store."
)
"""Description for the notebook tool."""

DEFAULT_NOTEBOOK_NAME = "default"
"""Name of the default notebook used when no name is provided."""

MAX_NOTEBOOKS = 64
"""Maximum number of notebooks that may exist in a single session."""

MAX_NOTEBOOK_NAME_LENGTH = 128
"""Maximum length of a notebook name in characters."""

MAX_NOTEBOOK_SIZE_BYTES = 1_048_576  # 1 MiB
"""Maximum size of any single notebook's content in bytes (UTF-8)."""

MAX_TOTAL_SIZE_BYTES = 8 * 1_048_576  # 8 MiB
"""Maximum combined size across all notebooks in a session, in bytes (UTF-8)."""
