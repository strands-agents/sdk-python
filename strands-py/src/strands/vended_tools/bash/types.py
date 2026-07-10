"""Shared types and constants for the bash tool."""

from typing import TypedDict


class BashOutput(TypedDict):
    """Output of a bash command execution.

    Attributes:
        output: Standard output captured from the command.
        error: Standard error captured from the command. Empty when there was none.
    """

    output: str
    error: str


SANDBOX_BASH_DESCRIPTION = (
    "Executes bash shell commands. Each call runs in a fresh shell; "
    "state such as variables and the working directory does not persist across calls."
)
"""Description for the bash tool."""
