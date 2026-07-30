"""Shared types and constants for the shell tool."""

from typing import TypedDict


class ShellOutput(TypedDict):
    """Output of a shell command execution.

    Attributes:
        output: Standard output captured from the command.
        error: Standard error captured from the command. Empty when there was none.
    """

    output: str
    error: str


SANDBOX_SHELL_DESCRIPTION = (
    "Executes shell commands. Each call runs in a fresh shell; "
    "state such as variables and the working directory does not persist across calls."
)
"""Description for the shell tool."""
