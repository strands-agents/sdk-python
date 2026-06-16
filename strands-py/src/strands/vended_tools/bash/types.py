"""Shared types and constants for the bash tools.

Shared types and constants for the bash tools. Used by both the
host-session :data:`~strands.vended_tools.bash.bash.bash` tool and the
sandbox-routed :func:`~strands.vended_tools.bash.bash.make_bash` factory.
"""

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
"""Description for the sandbox-routed, stateless bash tool."""


class BashTimeoutError(Exception):
    """Raised when a bash command exceeds its timeout."""


class BashSessionError(Exception):
    """Raised when a bash session encounters an error."""
