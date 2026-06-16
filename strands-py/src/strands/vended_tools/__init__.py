"""Built-in tools for executing commands and editing files.

These tools mirror ``strands-ts/src/vended-tools/``. The :data:`bash` tool runs a
persistent shell on the host; the :func:`make_bash` and :func:`make_file_editor`
factories produce sandbox-routed tools that either bind to a
:class:`~strands.sandbox.base.Sandbox` at creation (as the built-in Docker/SSH
sandboxes do when vending tools) or read the sandbox from the agent at call time.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import bash, file_editor

    agent = Agent(tools=[bash, file_editor])
    ```
"""

from .bash import (
    DEFAULT_BASH_DESCRIPTION,
    SANDBOX_BASH_DESCRIPTION,
    BashSessionError,
    BashTimeoutError,
    bash,
    make_bash,
)
from .file_editor import DEFAULT_FILE_EDITOR_DESCRIPTION, file_editor, make_file_editor

__all__ = [
    "DEFAULT_BASH_DESCRIPTION",
    "DEFAULT_FILE_EDITOR_DESCRIPTION",
    "SANDBOX_BASH_DESCRIPTION",
    "BashSessionError",
    "BashTimeoutError",
    "bash",
    "file_editor",
    "make_bash",
    "make_file_editor",
]
