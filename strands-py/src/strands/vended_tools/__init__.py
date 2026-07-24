"""Built-in tools for commands, files, HTTP, and pausing.

The :data:`bash` tool runs a
persistent shell on the host; the :func:`make_bash` and :func:`make_file_editor`
factories produce sandbox-routed tools that either bind to a
:class:`~strands.sandbox.base.Sandbox` at creation (as the built-in Docker/SSH
sandboxes do when vending tools) or read the sandbox from the agent at call time.
The :data:`sleep` tool pauses execution for a bounded, cancellable duration.

The :data:`http_request` tool makes raw HTTP calls with a strict default
security posture (private-network denial, redirect and body-size caps,
sensitive-header rejection); use :func:`make_http_request` to relax individual
controls when needed.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import bash, file_editor, http_request, sleep

    agent = Agent(tools=[bash, file_editor, http_request, sleep])
    ```
"""

from .bash import bash, make_bash
from .file_editor import file_editor, make_file_editor
from .http_request import http_request, make_http_request
from .sleep import make_sleep, sleep

__all__ = [
    "bash",
    "file_editor",
    "http_request",
    "make_bash",
    "make_file_editor",
    "make_http_request",
    "make_sleep",
    "sleep",
]
