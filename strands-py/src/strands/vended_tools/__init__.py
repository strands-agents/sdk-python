"""Built-in tools for commands, files, HTTP, and pausing.

The :func:`make_shell` and :func:`make_file_editor` factories produce
sandbox-routed tools that either bind to a
:class:`~strands.sandbox.base.Sandbox` at creation (as the built-in Docker/SSH
sandboxes do when vending tools) or read the sandbox from the agent at call time.
Each :data:`shell` call runs in a fresh shell, so state does not persist between
calls. The :data:`sleep` tool pauses execution for a bounded, cancellable duration.

The :data:`http_request` tool makes raw HTTP calls; use
:func:`make_http_request` to supply a pre-configured ``httpx.AsyncClient``
with custom timeouts, redirects, authentication, or proxies.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import file_editor, http_request, shell, sleep

    agent = Agent(tools=[file_editor, http_request, shell, sleep])
    ```
"""

import warnings
from typing import TYPE_CHECKING, Any

from typing_extensions import deprecated

from .file_editor import file_editor, make_file_editor
from .http_request import http_request, make_http_request
from .shell import make_shell, shell
from .sleep import make_sleep, sleep

if TYPE_CHECKING:
    from ..tools.decorator import DecoratedFunctionTool

_RENAME_RATIONALE = (
    "The tool routes commands through the sandbox, which runs sh or the remote login shell "
    "rather than bash specifically."
)


@deprecated(f"make_bash is deprecated and will be removed in v2.0.0. Use make_shell instead. {_RENAME_RATIONALE}")
def make_bash(**kwargs: Any) -> "DecoratedFunctionTool":
    """Deprecated alias for :func:`make_shell`.

    Defaults the tool name to ``"bash"`` so callers keyed on the pre-rename name
    (hooks, prompts, tool lists) keep working until the alias is removed.
    """
    kwargs.setdefault("name", "bash")
    return make_shell(**kwargs)


_bash: "DecoratedFunctionTool | None" = None


def __getattr__(name: str) -> Any:
    # ``bash`` is a tool instance rather than a function, so it cannot carry the
    # @deprecated decorator that ``make_bash`` uses; resolve it here instead.
    if name == "bash":
        warnings.warn(
            f"bash is deprecated and will be removed in v2.0.0. Use shell instead. {_RENAME_RATIONALE}",
            DeprecationWarning,
            stacklevel=2,
        )
        # A shell tool that still registers under the pre-rename name, so string
        # matches on "bash" (hooks, defaults lists, prompts) survive the upgrade.
        global _bash
        if _bash is None:
            _bash = make_shell(name="bash")
        return _bash
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "file_editor",
    "http_request",
    "make_file_editor",
    "make_http_request",
    "make_shell",
    "make_sleep",
    "shell",
    "sleep",
]
