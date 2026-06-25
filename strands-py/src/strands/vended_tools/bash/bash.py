"""Bash tool for executing shell commands through a sandbox.

Provides :func:`make_bash` (a factory for a stateless, sandbox-routed bash tool)
and :data:`bash` (the default instance that reads the sandbox from the agent at
call time). Each call runs in a fresh shell; state such as variables and the
working directory does not persist across calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...sandbox.errors import SandboxTimeoutError
from ...tools.decorator import tool
from ...types.tools import ToolContext
from .types import SANDBOX_BASH_DESCRIPTION, BashOutput

if TYPE_CHECKING:
    from ...sandbox.base import Sandbox
    from ...tools.decorator import DecoratedFunctionTool

_DEFAULT_TIMEOUT = 120


def make_bash(
    *,
    sandbox: Sandbox | None = None,
    name: str = "bash",
    description: str = SANDBOX_BASH_DESCRIPTION,
) -> DecoratedFunctionTool:
    """Create a stateless, sandbox-routed bash tool.

    If a ``sandbox`` is passed, it is bound at creation time. Otherwise the tool
    reads the sandbox from ``tool_context.agent.sandbox`` at call time. Used by
    sandbox implementations in :meth:`~strands.sandbox.base.Sandbox.get_tools`
    and by users who want a customized bash tool.

    Args:
        sandbox: Sandbox to bind at creation. When ``None``, the agent's
            configured sandbox is used at call time.
        name: Tool name. Defaults to ``"bash"``.
        description: Tool description shown to the model.

    Returns:
        A decorated tool that executes shell commands through the sandbox.
    """

    @tool(name=name, description=description, context="tool_context")
    async def bash_tool(command: str, tool_context: ToolContext, timeout: int = _DEFAULT_TIMEOUT) -> BashOutput:
        """Executes a bash shell command and returns its output.

        Args:
            command: The bash command to execute.
            tool_context: Injected by the framework. Not user-facing.
            timeout: Timeout in seconds (default: 120).
        """
        active = sandbox if sandbox is not None else tool_context.agent.sandbox
        try:
            result = await active.execute(command, timeout=timeout)
        except SandboxTimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(str(e)) from e
        return {"output": result.stdout, "error": result.stderr}

    return bash_tool


bash = make_bash()
"""Default bash tool. Reads the sandbox from the agent's context at call time."""
