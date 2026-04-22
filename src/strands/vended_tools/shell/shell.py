"""Shell tool implementation.

Executes shell commands in the agent's sandbox with persistent state tracking.
The tool delegates execution to ``tool_context.agent.sandbox.execute()`` and
reads configuration from ``tool_context.agent.state.get("strands_shell_tool")``.

Configuration keys (set via ``agent.state.set("strands_shell_tool", {...})``):

- ``require_confirmation`` (bool): When True, the tool raises an interrupt
  before executing each command, requiring human approval. Default: False.
- ``timeout`` (int): Default timeout in seconds. Overridden by the per-call
  ``timeout`` parameter. Default: 120.
"""

import asyncio
import logging
from typing import Any

from ...tools.decorator import tool
from ...types.tools import ToolContext

logger = logging.getLogger(__name__)

#: State key for shell tool configuration in agent.state
STATE_KEY = "strands_shell_tool"

#: Default timeout for shell commands (seconds)
DEFAULT_TIMEOUT = 120


def _get_config(tool_context: ToolContext) -> dict[str, Any]:
    """Read shell tool configuration from agent state.

    Args:
        tool_context: The tool context providing access to agent state.

    Returns:
        Configuration dict. Empty dict if no config is set.
    """
    return tool_context.agent.state.get(STATE_KEY) or {}


@tool(context=True)
async def shell(
    command: str,
    timeout: int | None = None,
    restart: bool = False,
    tool_context: ToolContext = None,  # type: ignore[assignment]
) -> str:
    """Execute a shell command in the agent's sandbox.

    The sandbox preserves working directory and environment variables across
    calls when using a persistent sandbox implementation. Use ``restart=True``
    to reset the shell state.

    Commands are executed via the agent's sandbox
    (``tool_context.agent.sandbox.execute()``). The sandbox may be a local
    host sandbox, a Docker container, a remote cloud sandbox, or any other
    implementation of the Sandbox interface.

    Configuration is read from ``agent.state.get("strands_shell_tool")``:

    - ``require_confirmation``: Interrupt before execution for human approval.
    - ``timeout``: Default timeout in seconds (overridden by per-call timeout).

    Args:
        command: The shell command to execute.
        timeout: Maximum execution time in seconds. Uses config default or 120s.
        restart: If True, reset shell state by clearing tracked working directory.
        tool_context: Framework-injected tool context.

    Returns:
        Command output. Includes stderr and exit code when non-zero.
    """
    config = _get_config(tool_context)
    sandbox = tool_context.agent.sandbox

    # Handle restart
    if restart:
        # Clear any tracked state
        _clear_shell_state(tool_context)
        if not command or not command.strip():
            return "Shell state reset."

    # Resolve timeout: per-call > config > default
    effective_timeout: int | None = timeout
    if effective_timeout is None:
        effective_timeout = config.get("timeout", DEFAULT_TIMEOUT)

    # Interrupt for confirmation if configured
    if config.get("require_confirmation"):
        approval = tool_context.interrupt(
            "shell_confirmation",
            reason={"command": command, "message": "Approve this shell command?"},
        )
        if approval != "approve":
            return f"Command not approved. Received: {approval}"

    # Get tracked working directory from state (for session continuity)
    shell_state = tool_context.agent.state.get("_strands_shell_state") or {}
    cwd = shell_state.get("cwd")

    # Execute via sandbox
    try:
        result = await sandbox.execute(
            command,
            timeout=effective_timeout,
            cwd=cwd,
        )
    except asyncio.TimeoutError:
        return f"Error: Command timed out after {effective_timeout} seconds."
    except NotImplementedError:
        return "Error: Sandbox does not support command execution (NoOpSandbox)."
    except Exception as e:
        return f"Error: {e}"

    # Track working directory changes
    # After each command, query the sandbox for cwd so cd persists
    try:
        cwd_result = await sandbox.execute("pwd", timeout=5, cwd=cwd)
        if cwd_result.exit_code == 0:
            new_cwd = cwd_result.stdout.strip()
            if new_cwd:
                shell_state["cwd"] = new_cwd
                tool_context.agent.state.set("_strands_shell_state", shell_state)
    except Exception:
        pass  # Best-effort cwd tracking

    # Format output
    output_parts = []
    if result.stdout:
        output_parts.append(result.stdout)
    if result.stderr:
        output_parts.append(result.stderr)

    output = "\n".join(output_parts).rstrip()

    if result.exit_code != 0:
        if output:
            output += f"\n\nExit code: {result.exit_code}"
        else:
            output = f"Command failed with exit code: {result.exit_code}"

    return output if output else "(no output)"


def _clear_shell_state(tool_context: ToolContext) -> None:
    """Clear tracked shell state from agent state.

    Args:
        tool_context: The tool context providing access to agent state.
    """
    try:
        tool_context.agent.state.delete("_strands_shell_state")
    except Exception:
        pass
