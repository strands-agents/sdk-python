"""Python REPL tool implementation.

Executes Python code in the agent's sandbox using
``sandbox.execute_code(code, language="python")``. Maintains state across
calls by tracking the sandbox session.

Configuration keys (set via ``agent.state.set("strands_python_repl_tool", {...})``):

- ``require_confirmation`` (bool): When True, the tool raises an interrupt
  before executing code, requiring human approval. Default: False.
- ``timeout`` (int): Default timeout in seconds for code execution.
  Overridden by the per-call ``timeout`` parameter. Default: 30.
"""

import asyncio
import logging
from typing import Any

from ...tools.decorator import tool
from ...types.tools import ToolContext

logger = logging.getLogger(__name__)

#: State key for python_repl tool configuration in agent.state
STATE_KEY = "strands_python_repl_tool"

#: Default timeout for code execution (seconds)
DEFAULT_TIMEOUT = 30


def _get_config(tool_context: ToolContext) -> dict[str, Any]:
    """Read python_repl tool configuration from agent state."""
    return tool_context.agent.state.get(STATE_KEY) or {}


@tool(context=True)
async def python_repl(
    code: str,
    timeout: int | None = None,
    reset: bool = False,
    tool_context: ToolContext = None,  # type: ignore[assignment]
) -> str:
    """Execute Python code in the agent's sandbox.

    Code is executed via the agent's sandbox using
    ``sandbox.execute_code(code, language="python")``. The sandbox determines
    the execution environment — it may be a local Python interpreter, a Docker
    container, or a remote cloud sandbox.

    Use ``reset=True`` to clear any sandbox-level state (e.g., restart the
    interpreter session if the sandbox supports it).

    Configuration is read from ``agent.state.get("strands_python_repl_tool")``:

    - ``require_confirmation``: Interrupt before execution for human approval.
    - ``timeout``: Default timeout in seconds (overridden by per-call timeout).

    Args:
        code: The Python code to execute.
        timeout: Maximum execution time in seconds. Uses config default or 30s.
        reset: If True, signal the sandbox to reset execution state.
        tool_context: Framework-injected tool context.

    Returns:
        Code output (stdout + stderr). Includes exit code when non-zero.
    """
    config = _get_config(tool_context)
    sandbox = tool_context.agent.sandbox

    # Handle reset
    if reset:
        # Clear any tracked state
        try:
            tool_context.agent.state.delete("_strands_python_repl_state")
        except Exception:
            pass
        if not code or not code.strip():
            return "Python REPL state reset."

    # Resolve timeout: per-call > config > default
    effective_timeout: int | None = timeout
    if effective_timeout is None:
        effective_timeout = config.get("timeout", DEFAULT_TIMEOUT)

    # Interrupt for confirmation if configured
    if config.get("require_confirmation"):
        # Show a preview of the code (truncated for readability)
        code_preview = code[:500] + ("..." if len(code) > 500 else "")
        approval = tool_context.interrupt(
            "python_repl_confirmation",
            reason={"code": code_preview, "message": "Approve this Python code execution?"},
        )
        if approval != "approve":
            return f"Code execution not approved. Received: {approval}"

    # Execute via sandbox
    try:
        result = await sandbox.execute_code(
            code,
            language="python",
            timeout=effective_timeout,
        )
    except asyncio.TimeoutError:
        return f"Error: Code execution timed out after {effective_timeout} seconds."
    except NotImplementedError:
        return "Error: Sandbox does not support code execution (NoOpSandbox)."
    except Exception as e:
        return f"Error: {e}"

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
            output = f"Code execution failed with exit code: {result.exit_code}"

    # Handle output files (images, charts, etc.)
    if result.output_files:
        file_names = [f.name for f in result.output_files]
        output += f"\n\nGenerated files: {', '.join(file_names)}"

    return output if output else "(no output)"
