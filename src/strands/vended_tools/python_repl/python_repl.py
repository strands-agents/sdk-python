"""Python REPL tool implementation with streaming support.

Executes Python code in the agent's sandbox using
``sandbox.execute_code_streaming(code, language="python")``. Each chunk of
stdout/stderr is yielded as a ``ToolStreamEvent`` in real time, allowing UI
consumers to display live output from code execution.

The tool is an **async generator**: ``StreamChunk`` objects from the sandbox
are yielded during execution, and the final yield is the formatted result
string that becomes the ``ToolResult``.

Configuration keys (set via ``agent.state.set("strands_python_repl_tool", {...})``):

- ``timeout`` (int): Default timeout in seconds for code execution.
  Overridden by the per-call ``timeout`` parameter. Default: 30.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from ...sandbox.base import ExecutionResult, StreamChunk
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
) -> AsyncGenerator[Any, None]:
    """Execute Python code in the agent's sandbox with live output streaming.

    Code is executed via the agent's sandbox using
    ``sandbox.execute_code_streaming(code, language="python")``. Each chunk
    of stdout/stderr is yielded as a streaming event that UI consumers can
    display in real time. The final yield is the formatted result string.

    Use ``reset=True`` to clear any sandbox-level state (e.g., restart the
    interpreter session if the sandbox supports it).

    Configuration is read from ``agent.state.get("strands_python_repl_tool")``:

    - ``timeout``: Default timeout in seconds (overridden by per-call timeout).

    Args:
        code: The Python code to execute.
        timeout: Maximum execution time in seconds. Uses config default or 30s.
        reset: If True, signal the sandbox to reset execution state.
        tool_context: Framework-injected tool context.

    Yields:
        :class:`~strands.sandbox.base.StreamChunk` objects during execution (wrapped as
        ``ToolStreamEvent`` by the SDK), then a final string result.
    """
    config = _get_config(tool_context)
    sandbox = tool_context.agent.sandbox

    # Handle reset
    if reset:
        try:
            tool_context.agent.state.delete("_strands_python_repl_state")
        except Exception:
            pass
        if not code or not code.strip():
            yield "Python REPL state reset."
            return

    # Resolve timeout: per-call > config > default
    effective_timeout: int | None = timeout
    if effective_timeout is None:
        effective_timeout = config.get("timeout", DEFAULT_TIMEOUT)

    # Execute via sandbox streaming
    result: ExecutionResult | None = None
    try:
        async for chunk in sandbox.execute_code_streaming(
            code,
            language="python",
            timeout=effective_timeout,
        ):
            if isinstance(chunk, StreamChunk):
                # Yield each chunk — the decorator wraps it as ToolStreamEvent
                yield chunk
            elif isinstance(chunk, ExecutionResult):
                result = chunk
    except asyncio.TimeoutError:
        yield f"Error: Code execution timed out after {effective_timeout} seconds."
        return
    except NotImplementedError:
        yield "Error: Sandbox does not support code execution (NoOpSandbox)."
        return
    except Exception as e:
        yield f"Error: {e}"
        return

    if result is None:
        yield "Error: Sandbox did not return an execution result."
        return

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
        if output:
            output += f"\n\nGenerated files: {', '.join(file_names)}"
        else:
            output = f"Generated files: {', '.join(file_names)}"

    yield output if output else "(no output)"
