"""Shell tool implementation with streaming support.

Executes shell commands in the agent's sandbox with persistent state tracking.
The tool uses ``sandbox.execute_streaming()`` so that stdout/stderr chunks are
yielded as ``ToolStreamEvent``s in real time. This allows UI consumers to display
live output from sandbox execution.

The tool is an **async generator**: each ``StreamChunk`` from the sandbox is
yielded directly (the SDK decorator wraps it in a ``ToolStreamEvent``), and the
final yield is the formatted result string (which becomes the ``ToolResult``).

Configuration keys (set via ``agent.state.set("strands_shell_tool", {...})``):

- ``timeout`` (int): Default timeout in seconds. Overridden by the per-call
  ``timeout`` parameter. Default: 120.

Example::

    from strands import Agent
    from strands.vended_tools import shell

    agent = Agent(tools=[shell])
    agent("List all Python files in the current directory")

    # Configure timeout
    agent.state.set("strands_shell_tool", {"timeout": 60})
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from ..sandbox.base import ExecutionResult, StreamChunk
from ..tools.decorator import tool
from ..types.tools import ToolContext
from ._utils import get_tool_config

logger = logging.getLogger(__name__)

#: State key for shell tool configuration in agent.state
STATE_KEY = "strands_shell_tool"

#: Default timeout for shell commands (seconds)
DEFAULT_TIMEOUT = 120

#: Internal marker used to separate user output from cwd tracking.
#: Must be unique enough to never appear in legitimate command output.
_CWD_MARKER = "__STRANDS_CWD__"


def _safe_yield_length(buffer: str, marker: str) -> int:
    """Return the number of chars from the start of buffer that are safe to yield.

    "Safe" means: no suffix of the yielded portion could be a prefix of the marker.
    This prevents partial marker leakage when the marker is split across chunks.
    """
    # Check if the buffer contains the full marker — if so, only yield up to it
    marker_pos = buffer.find(marker)
    if marker_pos != -1:
        return marker_pos

    # Check if the end of the buffer matches a prefix of the marker.
    # e.g., buffer ends with "__STRAN" which is a prefix of "__STRANDS_CWD__"
    max_overlap = min(len(marker) - 1, len(buffer))
    for i in range(max_overlap, 0, -1):
        if buffer.endswith(marker[:i]):
            return len(buffer) - i

    # No overlap — everything is safe
    return len(buffer)


@tool(context=True)
async def shell(
    command: str,
    timeout: int | None = None,
    restart: bool = False,
    tool_context: ToolContext = None,  # type: ignore[assignment]
) -> AsyncGenerator[Any, None]:
    """Execute a shell command in the agent's sandbox with live output streaming.

    The sandbox preserves working directory and environment variables across
    calls when using a persistent sandbox implementation. Use ``restart=True``
    to reset the shell state.

    Commands are executed via the agent's sandbox
    (``sandbox.execute_streaming()``). Each chunk of stdout/stderr is yielded
    as a streaming event that UI consumers can display in real time. The final
    yield is the formatted result string.

    Configuration is read from ``agent.state.get("strands_shell_tool")``:

    - ``timeout``: Default timeout in seconds (overridden by per-call timeout).

    Args:
        command: The shell command to execute.
        timeout: Maximum execution time in seconds. Uses config default or 120s.
        restart: If True, reset shell state by clearing tracked working directory.
        tool_context: Framework-injected tool context.

    Yields:
        :class:`~strands.sandbox.base.StreamChunk` objects during execution (wrapped as
        ``ToolStreamEvent`` by the SDK), then a final string result.
    """
    config = get_tool_config(tool_context, STATE_KEY)
    sandbox = tool_context.agent.sandbox

    # Handle restart
    if restart:
        _clear_shell_state(tool_context)
        if not command or not command.strip():
            yield "Shell state reset."
            return

    # Resolve timeout: per-call > config > default
    effective_timeout: int | None = timeout
    if effective_timeout is None:
        effective_timeout = config.get("timeout", DEFAULT_TIMEOUT)

    # Coerce timeout to int — JSON configs and LLMs may pass strings or floats
    if effective_timeout is not None:
        try:
            effective_timeout = int(effective_timeout)
        except (TypeError, ValueError):
            effective_timeout = DEFAULT_TIMEOUT

    # Get tracked working directory from state (for session continuity)
    shell_state = tool_context.agent.state.get("_strands_shell_state") or {}
    cwd = shell_state.get("cwd")

    # Append cwd tracking to the command. We use a unique marker so we can
    # reliably split the actual output from the cwd line. This captures the
    # final working directory even after `cd` commands (which only affect
    # the shell process they run in — a separate pwd call would not see them).
    tracked_command = f"{command}; echo {_CWD_MARKER}; pwd"

    # Streaming with marker-aware buffering:
    # We accumulate stdout in a buffer and yield only the "safe" prefix — i.e.,
    # bytes that cannot possibly be part of the CWD marker. This preserves
    # real-time streaming for all output while ensuring no partial or full
    # marker ever leaks to UI consumers. stderr is always yielded immediately.
    stdout_buffer = ""
    result: ExecutionResult | None = None

    try:
        async for chunk in sandbox.execute_streaming(
            tracked_command,
            timeout=effective_timeout,
            cwd=cwd,
        ):
            if isinstance(chunk, StreamChunk):
                if chunk.stream_type == "stderr":
                    # stderr never contains the marker — yield immediately
                    yield chunk
                else:
                    # Append to buffer, yield whatever is safely past the marker
                    stdout_buffer += chunk.data
                    safe_len = _safe_yield_length(stdout_buffer, _CWD_MARKER)
                    if safe_len > 0:
                        yield StreamChunk(
                            data=stdout_buffer[:safe_len], stream_type="stdout"
                        )
                        stdout_buffer = stdout_buffer[safe_len:]
            elif isinstance(chunk, ExecutionResult):
                result = chunk
    except asyncio.TimeoutError:
        yield f"Error: Command timed out after {effective_timeout} seconds."
        return
    except NotImplementedError:
        yield "Error: Sandbox does not support command execution (NoOpSandbox)."
        return
    except OSError as e:
        yield f"Error: {e}"
        return

    if result is None:
        yield "Error: Sandbox did not return an execution result."
        return

    # Extract cwd from the full stdout using rsplit — this splits on the LAST
    # occurrence of the marker, which is always the one we appended. This
    # prevents corruption if user commands happen to output the marker string.
    stdout = result.stdout or ""
    if _CWD_MARKER in stdout:
        parts = stdout.rsplit(_CWD_MARKER, 1)
        # Actual command output is before the LAST marker
        stdout = parts[0].rstrip("\n")
        # The cwd is the line after the last marker
        new_cwd = parts[1].strip()
        if new_cwd:
            shell_state["cwd"] = new_cwd
            tool_context.agent.state.set("_strands_shell_state", shell_state)

    # Flush remaining buffer with marker stripped.
    # The buffer holds the tail that overlapped with the marker prefix.
    if stdout_buffer:
        cleaned = stdout_buffer.rsplit(_CWD_MARKER, 1)[0].rstrip("\n")
        if cleaned:
            yield StreamChunk(data=cleaned, stream_type="stdout")

    # Format final output (becomes the ToolResult)
    output_parts = []
    if stdout:
        output_parts.append(stdout)
    if result.stderr:
        output_parts.append(result.stderr)

    output = "\n".join(output_parts).rstrip()

    if result.exit_code != 0:
        if output:
            output += f"\n\nExit code: {result.exit_code}"
        else:
            output = f"Command failed with exit code: {result.exit_code}"

    yield output if output else "(no output)"


def _clear_shell_state(tool_context: ToolContext) -> None:
    """Clear tracked shell state from agent state.

    Args:
        tool_context: The tool context providing access to agent state.
    """
    tool_context.agent.state.delete("_strands_shell_state")
