"""Bash tools for executing shell commands.

Mirrors ``strands-ts/src/vended-tools/bash/`` (``bash.ts`` + ``make-bash.ts``,
combined here since Python has no browser target requiring the Node-dependency
split). Provides two tools:

- :data:`bash` — a persistent host session per agent: variables and the working
  directory are retained between calls. Supports ``execute`` and ``restart``
  modes. Runs on the host without sandboxing, so use only with trusted input.
- :func:`make_bash` — a factory for a stateless, sandbox-routed bash tool. Each
  call runs in a fresh shell through the agent's (or a bound) sandbox.

The persistent shell is driven with a single :class:`subprocess.Popen` process,
background reader threads, and a unique sentinel echoed after each command to
detect completion. stdout and stderr are captured separately to match the
TypeScript oracle's ``BashOutput``.
"""

from __future__ import annotations

import atexit
import os
import threading
import uuid
import weakref
from typing import TYPE_CHECKING, Any, Literal

from ...sandbox.errors import SandboxTimeoutError
from ...tools.decorator import tool
from ...types.tools import ToolContext
from .types import SANDBOX_BASH_DESCRIPTION, BashOutput, BashSessionError, BashTimeoutError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...sandbox.base import Sandbox
    from ...tools.decorator import DecoratedFunctionTool

_DEFAULT_TIMEOUT = 120

DEFAULT_BASH_DESCRIPTION = (
    "Executes bash shell commands in a persistent session. Supports execute and restart modes. "
    "Commands persist state (variables, directory) within the session."
)
"""Description for the host-session bash tool."""


class _BashSession:
    """A persistent host bash process.

    Each call to :meth:`run` executes a command in the same shell, so state
    (variables, working directory) is retained. Completion is detected by
    echoing a unique sentinel after the command; output produced before the
    sentinel is the command's stdout.
    """

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        """Initialize the session.

        Args:
            timeout: Default per-command timeout in seconds.
        """
        self._timeout = timeout
        self._sentinel = f"__BASH_DONE_{uuid.uuid4().hex}__"
        self._process: Any = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the bash process if it is not already running."""
        import subprocess

        if self._process is not None and self._process.poll() is None:
            return
        try:
            env: Mapping[str, str] = {**os.environ, "PS1": "", "PS2": ""}
            self._process = subprocess.Popen(
                ["bash", "--noprofile", "--norc"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=dict(env),
                bufsize=0,
            )
        except OSError as e:
            raise BashSessionError(f"Failed to start bash session: {e}") from e
        _active_sessions.add(self)

    def stop(self) -> None:
        """Terminate the bash process and forget it."""
        if self._process is not None:
            try:
                self._process.kill()
            except OSError:
                pass
            self._process = None
        _active_sessions.discard(self)

    def run(self, command: str, timeout: float | None = None) -> BashOutput:
        """Run a command in the session and return its output.

        Args:
            command: The bash command to execute.
            timeout: Per-command timeout in seconds; falls back to the session default.

        Returns:
            A mapping with ``output`` (stdout) and ``error`` (stderr).

        Raises:
            BashTimeoutError: If the command does not complete within the timeout.
            BashSessionError: If the session is not running or the process exits unexpectedly.
        """
        # Serialize commands: the persistent shell handles one at a time.
        with self._lock:
            self.start()
            proc = self._process
            if proc is None or proc.stdin is None or proc.stdout is None or proc.stderr is None:
                raise BashSessionError("Bash session not properly initialized")

            effective_timeout = self._timeout if timeout is None else timeout
            sentinel = self._sentinel

            def drain(stream: Any, done: threading.Event) -> list[str]:
                # Read until the sentinel line is seen (collecting everything before it),
                # then signal completion. The sentinel is echoed to both stdout and stderr
                # so each reader knows when its stream is complete for this command.
                chunks: list[str] = []
                for raw in iter(stream.readline, b""):
                    line = raw.decode("utf-8", errors="replace")
                    if sentinel in line:
                        break
                    chunks.append(line)
                done.set()
                return chunks

            stdout_done = threading.Event()
            stderr_done = threading.Event()
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []

            def read_stdout() -> None:
                stdout_chunks.extend(drain(proc.stdout, stdout_done))

            def read_stderr() -> None:
                stderr_chunks.extend(drain(proc.stderr, stderr_done))

            threading.Thread(target=read_stdout, daemon=True).start()
            threading.Thread(target=read_stderr, daemon=True).start()

            try:
                # Echo the sentinel to both streams so both readers terminate deterministically.
                proc.stdin.write(f"{command}\necho {sentinel}\necho {sentinel} >&2\n".encode())
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                self.stop()
                raise BashSessionError(f"Failed to write command: {e}") from e

            deadline = effective_timeout
            if not stdout_done.wait(timeout=deadline) or not stderr_done.wait(timeout=deadline):
                self.stop()
                raise BashTimeoutError(f"Command timed out after {effective_timeout} seconds")

            # The process died before emitting the sentinel.
            if proc.poll() is not None:
                self.stop()
                raise BashSessionError(f"Bash process exited unexpectedly with code {proc.returncode}")

            return {"output": "".join(stdout_chunks).strip(), "error": "".join(stderr_chunks).strip()}


# Per-agent sessions, cleaned up automatically when the agent is garbage collected.
_sessions: weakref.WeakKeyDictionary[Any, _BashSession] = weakref.WeakKeyDictionary()

# Live sessions tracked for cleanup at interpreter exit.
_active_sessions: set[_BashSession] = set()


@atexit.register
def _cleanup_all_sessions() -> None:
    """Terminate any live bash sessions at interpreter exit."""
    for session in list(_active_sessions):
        session.stop()
    _active_sessions.clear()


@tool(context=True)
async def bash(  # noqa: D417
    mode: Literal["execute", "restart"],
    tool_context: ToolContext,
    command: str | None = None,
    timeout: int = _DEFAULT_TIMEOUT,
) -> BashOutput | str:
    """Executes bash commands in a persistent session that retains state across calls.

    State such as variables and the working directory persists within the
    session. Runs on the host without sandboxing; use only with trusted input.

    Args:
        mode: Operation mode: "execute" to run a command, "restart" to restart the session.
        command: The bash command to execute (required when mode is "execute").
        timeout: Timeout in seconds (default: 120, applies only to execute mode).
    """
    import asyncio

    agent = tool_context.agent

    if mode == "restart":
        existing = _sessions.get(agent)
        if existing is not None:
            existing.stop()
            del _sessions[agent]
        _sessions[agent] = _BashSession(_DEFAULT_TIMEOUT)
        return "Bash session restarted"

    if mode != "execute":
        raise BashSessionError(f'Unknown mode: {mode}. Expected "execute" or "restart".')

    if not command:
        raise BashSessionError('command is required when mode is "execute"')

    session = _sessions.get(agent)
    if session is None:
        session = _BashSession(timeout)
        _sessions[agent] = session

    # The session is blocking (subprocess + threads); run it off the event loop.
    return await asyncio.to_thread(session.run, command, timeout)


def make_bash(
    sandbox: Sandbox | None = None,
    *,
    name: str = "bash",
    description: str = SANDBOX_BASH_DESCRIPTION,
) -> DecoratedFunctionTool:
    """Create a stateless, sandbox-routed bash tool.

    If a ``sandbox`` is passed, it is bound at creation time. Otherwise the tool
    reads the sandbox from ``tool_context.agent.sandbox`` at call time. Used by
    sandbox implementations in :meth:`~strands.sandbox.base.Sandbox.get_tools`
    and by users who want a customized bash tool. Unlike :data:`bash`, each call
    runs in a fresh shell; state does not persist across calls.

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
        except SandboxTimeoutError as e:
            raise BashTimeoutError(str(e)) from e
        except Exception as e:
            raise BashSessionError(str(e)) from e
        return {"output": result.stdout, "error": result.stderr}

    return bash_tool
