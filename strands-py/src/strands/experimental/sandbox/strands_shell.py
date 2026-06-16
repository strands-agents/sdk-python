"""Sandbox backed by `Strands Shell <https://github.com/strands-agents/shell>`_.

:class:`StrandsShellSandbox` runs commands and file operations inside Strands
Shell — a Bourne-compatible shell that executes entirely in userspace, with no
``fork``/``exec``/syscalls. The agent only reaches what you declare: bound host
paths, allowlisted URLs, and per-URL credentials it never sees.

This is an **experimental** feature and may change without notice. It requires
the optional ``strands-shell`` dependency::

    pip install strands-agents[shell]

Example::

    from strands import Agent
    from strands.experimental.sandbox import StrandsShellSandbox

    sandbox = StrandsShellSandbox(
        binds=[{"source": "/my/project", "destination": "/workspace", "mode": "copy"}],
        timeout=30.0,
    )
    # The sandbox vends ``sandbox_bash`` and ``sandbox_file_editor`` tools, which
    # the agent registers automatically.
    agent = Agent(sandbox=sandbox)
    agent("List the Python files in /workspace and summarize them")

The native shell is *thread-pinned*: it must be created, used, and dropped on a
single OS thread. This sandbox enforces that by confining the shell to a
dedicated worker thread (:class:`._worker._ShellWorker`) and routing every
operation through it, so it is safe to use from Strands' threaded tool execution
and from asyncio.
"""

import asyncio
import logging
import shlex
import uuid
import weakref
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import TYPE_CHECKING, Any

from ...sandbox.base import Sandbox
from ...sandbox.constants import LANGUAGE_PATTERN
from ...sandbox.errors import SandboxPathNotFoundError
from ...sandbox.posix_shell import build_shell_env_prefix
from ...sandbox.types import ExecutionResult, FileInfo, StreamChunk
from ...types.tools import AgentTool
from ...vended_tools.bash import make_bash
from ...vended_tools.bash.types import SANDBOX_BASH_DESCRIPTION
from ...vended_tools.file_editor import make_file_editor
from ...vended_tools.file_editor.file_editor import DEFAULT_FILE_EDITOR_DESCRIPTION
from ._worker import _ShellWorker

if TYPE_CHECKING:
    import strands_shell

logger = logging.getLogger(__name__)


class StrandsShellSandbox(Sandbox):
    """A :class:`~strands.sandbox.base.Sandbox` backed by Strands Shell.

    Constructed with the same configuration surface as ``strands_shell.Shell``
    (binds, credentials, allowlisted URLs, env, umask, timeout, resource limits)
    plus the ability to load a TOML ``config_file``. The agent cannot change this
    configuration — it is fixed by whoever creates the sandbox.

    File operations use the shell's native VFS API (reporting real ``size``
    metadata); command execution runs through the in-process shell. Code
    execution writes the source to a temporary VFS file and runs the requested
    interpreter against it (Strands Shell ships ``lua``; other interpreters are
    only available if present in the sandbox).

    :meth:`get_tools` vends ``sandbox_file_editor`` and ``sandbox_bash`` tools
    bound to this sandbox, with descriptions that surface its mounts, timeout, and
    allowlists to the model. An agent constructed with this sandbox registers them
    automatically (``Agent(sandbox=sandbox)``).

    .. note::
        Unlike the base :class:`~strands.sandbox.base.Sandbox` contract, the
        per-call ``timeout`` argument to :meth:`execute`/:meth:`execute_streaming`
        (and the code variants) is **ignored**. Strands Shell enforces a single
        wall-clock timeout configured at construction (the ``timeout`` keyword);
        set it there to bound command duration.
    """

    def __init__(
        self,
        *,
        binds: Sequence[dict[str, Any]] | None = None,
        credentials: Sequence[dict[str, str]] | None = None,
        allowed_urls: Sequence[str] | None = None,
        env: dict[str, str] | None = None,
        umask: int | None = None,
        timeout: float | None = None,
        limits: "strands_shell.Limits | None" = None,
        config_file: str | None = None,
    ) -> None:
        """Initialize the sandbox and build the underlying shell.

        Args:
            binds: Bind mounts exposing host paths in the sandbox. Each is a dict
                with ``source`` and ``destination`` (both required), an optional
                ``mode`` (``"direct"`` passthrough, the default, or ``"copy"`` for
                a build-time snapshot), and an optional ``readonly`` flag.
            credentials: Per-URL credential injection rules. Each is a dict with a
                ``url`` and exactly one of ``token`` or ``env_var``. The agent
                never sees the secret; the kernel injects it per request.
            allowed_urls: URL prefixes ``curl`` may reach, bypassing the default
                SSRF guard for those hosts. Listing ``"https://"`` disables SSRF
                protection entirely — list specific endpoints instead.
            env: Environment variables to set in the shell.
            umask: File-creation mask (e.g. ``0o022``).
            timeout: Per-command wall-clock timeout in seconds. Must be positive
                and finite. ``None`` means no timeout.
            limits: A ``strands_shell.Limits`` bundle of resource caps (output
                size, file size, fds, inodes, ...). ``None`` uses the shell's
                defaults.
            config_file: Path to a TOML config file. Merged first; explicit
                keyword arguments above take precedence over it.

        Raises:
            ImportError: If the optional ``strands-shell`` package is not installed.
            ValueError: If ``timeout`` is not a positive, finite number.
        """
        try:
            import strands_shell
        except ImportError as e:
            raise ImportError(
                "StrandsShellSandbox requires the 'strands-shell' package. "
                "Install it with: pip install strands-agents[shell]"
            ) from e

        self._binds = [dict(b) for b in (binds or [])]
        self._allowed_urls = list(allowed_urls or [])
        self._credentials = [dict(c) for c in (credentials or [])]
        self._timeout = timeout

        bind_objs = [
            strands_shell.Bind(
                source=b["source"],
                destination=b["destination"],
                mode=b.get("mode", "direct"),
                readonly=b.get("readonly", False),
            )
            for b in self._binds
        ]
        cred_objs = [
            strands_shell.Cred(url=c["url"], token=c.get("token"), env_var=c.get("env_var")) for c in self._credentials
        ]

        # The native shell is thread-pinned (unsendable): it must be created,
        # used, and dropped on one OS thread. The worker owns that thread and
        # holds the shell as a thread-local, never exposing it to other threads.
        #
        # `_build` must capture only locals, never `self`: the worker thread
        # keeps the factory alive for its whole lifetime, so a captured `self`
        # would root the sandbox and stop it from ever being garbage-collected —
        # which would mean the finalizer below never runs and the thread leaks.
        allowed_urls = self._allowed_urls

        def _build() -> "strands_shell.Shell":
            return strands_shell.Shell(
                binds=bind_objs,
                credentials=cred_objs,
                allowed_urls=allowed_urls,
                env=env,
                umask=umask,
                timeout=timeout,
                limits=limits,
                config_file=config_file,
            )

        self._worker = _ShellWorker(_build)
        # Drop the worker (and the shell on its own thread) when this sandbox is
        # garbage collected, even without an explicit close.
        self._finalizer = weakref.finalize(self, self._worker.shutdown)

    # ---- Thread-pinned dispatch ----

    async def _call(self, fn: Callable[["strands_shell.Shell"], Any]) -> Any:
        """Run ``fn(shell)`` on the pinned worker thread and await the result."""
        return await asyncio.wrap_future(self._worker.submit(fn))

    # ---- Command execution ----

    async def execute_streaming(
        self,
        command: str,
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Execute a shell command in the sandbox.

        Strands Shell returns complete output rather than streaming it, so this
        yields the stdout and stderr as two :class:`StreamChunk` objects followed
        by the final :class:`ExecutionResult`.

        ``cwd`` and ``env`` are applied for this command only, by wrapping it in a
        subshell, leaving the session's persistent state untouched. The ``timeout``
        argument is **not** honored per call — Strands Shell enforces the timeout
        configured at construction; pass ``timeout`` to the constructor instead.

        Args:
            command: The shell command to execute.
            timeout: Ignored (see above); the constructor's ``timeout`` governs.
            cwd: Working directory for this command. ``None`` uses the session's
                current directory.
            env: Environment variables for this command only.
            **kwargs: Additional keyword arguments for forward compatibility.

        Yields:
            Two :class:`StreamChunk` objects (stdout, stderr) then an
            :class:`ExecutionResult`.

        Raises:
            ValueError: If an environment variable name is invalid.
        """
        wrapped = self._wrap_command(command, cwd=cwd, env=env)
        output = await self._call(lambda shell: shell.run(wrapped))
        if output.stdout:
            yield StreamChunk(data=output.stdout, stream_type="stdout")
        if output.stderr:
            yield StreamChunk(data=output.stderr, stream_type="stderr")
        yield ExecutionResult(exit_code=output.status, stdout=output.stdout, stderr=output.stderr)

    async def execute_code_streaming(
        self,
        code: str,
        language: str,
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Execute source code by writing it to a temporary VFS file and running an interpreter.

        Strands Shell ships ``lua`` (Lua 5.4); other interpreters (``python3``,
        ``node``, ...) only run if present in the sandbox. ``language`` is
        validated against :data:`~strands.sandbox.constants.LANGUAGE_PATTERN`.

        Args:
            code: The source code to execute.
            language: The interpreter to use (e.g. ``"lua"``).
            timeout: Ignored per call; the constructor's ``timeout`` governs.
            cwd: Working directory for this command.
            env: Environment variables for this command only.
            **kwargs: Additional keyword arguments for forward compatibility.

        Yields:
            :class:`StreamChunk` objects then a final :class:`ExecutionResult`.

        Raises:
            ValueError: If ``language`` contains invalid characters or an
                environment variable name is invalid.
        """
        if not LANGUAGE_PATTERN.fullmatch(language):
            raise ValueError(f"language parameter contains invalid characters: {language}")

        # Write the source to a unique temp file in the VFS, then run the
        # interpreter against it. This avoids shell-escaping the code and works
        # without a `base64` command (which Strands Shell does not provide).
        path = f"/tmp/strands_code_{_token()}"
        try:
            await self.write_file(path, code.encode("utf-8"))
        except OSError as e:
            # A VFS write failure (e.g. inode/size cap) is reported as a failed
            # execution rather than raised, matching how shell-backed sandboxes
            # surface failures through the stream.
            yield ExecutionResult(exit_code=1, stdout="", stderr=f"failed to stage code for execution: {e}")
            return
        try:
            command = f"{language} {path}"
            wrapped = self._wrap_command(command, cwd=cwd, env=env)
            output = await self._call(lambda shell: shell.run(wrapped))
        finally:
            try:
                await self.remove_file(path)
            except OSError:
                logger.debug("path=<%s> | failed to remove temporary code file", path)

        if output.stdout:
            yield StreamChunk(data=output.stdout, stream_type="stdout")
        if output.stderr:
            yield StreamChunk(data=output.stderr, stream_type="stderr")
        yield ExecutionResult(exit_code=output.status, stdout=output.stdout, stderr=output.stderr)

    @staticmethod
    def _wrap_command(command: str, *, cwd: str | None, env: dict[str, str] | None) -> str:
        """Wrap ``command`` in a subshell applying ``cwd``/``env`` without leaking state.

        Raises:
            ValueError: If an environment variable name is invalid.
        """
        if cwd is None and not env:
            return command
        env_prefix = build_shell_env_prefix(env)
        cd_prefix = f"cd {shlex.quote(cwd)} && " if cwd is not None else ""
        return f"( {cd_prefix}{env_prefix}{command} )"

    # ---- VFS file operations (native) ----

    async def read_file(self, path: str, **kwargs: Any) -> bytes:
        """Read a file from the sandbox VFS as raw bytes.

        Args:
            path: Path to the file to read.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            The file contents as raw bytes.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        data: bytes = await self._call(lambda shell: bytes(shell.read_file(path)))
        return data

    async def write_file(self, path: str, content: bytes, **kwargs: Any) -> None:
        """Write raw bytes to a file in the sandbox VFS, creating parent directories.

        Args:
            path: Path to the file to write.
            content: The content to write.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            OSError: If the file cannot be written.
        """
        await self._call(lambda shell: shell.write_file(path, content))

    async def remove_file(self, path: str, **kwargs: Any) -> None:
        """Remove a file from the sandbox VFS.

        Args:
            path: Path to the file to remove.
            **kwargs: Additional keyword arguments for forward compatibility.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        await self._call(lambda shell: shell.remove_file(path))

    async def list_files(self, path: str, **kwargs: Any) -> list[FileInfo]:
        """List files in a sandbox VFS directory.

        Args:
            path: Path to the directory to list.
            **kwargs: Additional keyword arguments for forward compatibility.

        Returns:
            A list of :class:`~strands.sandbox.types.FileInfo` entries.

        Raises:
            SandboxPathNotFoundError: If the directory does not exist.
        """
        try:
            entries = await self._call(lambda shell: shell.list_files(path))
        except FileNotFoundError as e:
            # Map the shell's missing-path error onto the sandbox contract so the
            # file editor and other callers can distinguish absence from failure.
            raise SandboxPathNotFoundError(path) from e
        return [FileInfo(name=e.name, is_dir=e.is_dir, size=e.size) for e in entries]

    # ---- Tools ----

    def get_tools(self) -> list[AgentTool]:
        """Return ``sandbox_file_editor`` and ``sandbox_bash`` tools bound to this sandbox.

        These are registered automatically when an agent is constructed with this
        sandbox (``Agent(sandbox=sandbox)``); a tool is skipped if the user
        already registered one with the same name. Each tool's description
        surfaces the sandbox's mounts, timeout, and allowlists so the model knows
        what it can reach.

        Returns:
            The tools bound to this sandbox.
        """
        suffix = self._dynamic_suffix()
        return [
            make_file_editor(
                sandbox=self,
                name="sandbox_file_editor",
                description=DEFAULT_FILE_EDITOR_DESCRIPTION + suffix,
            ),
            make_bash(
                sandbox=self,
                name="sandbox_bash",
                description=SANDBOX_BASH_DESCRIPTION + suffix,
            ),
        ]

    def _dynamic_suffix(self) -> str:
        """Build a description suffix listing this sandbox's reachable surface, or ``""``."""
        info: list[str] = []
        bind_dests = [b["destination"] for b in self._binds if b.get("destination")]
        if bind_dests:
            info.append(f"Host paths are mounted at: {', '.join(bind_dests)}.")
            info.append("Writes outside mounted paths are in-memory only and do not reach the host.")
        if self._timeout is not None:
            info.append(f"Commands time out after {self._timeout}s.")
        if self._allowed_urls:
            info.append(f"curl may reach these URL prefixes: {', '.join(self._allowed_urls)}.")
        cred_urls = [c["url"] for c in self._credentials if c.get("url")]
        if cred_urls:
            info.append(
                f"Credentials are injected automatically for: {', '.join(cred_urls)} "
                "(do not add auth headers or tokens yourself)."
            )
        return (" " + " ".join(info)) if info else ""


def _token() -> str:
    """Generate a short unique token for temporary file names."""
    return uuid.uuid4().hex[:16]
