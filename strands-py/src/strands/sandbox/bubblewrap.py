"""Bubblewrap sandbox -- runs commands inside an unprivileged ``bwrap`` jail.

:class:`BubblewrapSandbox` hardens :class:`~strands.sandbox.shell.PosixShellSandbox`
with real OS-level isolation. It composes the shell backend: file and code
operations are inherited (they run as shell commands), and only
:meth:`~BubblewrapSandbox.execute_streaming` is implemented -- it wraps the
shell command in a configurable `bubblewrap <https://github.com/containers/bubblewrap>`_
``bwrap`` argv that places execution inside fresh Linux namespaces (user, mount,
PID, IPC, UTS, cgroup, and -- by default -- network).

Bubblewrap is the unprivileged sandboxing engine behind Flatpak: a single
``setuid``-or-userns ``bwrap`` binary, no daemon, no server, no microVM. It is
**Linux-only** and provides **namespace-level** (not VM-level) isolation.

Security model (secure-by-default):

- All namespaces are unshared (``--unshare-all``); network access is **denied**
  unless :attr:`BubblewrapConfig.network` is ``True`` (which re-shares the net
  namespace via ``--share-net``).
- The process is killed if the parent dies (``--die-with-parent``) and runs in a
  fresh session (``--new-session``) to block TIOCSTI terminal-injection escapes.
- The environment is cleared (``--clearenv``); only explicitly passed (``env``)
  or passed-through (:attr:`BubblewrapConfig.env_passthrough`) variables reach
  the jail.
- By default a curated set of read-only system directories (``/usr``, ``/bin``,
  ``/lib`` ...) is mounted via ``--ro-bind-try`` so a shell works out of the box
  **without** exposing ``/home``, ``/root``, ``/etc``, or ``/var``. Callers add
  exactly what their workload needs through :attr:`BubblewrapConfig.ro_binds`
  and :attr:`BubblewrapConfig.rw_binds`.

This backend has no TypeScript oracle counterpart; it is a Python-first addition
that mirrors the structure of :mod:`strands.sandbox.docker` and
:mod:`strands.sandbox.ssh`.
"""

import logging
import os
import shutil
import sys
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

from .constants import ENV_KEY_PATTERN
from .shell import PosixShellSandbox, validate_env_keys
from .stream_process import stream_process
from .types import ExecutionResult, StreamChunk

logger = logging.getLogger(__name__)

#: Default executable name for the bubblewrap binary.
_DEFAULT_BWRAP_BINARY = "bwrap"

#: Curated read-only system directories mounted by default via ``--ro-bind-try``.
#:
#: These give a functional shell (the interpreter, its shared libraries) while
#: deliberately omitting ``/home``, ``/root``, ``/etc``, ``/var``, and ``/srv``
#: so host configuration and secrets are not exposed. ``--ro-bind-try`` skips
#: any path that does not exist, so the same list is safe across distributions
#: that do or do not use the usr-merge layout.
_DEFAULT_SYSTEM_PATHS: tuple[str, ...] = (
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib32",
    "/lib64",
)

#: Message surfaced (with exit code 127) when the ``bwrap`` binary is not found.
_ENOENT_MESSAGE = (
    "bwrap (bubblewrap) is not installed or not on PATH. "
    "Bubblewrap is Linux-only; install it via your distribution's package manager "
    "(e.g. 'apt install bubblewrap' or 'dnf install bubblewrap')."
)


def _is_valid_env_name(name: str) -> bool:
    """Return whether ``name`` is a valid POSIX environment variable name.

    Mirrors the check in :func:`strands.sandbox.shell.validate_env_keys` for a
    single name, returning a bool instead of raising so the caller can build a
    message that names the offending source (``env_passthrough``).

    Args:
        name: The environment variable name to validate.

    Returns:
        ``True`` if ``name`` matches :data:`~strands.sandbox.constants.ENV_KEY_PATTERN`.
    """
    return ENV_KEY_PATTERN.fullmatch(name) is not None


@dataclass(frozen=True)
class BindMount:
    """A single bind mount exposing a host path inside the sandbox.

    Attributes:
        source: Path on the host to expose.
        dest: Mount point inside the sandbox. Defaults to ``source`` (mounted at
            the same path) when ``None``.
    """

    source: str
    dest: str | None = None

    @property
    def target(self) -> str:
        """The in-sandbox mount point (``dest`` if set, otherwise ``source``)."""
        return self.dest if self.dest is not None else self.source


@dataclass
class BubblewrapConfig:
    """Configuration for :class:`BubblewrapSandbox` with secure defaults.

    Every default is chosen to minimize the sandbox's authority: no network, a
    cleared environment, all namespaces unshared, and only read-only system
    directories visible. Relax these deliberately for the workload at hand.

    Attributes:
        bwrap_path: Path to (or name of) the ``bwrap`` binary. Defaults to
            ``"bwrap"`` (resolved on ``PATH``).
        ro_binds: Additional read-only bind mounts (``--ro-bind``). Unlike the
            default system directories these use the strict ``--ro-bind`` (the
            command fails if the source is missing), making the caller's intent
            explicit.
        rw_binds: Read-write bind mounts (``--bind``). Use sparingly -- these are
            the only host paths the jail can modify.
        tmpfs: Paths to mount as fresh, writable, in-memory ``tmpfs`` filesystems.
            Defaults to ``["/tmp"]``.
        network: Whether to allow network access. ``False`` (default) leaves the
            network namespace unshared (no connectivity); ``True`` re-shares the
            host network via ``--share-net``.
        bind_system_dirs: Whether to mount the curated read-only system
            directories (:data:`_DEFAULT_SYSTEM_PATHS`) so a shell works out of
            the box. Set ``False`` for a near-empty root you populate entirely
            via ``ro_binds``/``rw_binds``.
        proc: Mount point for a new ``proc`` filesystem (``--proc``), or ``None``
            to omit it. Defaults to ``"/proc"`` (many tools require it).
        dev: Mount point for a minimal ``/dev`` (``--dev``), or ``None`` to omit
            it. Defaults to ``"/dev"``.
        clear_env: Whether to clear the environment (``--clearenv``) before
            applying ``env_passthrough`` and per-command ``env``. Defaults to
            ``True``. Setting this ``False`` forwards the **entire** host
            environment into the jail (including secrets), which undermines the
            secure-by-default posture -- prefer ``env_passthrough`` to forward
            only named variables.
        env_passthrough: Names of host environment variables to forward into the
            jail (read from :data:`os.environ` at execution time). Missing names
            are skipped. Per-command ``env`` values override these.
        die_with_parent: Whether to kill the sandboxed process if the supervising
            process dies (``--die-with-parent``). Defaults to ``True``.
        new_session: Whether to run in a new session (``--new-session``) to block
            TIOCSTI terminal-injection escapes. Defaults to ``True``.
        working_dir: Default working directory inside the sandbox (``--chdir``).
            The per-command ``cwd`` overrides it. The directory must exist inside
            the jail (i.e. be covered by a bind mount or tmpfs).
        extra_args: Raw extra arguments appended verbatim to the ``bwrap`` argv
            (before the ``--`` terminator) for options not modeled above.
    """

    bwrap_path: str = _DEFAULT_BWRAP_BINARY
    ro_binds: list[BindMount] = field(default_factory=list)
    rw_binds: list[BindMount] = field(default_factory=list)
    tmpfs: list[str] = field(default_factory=lambda: ["/tmp"])
    network: bool = False
    bind_system_dirs: bool = True
    proc: str | None = "/proc"
    dev: str | None = "/dev"
    clear_env: bool = True
    env_passthrough: list[str] = field(default_factory=list)
    die_with_parent: bool = True
    new_session: bool = True
    working_dir: str | None = None
    extra_args: list[str] = field(default_factory=list)


def is_bubblewrap_available(bwrap_path: str = _DEFAULT_BWRAP_BINARY) -> bool:
    """Return whether a usable ``bwrap`` binary is present on this host.

    Bubblewrap is Linux-only, so this returns ``False`` on any other platform
    without probing. On Linux it returns ``True`` if ``bwrap_path`` resolves on
    ``PATH`` (via :func:`shutil.which`), or -- when given an **absolute** path --
    if that path is an executable file. It does **not** verify that unprivileged
    user namespaces are enabled -- that surfaces at execution time as a clear
    ``bwrap`` error on stderr (e.g. ``"No permissions to creating new namespace"``).

    Use this to skip integration tests or to preflight a sandbox before relying
    on it.

    Args:
        bwrap_path: Path to (or name of) the ``bwrap`` binary.

    Returns:
        ``True`` if running on Linux and the binary is resolvable, else ``False``.
    """
    if sys.platform != "linux":
        return False
    if shutil.which(bwrap_path) is not None:
        return True
    # Only honor os.access for an explicit path. A bare name like "bwrap" would
    # otherwise resolve against the CWD, a false positive if a file named "bwrap"
    # happens to sit there; PATH resolution is shutil.which's job above.
    return os.path.isabs(bwrap_path) and os.access(bwrap_path, os.X_OK)


class BubblewrapSandbox(PosixShellSandbox):
    """Execute commands inside an unprivileged bubblewrap (``bwrap``) jail.

    A :class:`~strands.sandbox.shell.PosixShellSandbox` backend: file and code
    operations are inherited (they run as shell commands), and only
    :meth:`execute_streaming` is implemented -- it prefixes the shell command
    with a ``bwrap`` argv built from :class:`BubblewrapConfig`.

    Stateless: each :meth:`execute_streaming` call spawns a fresh ``bwrap``
    process. Bubblewrap is **Linux-only** and provides **namespace-level** (not
    VM) isolation with **no daemon**.

    Example:
        Run an untrusted command with no network and a writable workspace::

            from strands.sandbox import BindMount, BubblewrapConfig, BubblewrapSandbox

            sandbox = BubblewrapSandbox(
                BubblewrapConfig(
                    rw_binds=[BindMount("/tmp/workspace", "/work")],
                    working_dir="/work",
                )
            )
            result = await sandbox.execute("echo hello")
    """

    def __init__(self, config: BubblewrapConfig | None = None) -> None:
        """Initialize the bubblewrap sandbox.

        Args:
            config: Sandbox configuration. Defaults to a fresh
                :class:`BubblewrapConfig` (secure defaults: no network, cleared
                environment, read-only system directories, all namespaces
                unshared).
        """
        self.config = config if config is not None else BubblewrapConfig()
        # Mirror the other backends' public attribute for the effective default cwd.
        self.working_dir = self.config.working_dir

    def _build_bwrap_args(
        self,
        command: str,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
    ) -> list[str]:
        """Build the ``bwrap`` argument vector for a single command.

        Pure apart from reading :data:`os.environ` for
        :attr:`BubblewrapConfig.env_passthrough`, so the produced argv is
        deterministic for a given config and inputs.

        Args:
            command: The shell command to run inside the jail (via ``sh -c``).
            cwd: Per-command working directory, overriding
                :attr:`BubblewrapConfig.working_dir`.
            env: Per-command environment variables, applied via ``--setenv`` and
                overriding any matching :attr:`BubblewrapConfig.env_passthrough`
                value.

        Returns:
            The full argument vector for ``bwrap`` (excluding the program name).

        Raises:
            ValueError: If an environment variable name is invalid.
        """
        config = self.config
        args: list[str] = ["--unshare-all"]

        # --unshare-all unshares the network namespace; re-share it only on opt-in.
        if config.network:
            args.append("--share-net")

        if config.die_with_parent:
            args.append("--die-with-parent")
        if config.new_session:
            args.append("--new-session")

        # Clear the environment before any --setenv so passed values survive.
        if config.clear_env:
            args.append("--clearenv")

        # Read-only system directories first. --ro-bind-try skips missing sources,
        # keeping the default usable across distros without runtime stat() calls.
        if config.bind_system_dirs:
            for path in _DEFAULT_SYSTEM_PATHS:
                args += ["--ro-bind-try", path, path]

        # tmpfs mounts go before the caller's binds: bwrap applies mount ops in
        # argv order, so a tmpfs must be laid down first for a bind nested under
        # it (e.g. a rw-bind at /tmp/work) to survive rather than be shadowed.
        for path in config.tmpfs:
            args += ["--tmpfs", path]

        # Explicit caller binds use strict --ro-bind/--bind: a missing source is a
        # configuration error and should fail loudly.
        for mount in config.ro_binds:
            args += ["--ro-bind", mount.source, mount.target]
        for mount in config.rw_binds:
            args += ["--bind", mount.source, mount.target]

        if config.proc is not None:
            args += ["--proc", config.proc]
        if config.dev is not None:
            args += ["--dev", config.dev]

        # Environment: passthrough first, then per-command env (which overrides).
        # --setenv writes argv values verbatim (no shell evaluation), so values
        # containing shell metacharacters reach the process literally.
        for name in config.env_passthrough:
            if not _is_valid_env_name(name):
                raise ValueError(f"Invalid environment variable name: {name}")
            value = os.environ.get(name)
            if value is not None:
                args += ["--setenv", name, value]
        if env:
            validate_env_keys(env)
            for key, value in env.items():
                args += ["--setenv", key, value]

        effective_cwd = cwd if cwd is not None else self.working_dir
        if effective_cwd is not None:
            args += ["--chdir", effective_cwd]

        args += list(config.extra_args)

        # '--' terminates bwrap option parsing so the command and its arguments are
        # never mistaken for bwrap flags. The command runs through 'sh -c' so the
        # inherited shell-based file/code operations (heredocs, pipes) work.
        args += ["--", "sh", "-c", command]
        return args

    async def execute_streaming(
        self,
        command: str,
        *,
        timeout: float | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ExecutionResult, None]:
        """Execute a command inside the bubblewrap jail, streaming output.

        Args:
            command: The shell command to execute.
            timeout: Maximum wall-clock execution time in seconds. ``None`` means
                no timeout.
            cwd: Working directory for this command, overriding
                :attr:`BubblewrapConfig.working_dir`. Must exist inside the jail.
            env: Environment variables to set, applied via ``bwrap --setenv``
                (verbatim, not shell-evaluated).
            **kwargs: Additional keyword arguments for forward compatibility.

        Yields:
            :class:`StreamChunk` objects for output, then a final
            :class:`ExecutionResult`. If ``bwrap`` is missing, a single
            :class:`ExecutionResult` with exit code 127 and an explanatory
            ``stderr`` is yielded.

        Raises:
            ValueError: If an environment variable name is invalid.
            TimeoutError: If execution exceeds ``timeout`` seconds.
        """
        args = self._build_bwrap_args(command, cwd=cwd, env=env)
        logger.debug(
            "bwrap_path=<%s>, network=<%s>, arg_count=<%d> | running command in bubblewrap jail",
            self.config.bwrap_path,
            self.config.network,
            len(args),
        )
        # aclosing() guarantees stream_process's finally (which kills the bwrap
        # process tree) runs deterministically when the consumer cancels or breaks
        # out early -- a plain "async for" would defer that cleanup to the event
        # loop's async-generator finalizer, leaving the jailed process running.
        async with aclosing(
            stream_process(self.config.bwrap_path, args, timeout=timeout, enoent_message=_ENOENT_MESSAGE)
        ) as stream:
            async for chunk in stream:
                yield chunk
