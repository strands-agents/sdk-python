"""Integration tests for :class:`~strands.sandbox.BubblewrapSandbox`.

These run real ``bwrap`` commands inside an unprivileged Linux user-namespace
jail, so they are skipped unless ``bwrap`` is installed *and* unprivileged user
namespaces actually work on this host (a smoke ``bwrap true`` must succeed --
some hardened kernels ship ``bwrap`` but disable unprivileged userns).

The default :class:`~strands.sandbox.BubblewrapConfig` mounts the read-only
system directories needed for a shell, a fresh ``/tmp`` tmpfs, and ``/proc`` +
``/dev`` -- enough for ``sh``, ``base64``, ``printenv``, and the inherited
file/code operations. A writable workspace is bind-mounted per test.
"""

import subprocess
import sys

import pytest

from strands.sandbox import BindMount, BubblewrapConfig, BubblewrapSandbox, is_bubblewrap_available


def _userns_works() -> bool:
    """Return whether ``bwrap`` can actually create an unprivileged namespace.

    Presence of the binary is necessary but not sufficient: hardened kernels may
    disable unprivileged user namespaces. A trivial ``bwrap ... true`` proves the
    jail can be created before the suite relies on it.
    """
    if not is_bubblewrap_available():
        return False
    try:
        proc = subprocess.run(
            ["bwrap", "--unshare-all", "--ro-bind", "/usr", "/usr", "--", "true"],
            capture_output=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="bubblewrap is Linux-only"),
    pytest.mark.skipif(not _userns_works(), reason="bwrap unavailable or unprivileged userns disabled"),
    pytest.mark.asyncio,
]


async def test_runs_commands_capturing_stdout_stderr_and_exit_code():
    sandbox = BubblewrapSandbox()

    result = await sandbox.execute("echo hello && echo err >&2")
    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    assert result.stderr == "err\n"

    failed = await sandbox.execute("exit 42")
    assert failed.exit_code == 42


async def test_network_is_denied_by_default():
    # With the network namespace unshared and no interfaces, even loopback DNS
    # fails. We assert the command cannot reach out, not a specific errno.
    sandbox = BubblewrapSandbox()
    result = await sandbox.execute("getent hosts example.com >/dev/null 2>&1 && echo ONLINE || echo OFFLINE")
    assert result.stdout.strip() == "OFFLINE"


async def test_environment_is_cleared_by_default(monkeypatch):
    # A host env var must NOT leak into the jail when clear_env is on (default).
    monkeypatch.setenv("HOST_SECRET", "leaked")
    sandbox = BubblewrapSandbox()
    result = await sandbox.execute("printenv HOST_SECRET || echo ABSENT")
    assert result.stdout.strip() == "ABSENT"


async def test_passes_env_vars_to_the_command():
    result = await BubblewrapSandbox().execute("printenv MY_VAR", env={"MY_VAR": "hello-from-env"})
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello-from-env"


async def test_passes_env_values_with_metacharacters_literally():
    # --setenv values are argv, not shell input, so they reach printenv verbatim.
    value = "$(whoami) $HOME `id`"
    result = await BubblewrapSandbox().execute("printenv MY_VAR", env={"MY_VAR": value})
    assert result.exit_code == 0
    assert result.stdout.strip() == value


async def test_env_passthrough_forwards_host_var(monkeypatch):
    monkeypatch.setenv("FORWARD_ME", "from-host")
    sandbox = BubblewrapSandbox(BubblewrapConfig(env_passthrough=["FORWARD_ME"]))
    result = await sandbox.execute("printenv FORWARD_ME")
    assert result.stdout.strip() == "from-host"


async def test_runs_code_via_execute_code():
    result = await BubblewrapSandbox().execute_code("echo $((6 * 7))", "sh")
    assert result.exit_code == 0
    assert result.stdout == "42\n"


async def test_round_trips_files_in_a_writable_bind(tmp_path):
    # File ops are inherited shell commands; they need a writable path in the jail.
    work = tmp_path / "work"
    work.mkdir()
    sandbox = BubblewrapSandbox(BubblewrapConfig(rw_binds=[BindMount(str(work), "/work")], working_dir="/work"))

    await sandbox.write_text("greeting.txt", "hello bwrap")
    assert await sandbox.read_text("greeting.txt") == "hello bwrap"

    data = bytes([0, 1, 2, 127, 128, 254, 255])
    await sandbox.write_file("binary.bin", data)
    assert await sandbox.read_file("binary.bin") == data


async def test_lists_and_removes_files_in_a_writable_bind(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    sandbox = BubblewrapSandbox(BubblewrapConfig(rw_binds=[BindMount(str(work), "/work")], working_dir="/work"))

    await sandbox.write_text("a.txt", "a")
    await sandbox.write_text("b.txt", "b")

    names = [f.name for f in await sandbox.list_files(".")]
    assert "a.txt" in names
    assert "b.txt" in names

    await sandbox.remove_file("a.txt")
    with pytest.raises(FileNotFoundError):
        await sandbox.read_file("a.txt")


async def test_read_only_system_dirs_cannot_be_written():
    # /usr is mounted --ro-bind; writing to it must fail inside the jail.
    result = await BubblewrapSandbox().execute("touch /usr/should-not-write 2>&1")
    assert result.exit_code != 0


async def test_respects_per_command_cwd_override(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    sandbox = BubblewrapSandbox(BubblewrapConfig(rw_binds=[BindMount(str(work), "/work")]))
    result = await sandbox.execute("pwd", cwd="/work")
    assert result.stdout.strip() == "/work"


async def test_kills_command_on_timeout():
    with pytest.raises(TimeoutError, match="timed out"):
        await BubblewrapSandbox().execute("sleep 60", timeout=0.5)


async def test_missing_binary_returns_127():
    # A bogus bwrap path surfaces as exit 127 with the install hint, not a crash.
    sandbox = BubblewrapSandbox(BubblewrapConfig(bwrap_path="bwrap-does-not-exist-xyz"))
    result = await sandbox.execute("echo hi")
    assert result.exit_code == 127
    assert "bubblewrap" in result.stderr.lower()
