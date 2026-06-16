"""Tests for :class:`~strands.sandbox.bubblewrap.BubblewrapSandbox`.

Mirrors the argv-assertion style of ``tests/strands/sandbox/test_docker.py``:
the process pump (``stream_process``) is mocked, so **no ``bwrap`` binary is
required** to run these tests. They assert the ``bwrap`` argv the sandbox builds
from a :class:`~strands.sandbox.bubblewrap.BubblewrapConfig`, exercising the
secure defaults, the namespace/mount/env flags, and the security-sensitive
boundaries (``--`` flag terminator, no-network default, env verbatim handling).
"""

import unittest.mock

import pytest

from strands.sandbox import (
    BindMount,
    BubblewrapConfig,
    BubblewrapSandbox,
    is_bubblewrap_available,
)
from strands.sandbox.bubblewrap import (
    _DEFAULT_SYSTEM_PATHS,
    _ENOENT_MESSAGE,
)
from strands.sandbox.types import ExecutionResult, StreamChunk


@pytest.fixture
def mock_stream_process(agenerator):
    """Patch ``stream_process`` in the bubblewrap module, returning a no-op result.

    Yields the mock so tests can inspect the ``(program, args, kwargs)`` it was
    called with via ``mock.call_args``.
    """
    with unittest.mock.patch("strands.sandbox.bubblewrap.stream_process") as mock:
        mock.return_value = agenerator([ExecutionResult(exit_code=0, stdout="", stderr="")])
        yield mock


def _program(mock_stream_process) -> str:
    """The program passed to ``stream_process`` (its first positional argument)."""
    return mock_stream_process.call_args.args[0]


def _args(mock_stream_process) -> list[str]:
    """The argv passed to ``stream_process`` (its second positional argument)."""
    return mock_stream_process.call_args.args[1]


# ---- constructor ----


def test_defaults_to_secure_config():
    sandbox = BubblewrapSandbox()
    assert sandbox.config.network is False
    assert sandbox.config.clear_env is True
    assert sandbox.config.die_with_parent is True
    assert sandbox.config.new_session is True
    assert sandbox.config.bind_system_dirs is True
    assert sandbox.working_dir is None


def test_working_dir_mirrors_config():
    sandbox = BubblewrapSandbox(BubblewrapConfig(working_dir="/work"))
    assert sandbox.working_dir == "/work"


# ---- argv construction: secure defaults ----


@pytest.mark.asyncio
async def test_default_argv_is_secure(mock_stream_process):
    await BubblewrapSandbox().execute("echo hi")
    args = _args(mock_stream_process)

    # Namespaces fully unshared, no opt-in network share.
    assert "--unshare-all" in args
    assert "--share-net" not in args
    # Hardening flags on by default.
    assert "--die-with-parent" in args
    assert "--new-session" in args
    assert "--clearenv" in args
    # proc/dev/tmpfs defaults.
    assert args[args.index("--proc") + 1] == "/proc"
    assert args[args.index("--dev") + 1] == "/dev"
    assert args[args.index("--tmpfs") + 1] == "/tmp"


@pytest.mark.asyncio
async def test_default_binds_system_dirs_readonly(mock_stream_process):
    await BubblewrapSandbox().execute("echo hi")
    args = _args(mock_stream_process)
    for path in _DEFAULT_SYSTEM_PATHS:
        # Each default system dir is mounted read-only-try at the same path.
        assert ["--ro-bind-try", path, path] == _triple_at(args, "--ro-bind-try", path)


def _triple_at(args: list[str], flag: str, source: str) -> list[str]:
    """Return the ``[flag, source, dest]`` slice starting at the matching flag+source."""
    for i in range(len(args) - 2):
        if args[i] == flag and args[i + 1] == source:
            return args[i : i + 3]
    return []


@pytest.mark.asyncio
async def test_command_runs_through_sh_after_double_dash(mock_stream_process):
    # The command must be positional after '--' so it is never parsed as a bwrap flag.
    await BubblewrapSandbox().execute("echo hi")
    args = _args(mock_stream_process)
    assert args[-4:] == ["--", "sh", "-c", "echo hi"]


@pytest.mark.asyncio
async def test_uses_configured_bwrap_path(mock_stream_process):
    await BubblewrapSandbox(BubblewrapConfig(bwrap_path="/opt/bin/bwrap")).execute("echo hi")
    assert _program(mock_stream_process) == "/opt/bin/bwrap"


@pytest.mark.asyncio
async def test_malicious_command_cannot_inject_bwrap_flags(mock_stream_process):
    # A command that looks like a bwrap flag stays a single sh -c argument.
    await BubblewrapSandbox().execute("--share-net; rm -rf /")
    args = _args(mock_stream_process)
    assert args[-4:] == ["--", "sh", "-c", "--share-net; rm -rf /"]
    # The injected text is only the sh -c payload, never a real flag before '--'.
    assert args.index("--") == len(args) - 4


# ---- network ----


@pytest.mark.asyncio
async def test_network_opt_in_shares_net(mock_stream_process):
    await BubblewrapSandbox(BubblewrapConfig(network=True)).execute("curl x")
    args = _args(mock_stream_process)
    assert "--unshare-all" in args
    assert "--share-net" in args


@pytest.mark.asyncio
async def test_network_denied_by_default(mock_stream_process):
    await BubblewrapSandbox(BubblewrapConfig(network=False)).execute("curl x")
    assert "--share-net" not in _args(mock_stream_process)


# ---- hardening toggles ----


@pytest.mark.asyncio
async def test_hardening_flags_can_be_disabled(mock_stream_process):
    await BubblewrapSandbox(BubblewrapConfig(die_with_parent=False, new_session=False, clear_env=False)).execute(
        "echo hi"
    )
    args = _args(mock_stream_process)
    assert "--die-with-parent" not in args
    assert "--new-session" not in args
    assert "--clearenv" not in args


@pytest.mark.asyncio
async def test_system_dirs_can_be_disabled(mock_stream_process):
    await BubblewrapSandbox(BubblewrapConfig(bind_system_dirs=False)).execute("echo hi")
    assert "--ro-bind-try" not in _args(mock_stream_process)


@pytest.mark.asyncio
async def test_proc_and_dev_can_be_omitted(mock_stream_process):
    await BubblewrapSandbox(BubblewrapConfig(proc=None, dev=None)).execute("echo hi")
    args = _args(mock_stream_process)
    assert "--proc" not in args
    assert "--dev" not in args


@pytest.mark.asyncio
async def test_custom_proc_and_dev_mount_points(mock_stream_process):
    await BubblewrapSandbox(BubblewrapConfig(proc="/p", dev="/d")).execute("echo hi")
    args = _args(mock_stream_process)
    assert args[args.index("--proc") + 1] == "/p"
    assert args[args.index("--dev") + 1] == "/d"


# ---- bind mounts ----


@pytest.mark.asyncio
async def test_ro_and_rw_binds(mock_stream_process):
    await BubblewrapSandbox(
        BubblewrapConfig(
            ro_binds=[BindMount("/host/ro", "/ro")],
            rw_binds=[BindMount("/host/rw", "/rw")],
        )
    ).execute("echo hi")
    args = _args(mock_stream_process)
    assert _triple_at(args, "--ro-bind", "/host/ro") == ["--ro-bind", "/host/ro", "/ro"]
    assert _triple_at(args, "--bind", "/host/rw") == ["--bind", "/host/rw", "/rw"]


@pytest.mark.asyncio
async def test_bind_mount_defaults_dest_to_source(mock_stream_process):
    await BubblewrapSandbox(BubblewrapConfig(rw_binds=[BindMount("/same/path")])).execute("echo hi")
    args = _args(mock_stream_process)
    assert _triple_at(args, "--bind", "/same/path") == ["--bind", "/same/path", "/same/path"]


def test_bind_mount_target_property():
    assert BindMount("/a").target == "/a"
    assert BindMount("/a", "/b").target == "/b"


@pytest.mark.asyncio
async def test_multiple_tmpfs_paths(mock_stream_process):
    await BubblewrapSandbox(BubblewrapConfig(tmpfs=["/tmp", "/run"])).execute("echo hi")
    args = _args(mock_stream_process)
    tmpfs_targets = [args[i + 1] for i, a in enumerate(args) if a == "--tmpfs"]
    assert tmpfs_targets == ["/tmp", "/run"]


@pytest.mark.asyncio
async def test_tmpfs_mounted_before_caller_binds(mock_stream_process):
    # bwrap applies mount ops in argv order. A rw-bind nested under a tmpfs must
    # come AFTER the tmpfs, or the tmpfs would shadow it. Lock the ordering.
    await BubblewrapSandbox(BubblewrapConfig(tmpfs=["/tmp"], rw_binds=[BindMount("/host/work", "/tmp/work")])).execute(
        "echo hi"
    )
    args = _args(mock_stream_process)
    assert args.index("--tmpfs") < args.index("--bind")


# ---- working dir ----


@pytest.mark.asyncio
async def test_working_dir_sets_chdir(mock_stream_process):
    await BubblewrapSandbox(BubblewrapConfig(working_dir="/work")).execute("pwd")
    args = _args(mock_stream_process)
    assert args[args.index("--chdir") + 1] == "/work"


@pytest.mark.asyncio
async def test_cwd_option_overrides_working_dir(mock_stream_process):
    await BubblewrapSandbox(BubblewrapConfig(working_dir="/work")).execute("pwd", cwd="/override")
    args = _args(mock_stream_process)
    assert args[args.index("--chdir") + 1] == "/override"


@pytest.mark.asyncio
async def test_no_chdir_when_unset(mock_stream_process):
    await BubblewrapSandbox().execute("pwd")
    assert "--chdir" not in _args(mock_stream_process)


# ---- environment ----


@pytest.mark.asyncio
async def test_env_passed_as_setenv(mock_stream_process):
    await BubblewrapSandbox().execute("echo $FOO", env={"FOO": "bar", "BAZ": "qux"})
    args = _args(mock_stream_process)
    assert _triple_at(args, "--setenv", "FOO") == ["--setenv", "FOO", "bar"]
    assert _triple_at(args, "--setenv", "BAZ") == ["--setenv", "BAZ", "qux"]


@pytest.mark.asyncio
async def test_env_values_are_verbatim_not_shell_evaluated(mock_stream_process):
    # --setenv values are argv, not shell input: metacharacters reach the process literally.
    await BubblewrapSandbox().execute("printenv FOO", env={"FOO": "$(whoami) `id`"})
    args = _args(mock_stream_process)
    assert _triple_at(args, "--setenv", "FOO") == ["--setenv", "FOO", "$(whoami) `id`"]


@pytest.mark.asyncio
async def test_rejects_invalid_env_var_names(mock_stream_process):
    with pytest.raises(ValueError, match="Invalid environment variable name"):
        await BubblewrapSandbox().execute("cmd", env={"BAD-KEY": "v"})


@pytest.mark.asyncio
async def test_env_passthrough_reads_from_os_environ(mock_stream_process, monkeypatch):
    monkeypatch.setenv("PASS_ME", "from-host")
    monkeypatch.delenv("ABSENT_VAR", raising=False)
    await BubblewrapSandbox(BubblewrapConfig(env_passthrough=["PASS_ME", "ABSENT_VAR"])).execute("echo hi")
    args = _args(mock_stream_process)
    # Present var is forwarded; absent var is silently skipped.
    assert _triple_at(args, "--setenv", "PASS_ME") == ["--setenv", "PASS_ME", "from-host"]
    assert "ABSENT_VAR" not in args


@pytest.mark.asyncio
async def test_per_command_env_overrides_passthrough(mock_stream_process, monkeypatch):
    monkeypatch.setenv("SHARED", "from-host")
    await BubblewrapSandbox(BubblewrapConfig(env_passthrough=["SHARED"])).execute(
        "echo hi", env={"SHARED": "from-call"}
    )
    args = _args(mock_stream_process)
    setenv_values = [args[i + 1 : i + 3] for i, a in enumerate(args) if a == "--setenv"]
    # Both appear, but the per-command value comes last so it wins at exec time.
    assert ["SHARED", "from-host"] in setenv_values
    assert ["SHARED", "from-call"] in setenv_values
    assert setenv_values[-1] == ["SHARED", "from-call"]


@pytest.mark.asyncio
async def test_rejects_invalid_env_passthrough_name(mock_stream_process):
    with pytest.raises(ValueError, match="Invalid environment variable name"):
        await BubblewrapSandbox(BubblewrapConfig(env_passthrough=["BAD-KEY"])).execute("echo hi")


# ---- extra args ----


@pytest.mark.asyncio
async def test_extra_args_appended_before_terminator(mock_stream_process):
    await BubblewrapSandbox(BubblewrapConfig(extra_args=["--hostname", "jail"])).execute("echo hi")
    args = _args(mock_stream_process)
    terminator = args.index("--")
    assert args[terminator - 2 : terminator] == ["--hostname", "jail"]


# ---- timeout / enoent passthrough to stream_process ----


@pytest.mark.asyncio
async def test_forwards_timeout_and_enoent_message(mock_stream_process):
    await BubblewrapSandbox().execute("sleep 10", timeout=5)
    assert mock_stream_process.call_args.kwargs == {
        "timeout": 5,
        "enoent_message": _ENOENT_MESSAGE,
    }


# ---- inherited shell ops build the right wrapped command ----


@pytest.mark.asyncio
async def test_execute_code_runs_through_bwrap(mock_stream_process):
    # The inherited PosixShellSandbox.execute_code path must still wrap in bwrap:
    # the base64 heredoc pipeline becomes the sh -c payload inside the jail.
    await BubblewrapSandbox().execute_code("print(1)", "python3")
    args = _args(mock_stream_process)
    assert args[-4:-1] == ["--", "sh", "-c"]
    payload = args[-1]
    assert "base64 -d" in payload
    assert "| python3" in payload


@pytest.mark.asyncio
async def test_read_file_runs_through_bwrap(mock_stream_process):
    await BubblewrapSandbox().read_file("/x.txt")
    args = _args(mock_stream_process)
    assert args[-4:-1] == ["--", "sh", "-c"]
    assert "base64 < /x.txt" in args[-1]


# ---- execute_streaming body (real call path, no bwrap binary required) ----


@pytest.mark.asyncio
async def test_execute_streaming_surfaces_missing_binary_as_127():
    # Drives the real execute_streaming/stream_process path (not mocked) with a
    # bogus binary: bwrap is never spawned, so this needs no bwrap installed and
    # still exercises the live wiring -- guarding against regressions like a stale
    # local name in the stream_process call.
    sandbox = BubblewrapSandbox(BubblewrapConfig(bwrap_path="strands-bwrap-does-not-exist-xyz"))
    result = await sandbox.execute("echo hi")
    assert result.exit_code == 127
    assert "bubblewrap" in result.stderr.lower()


@pytest.mark.asyncio
async def test_execute_streaming_yields_chunks_then_result(agenerator):
    # Patch only the process pump; assert execute_streaming faithfully relays the
    # chunk(s) and final result from stream_process.
    chunks = [
        StreamChunk(data="out", stream_type="stdout"),
        ExecutionResult(exit_code=0, stdout="out", stderr=""),
    ]
    with unittest.mock.patch("strands.sandbox.bubblewrap.stream_process") as mock:
        mock.return_value = agenerator(chunks)
        relayed = [c async for c in BubblewrapSandbox().execute_streaming("echo out")]
    assert relayed == chunks


@pytest.mark.asyncio
async def test_execute_streaming_closes_inner_generator_on_early_break():
    # On an early break the consumer must close our generator, which (via
    # aclosing) must close stream_process so its finally -- killing the bwrap
    # process tree -- runs deterministically rather than at GC time.
    closed = False

    async def fake_stream(*_args, **_kwargs):
        nonlocal closed
        try:
            yield StreamChunk(data="first", stream_type="stdout")
            yield ExecutionResult(exit_code=0, stdout="", stderr="")
        finally:
            closed = True

    with unittest.mock.patch("strands.sandbox.bubblewrap.stream_process", fake_stream):
        gen = BubblewrapSandbox().execute_streaming("echo hi")
        first = await gen.__anext__()
        assert isinstance(first, StreamChunk)
        await gen.aclose()

    assert closed is True


# ---- is_bubblewrap_available ----
def test_is_bubblewrap_available_false_off_linux(monkeypatch):
    monkeypatch.setattr("strands.sandbox.bubblewrap.sys.platform", "darwin")
    assert is_bubblewrap_available() is False


def test_is_bubblewrap_available_true_when_resolvable(monkeypatch):
    monkeypatch.setattr("strands.sandbox.bubblewrap.sys.platform", "linux")
    monkeypatch.setattr("strands.sandbox.bubblewrap.shutil.which", lambda _: "/usr/bin/bwrap")
    assert is_bubblewrap_available() is True


def test_is_bubblewrap_available_false_when_missing(monkeypatch):
    monkeypatch.setattr("strands.sandbox.bubblewrap.sys.platform", "linux")
    monkeypatch.setattr("strands.sandbox.bubblewrap.shutil.which", lambda _: None)
    monkeypatch.setattr("strands.sandbox.bubblewrap.os.access", lambda *a: False)
    assert is_bubblewrap_available() is False


def test_is_bubblewrap_available_ignores_relative_path_access(monkeypatch):
    # A bare name must NOT be probed with os.access (would resolve against CWD);
    # only shutil.which may resolve it on PATH.
    monkeypatch.setattr("strands.sandbox.bubblewrap.sys.platform", "linux")
    monkeypatch.setattr("strands.sandbox.bubblewrap.shutil.which", lambda _: None)
    monkeypatch.setattr("strands.sandbox.bubblewrap.os.access", lambda *a: True)
    assert is_bubblewrap_available("bwrap") is False


def test_is_bubblewrap_available_honors_absolute_path_access(monkeypatch):
    monkeypatch.setattr("strands.sandbox.bubblewrap.sys.platform", "linux")
    monkeypatch.setattr("strands.sandbox.bubblewrap.shutil.which", lambda _: None)
    monkeypatch.setattr("strands.sandbox.bubblewrap.os.access", lambda *a: True)
    assert is_bubblewrap_available("/opt/bin/bwrap") is True
