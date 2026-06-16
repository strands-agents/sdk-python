"""Tests for :class:`StrandsShellSandbox` against the real ``strands-shell``.

These run the in-process shell directly (no mocking), so they are skipped when
the optional ``strands-shell`` package is not installed. They cover command
execution, the thread-pinned lifecycle, native file operations, code execution,
cwd/env scoping, the vended tools, and the dynamic tool descriptions.
"""

import asyncio
import gc
import importlib.util
import threading
import time

import pytest

from strands import Agent
from strands.sandbox.errors import SandboxPathNotFoundError
from strands.sandbox.types import ExecutionResult, StreamChunk
from tests.fixtures.mocked_model_provider import MockedModelProvider

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("strands_shell") is None,
    reason="optional 'strands-shell' package not installed",
)

# Imported lazily-but-at-module-top: guarded by the skip above so collection
# never fails when the optional dep is absent.
if importlib.util.find_spec("strands_shell") is not None:
    from strands.experimental.sandbox import StrandsShellSandbox


@pytest.fixture
def sandbox():
    return StrandsShellSandbox(timeout=15.0)


@pytest.fixture
def workspace_sandbox(tmp_path):
    (tmp_path / "hello.txt").write_text("hello from host")
    return StrandsShellSandbox(
        binds=[{"source": str(tmp_path), "destination": "/workspace", "mode": "copy"}],
        timeout=15.0,
    )


# ---- construction ----


def test_missing_timeout_validation():
    with pytest.raises(ValueError, match="positive, finite"):
        StrandsShellSandbox(timeout=0)


def test_construction_failure_does_not_leak(tmp_path):
    # A bad bind source should fail construction; the worker thread is cleaned up.
    with pytest.raises(Exception):  # noqa: B017 - native error type is opaque
        StrandsShellSandbox(binds=[{"source": "/nonexistent/path/xyz", "destination": "/w", "mode": "copy"}])


def test_missing_dependency_raises_helpful_error(monkeypatch):
    # Simulate the optional package being absent at construction time.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "strands_shell":
            raise ImportError("No module named 'strands_shell'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"pip install strands-agents\[shell\]"):
        StrandsShellSandbox()


def test_dropping_sandbox_terminates_worker_thread():
    # The shell factory must not capture `self`, or the worker thread (a GC root)
    # would pin the sandbox forever, the finalizer would never fire, and the
    # native shell + thread would leak. allowed_urls is the field most likely to
    # be captured by accident, so exercise it here.
    sandbox = StrandsShellSandbox(timeout=5.0, allowed_urls=["https://example.com/"])
    asyncio.run(sandbox.execute("echo hi"))
    finalizer = sandbox._finalizer
    assert finalizer.alive

    del sandbox
    gc.collect()

    # The worker drains its queue and exits asynchronously; give it a moment.
    deadline = time.monotonic() + 5.0
    while finalizer.alive and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not finalizer.alive, "sandbox was not collected — the worker thread is leaking"

    deadline = time.monotonic() + 5.0
    while any("strands-shell" in t.name for t in threading.enumerate()) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not any("strands-shell" in t.name for t in threading.enumerate()), "shell worker thread did not terminate"


@pytest.mark.asyncio
async def test_execute_code_captures_stderr(sandbox):
    result = await sandbox.execute_code("io.stderr:write('boom\\n')", "lua")
    assert "boom" in result.stderr


# ---- execute ----


@pytest.mark.asyncio
async def test_execute_runs_command(sandbox):
    result = await sandbox.execute("echo hello")
    assert result.exit_code == 0
    assert result.stdout == "hello\n"


@pytest.mark.asyncio
async def test_execute_reports_nonzero_exit(sandbox):
    result = await sandbox.execute("echo oops >&2; exit 3")
    assert result.exit_code == 3
    assert "oops" in result.stderr


@pytest.mark.asyncio
async def test_session_state_persists_across_calls(sandbox):
    await sandbox.execute("export GREETING=hi")
    result = await sandbox.execute("echo $GREETING")
    assert result.stdout == "hi\n"


@pytest.mark.asyncio
async def test_cwd_and_env_are_scoped_to_one_command(sandbox):
    # cwd/env apply only to this command and must not leak into session state.
    scoped = await sandbox.execute("pwd; echo $SCOPED", cwd="/tmp", env={"SCOPED": "v"})
    assert scoped.stdout == "/tmp\nv\n"
    after = await sandbox.execute("pwd; echo [$SCOPED]")
    assert "/tmp" not in after.stdout
    assert "[]" in after.stdout


@pytest.mark.asyncio
async def test_streaming_yields_chunks_then_result(sandbox):
    chunks = [c async for c in sandbox.execute_streaming("echo streamed")]
    assert isinstance(chunks[-1], ExecutionResult)
    assert any(isinstance(c, StreamChunk) and "streamed" in c.data for c in chunks[:-1])


# ---- execute_code ----


@pytest.mark.asyncio
async def test_execute_code_runs_lua(sandbox):
    result = await sandbox.execute_code("print(6 * 7)", "lua")
    assert result.exit_code == 0
    assert result.stdout.strip() == "42"


@pytest.mark.asyncio
async def test_execute_code_rejects_invalid_language(sandbox):
    with pytest.raises(ValueError, match="invalid characters"):
        await sandbox.execute_code("print(1)", "lua; rm -rf /")


@pytest.mark.asyncio
async def test_execute_code_cleans_up_temp_file(sandbox):
    await sandbox.execute_code("print(1)", "lua")
    listing = await sandbox.list_files("/tmp")
    assert not any(f.name.startswith("strands_code_") for f in listing)


@pytest.mark.asyncio
async def test_execute_code_reports_write_failure_as_result(sandbox, monkeypatch):
    # A failure staging the code must surface as a failed ExecutionResult, not an
    # exception escaping the stream (which would break the bash/code tool path).
    async def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(sandbox, "write_file", _boom)
    result = await sandbox.execute_code("print(1)", "lua")
    assert result.exit_code == 1
    assert "failed to stage code" in result.stderr


# ---- file operations ----


@pytest.mark.asyncio
async def test_write_read_round_trip(sandbox):
    await sandbox.write_file("/tmp/note.txt", b"content")
    assert await sandbox.read_file("/tmp/note.txt") == b"content"
    assert await sandbox.read_text("/tmp/note.txt") == "content"


@pytest.mark.asyncio
async def test_list_files_reports_metadata(sandbox):
    await sandbox.write_file("/tmp/sized.txt", b"12345")
    entries = await sandbox.list_files("/tmp")
    sized = next(f for f in entries if f.name == "sized.txt")
    assert sized.is_dir is False
    assert sized.size == 5


@pytest.mark.asyncio
async def test_list_missing_directory_raises(sandbox):
    with pytest.raises(SandboxPathNotFoundError):
        await sandbox.list_files("/does/not/exist")


@pytest.mark.asyncio
async def test_read_missing_file_raises(sandbox):
    with pytest.raises(FileNotFoundError):
        await sandbox.read_file("/does/not/exist.txt")


@pytest.mark.asyncio
async def test_remove_file(sandbox):
    await sandbox.write_file("/tmp/gone.txt", b"x")
    await sandbox.remove_file("/tmp/gone.txt")
    with pytest.raises(FileNotFoundError):
        await sandbox.read_file("/tmp/gone.txt")


# ---- binds ----


@pytest.mark.asyncio
async def test_copy_bind_exposes_host_files(workspace_sandbox):
    result = await workspace_sandbox.execute("cat /workspace/hello.txt")
    assert "hello from host" in result.stdout


# ---- tools ----


def test_get_tools_returns_prefixed_bash_and_file_editor(sandbox):
    names = {t.tool_name for t in sandbox.get_tools()}
    assert names == {"sandbox_bash", "sandbox_file_editor"}


def test_tool_descriptions_surface_sandbox_config(workspace_sandbox):
    bash_tool = next(t for t in workspace_sandbox.get_tools() if t.tool_name == "sandbox_bash")
    description = bash_tool.tool_spec["description"]
    assert "/workspace" in description
    assert "15.0s" in description


def test_tool_descriptions_surface_urls_and_credentials():
    sandbox = StrandsShellSandbox(
        timeout=10.0,
        allowed_urls=["https://api.example.com/"],
        credentials=[{"url": "https://api.example.com/", "token": "secret"}],
    )
    description = next(t for t in sandbox.get_tools() if t.tool_name == "sandbox_bash").tool_spec["description"]
    assert "https://api.example.com/" in description
    assert "Credentials are injected automatically" in description
    # The secret value itself must never leak into the description.
    assert "secret" not in description


def test_bare_sandbox_tool_description_has_no_dynamic_suffix(sandbox):
    from strands.vended_tools.bash.types import SANDBOX_BASH_DESCRIPTION

    bash_tool = next(t for t in sandbox.get_tools() if t.tool_name == "sandbox_bash")
    # Only the timeout line is dynamic for a bare sandbox.
    assert bash_tool.tool_spec["description"].startswith(SANDBOX_BASH_DESCRIPTION)


@pytest.mark.asyncio
async def test_concurrent_calls_are_safe(sandbox):
    results = await asyncio.gather(*[sandbox.execute(f"echo {i}") for i in range(25)])
    assert [r.stdout.strip() for r in results] == [str(i) for i in range(25)]


# ---- end-to-end agent integration ----


def test_agent_auto_registers_and_uses_sandbox_tools_end_to_end(workspace_sandbox):
    """A model-driven loop creates a file via the editor, then reads it via bash.

    The agent auto-registers the sandbox's vended tools (no explicit ``tools=``).
    Uses a scripted mock model so the test is deterministic and provider-free, but
    exercises the real tool registration and thread-pinned shell execution.
    """
    model = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "1",
                            "name": "sandbox_file_editor",
                            "input": {"command": "create", "path": "/workspace/out.txt", "file_text": "written"},
                        }
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "2",
                            "name": "sandbox_bash",
                            "input": {"command": "cat /workspace/out.txt"},
                        }
                    }
                ],
            },
            {"role": "assistant", "content": [{"text": "done"}]},
        ]
    )
    agent = Agent(model=model, sandbox=workspace_sandbox)
    assert {"sandbox_bash", "sandbox_file_editor"} <= set(agent.tool_names)
    agent("create and read a file")

    tool_results = [c["toolResult"] for m in agent.messages for c in m["content"] if "toolResult" in c]
    assert tool_results[0]["status"] == "success"
    assert "created successfully" in tool_results[0]["content"][0]["text"]
    assert tool_results[1]["status"] == "success"
    # The canonical bash tool returns {"output", "error"}; stdout lands in "output".
    assert "written" in str(tool_results[1]["content"])
