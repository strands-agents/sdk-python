"""Adversarial tests: Shared workspaces and concurrent tool calls.

Tests what happens when:
- Multiple agents share the same workspace instance
- Multiple concurrent execute() calls hit the same workspace
- File operations overlap (write/read races)
- Lifecycle races (start/stop during execution)
"""

import asyncio

import pytest

from strands.workspace.base import ExecutionResult
from strands.workspace.local import LocalWorkspace


class TestSharedWorkspaceConcurrentExecution:
    """What happens when multiple coroutines call execute() on the same workspace concurrently?"""

    @pytest.mark.asyncio
    async def test_concurrent_executes_same_workspace(self, tmp_path):
        """Multiple concurrent execute() calls on same workspace should not corrupt each other."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))

        async def run_command(cmd: str) -> ExecutionResult:
            return await workspace._execute_to_result(cmd)

        # Run 10 concurrent commands
        results = await asyncio.gather(
            run_command("echo cmd0"),
            run_command("echo cmd1"),
            run_command("echo cmd2"),
            run_command("echo cmd3"),
            run_command("echo cmd4"),
            run_command("echo cmd5"),
            run_command("echo cmd6"),
            run_command("echo cmd7"),
            run_command("echo cmd8"),
            run_command("echo cmd9"),
        )

        # Each command should have its own exit code and output
        for i, result in enumerate(results):
            assert result.exit_code == 0, f"cmd{i} failed: {result.stderr}"
            assert result.stdout.strip() == f"cmd{i}", f"cmd{i} got wrong output: {result.stdout!r}"

    @pytest.mark.asyncio
    async def test_concurrent_file_write_same_file(self, tmp_path):
        """Two concurrent writes to the same file — last write wins, no crash."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))

        async def write_content(content: bytes):
            await workspace.write_file("shared.txt", content)

        # Run concurrent writes
        await asyncio.gather(
            write_content(b"content_A"),
            write_content(b"content_B"),
        )

        # File should exist and contain one of the values (no corruption)
        content = await workspace.read_file("shared.txt")
        assert content in (b"content_A", b"content_B"), f"Corrupted content: {content!r}"

    @pytest.mark.asyncio
    async def test_concurrent_file_write_different_files(self, tmp_path):
        """Concurrent writes to different files should all succeed."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))

        async def write_file(name: str, content: bytes):
            await workspace.write_file(name, content)

        await asyncio.gather(*[write_file(f"file_{i}.txt", f"content_{i}".encode()) for i in range(20)])

        # All files should be written correctly
        for i in range(20):
            content = await workspace.read_file(f"file_{i}.txt")
            assert content == f"content_{i}".encode(), f"file_{i}.txt has wrong content: {content!r}"

    @pytest.mark.asyncio
    async def test_concurrent_read_write_same_file(self, tmp_path):
        """Concurrent read + write on same file — should not crash."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.write_file("test.txt", b"initial")

        async def writer():
            for i in range(10):
                await workspace.write_file("test.txt", f"version_{i}".encode())
                await asyncio.sleep(0.001)

        async def reader():
            results = []
            for _ in range(10):
                try:
                    content = await workspace.read_file("test.txt")
                    results.append(content)
                except FileNotFoundError:
                    # File might be in the middle of being overwritten
                    results.append("FILE_NOT_FOUND")
                await asyncio.sleep(0.001)
            return results

        # This should not raise any unhandled exceptions
        writer_task = asyncio.create_task(writer())
        reader_task = asyncio.create_task(reader())
        results = await reader_task
        await writer_task

        # At least some reads should succeed
        successful_reads = [r for r in results if r != "FILE_NOT_FOUND"]
        assert len(successful_reads) > 0

    @pytest.mark.asyncio
    async def test_concurrent_auto_start_race(self, tmp_path):
        """Multiple concurrent execute() calls race to auto-start — only one start() should happen."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        start_count = 0
        original_start = workspace.start

        async def counting_start():
            nonlocal start_count
            start_count += 1
            await original_start()

        workspace.start = counting_start  # type: ignore

        # Trigger 5 concurrent executes, all racing to auto-start
        results = await asyncio.gather(
            workspace._execute_to_result("echo 0"),
            workspace._execute_to_result("echo 1"),
            workspace._execute_to_result("echo 2"),
            workspace._execute_to_result("echo 3"),
            workspace._execute_to_result("echo 4"),
        )

        for result in results:
            assert result.exit_code == 0

        # _ensure_started() uses asyncio.Lock with double-checked locking
        # to prevent multiple starts. Verify exactly one start() was called.
        assert start_count == 1, (
            f"Expected exactly 1 start() call but got {start_count}. "
            f"The asyncio.Lock in _ensure_started() should prevent concurrent starts."
        )


class TestSharedWorkspaceBetweenAgents:
    """What happens when two Agent instances share the same workspace?"""

    def test_two_agents_same_workspace_instance(self):
        """Two agents sharing the same workspace should reference the same object."""
        from strands import Agent

        workspace = LocalWorkspace(working_dir="/tmp/shared")
        agent1 = Agent(workspace=workspace)
        agent2 = Agent(workspace=workspace)
        assert agent1.workspace is agent2.workspace

    @pytest.mark.asyncio
    async def test_shared_workspace_working_dir_isolation(self, tmp_path):
        """Commands from both agents should execute in the same working directory."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))

        # Agent 1 creates a file
        await workspace.write_file("from_agent1.txt", b"hello from 1")
        # Agent 2 should see it
        content = await workspace.read_file("from_agent1.txt")
        assert content == b"hello from 1"

    @pytest.mark.asyncio
    async def test_shared_workspace_stop_kills_both(self, tmp_path):
        """Stopping shared workspace affects all agents using it."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.start()
        assert workspace._started

        await workspace.stop()
        assert not workspace._started

        # Next call should auto-start again
        result = await workspace._execute_to_result("echo recovered")
        assert result.exit_code == 0
        assert workspace._started


class TestWorkspaceLifecycleEdgeCases:
    """Edge cases in start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_double_start(self, tmp_path):
        """Calling start() twice should be safe."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.start()
        await workspace.start()  # Should not raise
        assert workspace._started

    @pytest.mark.asyncio
    async def test_double_stop(self, tmp_path):
        """Calling stop() twice should be safe."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.start()
        await workspace.stop()
        await workspace.stop()  # Should not raise
        assert not workspace._started

    @pytest.mark.asyncio
    async def test_stop_then_execute(self, tmp_path):
        """Executing after stop should auto-restart."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))
        await workspace.start()
        await workspace.stop()

        # Should auto-start
        result = await workspace._execute_to_result("echo after_stop")
        assert result.exit_code == 0
        assert result.stdout.strip() == "after_stop"

    @pytest.mark.asyncio
    async def test_context_manager_reentry(self, tmp_path):
        """Using context manager twice should work."""
        workspace = LocalWorkspace(working_dir=str(tmp_path))

        async with workspace:
            result = await workspace._execute_to_result("echo first")
            assert result.stdout.strip() == "first"

        assert not workspace._started

        async with workspace:
            result = await workspace._execute_to_result("echo second")
            assert result.stdout.strip() == "second"

    @pytest.mark.asyncio
    async def test_stop_during_execution(self, tmp_path):
        """stop() during execution should not crash or corrupt workspace state.

        After the concurrent stop + execute settle, the workspace should be
        in a consistent state: either stopped (and auto-restartable) or
        still running. No corruption.
        """
        workspace = LocalWorkspace(working_dir=str(tmp_path))

        async def long_running():
            return await workspace._execute_to_result("sleep 5", timeout=10)

        async def stopper():
            await asyncio.sleep(0.1)
            await workspace.stop()

        # Both tasks run concurrently — gather with return_exceptions
        results = await asyncio.gather(
            long_running(),
            stopper(),
            return_exceptions=True,
        )

        # Verify no unexpected exception types
        for r in results:
            if isinstance(r, Exception):
                assert isinstance(r, (asyncio.TimeoutError, asyncio.CancelledError, ProcessLookupError, OSError)), (
                    f"Unexpected exception during concurrent stop: {type(r).__name__}: {r}"
                )

        # After the dust settles, workspace must be in a usable state:
        # auto-start should recover it for the next command
        recovery_result = await workspace._execute_to_result("echo recovered")
        assert recovery_result.exit_code == 0
        assert recovery_result.stdout.strip() == "recovered"
        assert workspace._started
