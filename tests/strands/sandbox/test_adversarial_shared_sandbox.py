"""Adversarial tests: Shared sandboxs and concurrent tool calls.

Tests what happens when:
- Multiple agents share the same sandbox instance
- Multiple concurrent execute() calls hit the same sandbox
- File operations overlap (write/read races)
- Lifecycle races (start/stop during execution)
"""

import asyncio

import pytest

from strands.sandbox.base import ExecutionResult
from strands.sandbox.local import LocalSandbox


class TestSharedSandboxConcurrentExecution:
    """What happens when multiple coroutines call execute() on the same sandbox concurrently?"""

    @pytest.mark.asyncio
    async def test_concurrent_executes_same_sandbox(self, tmp_path):
        """Multiple concurrent execute() calls on same sandbox should not corrupt each other."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))

        async def run_command(cmd: str) -> ExecutionResult:
            return await sandbox.execute(cmd)

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
        sandbox = LocalSandbox(working_dir=str(tmp_path))

        async def write_content(content: bytes):
            await sandbox.write_file("shared.txt", content)

        # Run concurrent writes
        await asyncio.gather(
            write_content(b"content_A"),
            write_content(b"content_B"),
        )

        # File should exist and contain one of the values (no corruption)
        content = await sandbox.read_file("shared.txt")
        assert content in (b"content_A", b"content_B"), f"Corrupted content: {content!r}"

    @pytest.mark.asyncio
    async def test_concurrent_file_write_different_files(self, tmp_path):
        """Concurrent writes to different files should all succeed."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))

        async def write_file(name: str, content: bytes):
            await sandbox.write_file(name, content)

        await asyncio.gather(*[write_file(f"file_{i}.txt", f"content_{i}".encode()) for i in range(20)])

        # All files should be written correctly
        for i in range(20):
            content = await sandbox.read_file(f"file_{i}.txt")
            assert content == f"content_{i}".encode(), f"file_{i}.txt has wrong content: {content!r}"

    @pytest.mark.asyncio
    async def test_concurrent_read_write_same_file(self, tmp_path):
        """Concurrent read + write on same file — should not crash."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.write_file("test.txt", b"initial")

        async def writer():
            for i in range(10):
                await sandbox.write_file("test.txt", f"version_{i}".encode())
                await asyncio.sleep(0.001)

        async def reader():
            results = []
            for _ in range(10):
                try:
                    content = await sandbox.read_file("test.txt")
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
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        start_count = 0
        original_start = sandbox.start

        async def counting_start():
            nonlocal start_count
            start_count += 1
            await original_start()

        sandbox.start = counting_start  # type: ignore

        # Trigger 5 concurrent executes, all racing to auto-start
        results = await asyncio.gather(
            sandbox.execute("echo 0"),
            sandbox.execute("echo 1"),
            sandbox.execute("echo 2"),
            sandbox.execute("echo 3"),
            sandbox.execute("echo 4"),
        )

        for result in results:
            assert result.exit_code == 0

        # _ensure_started() uses asyncio.Lock with double-checked locking
        # to prevent multiple starts. Verify exactly one start() was called.
        assert start_count == 1, (
            f"Expected exactly 1 start() call but got {start_count}. "
            f"The asyncio.Lock in _ensure_started() should prevent concurrent starts."
        )


class TestSharedSandboxBetweenAgents:
    """What happens when two Agent instances share the same sandbox?"""

    def test_two_agents_same_sandbox_instance(self):
        """Two agents sharing the same sandbox should reference the same object."""
        from strands import Agent

        sandbox = LocalSandbox(working_dir="/tmp/shared")
        agent1 = Agent(sandbox=sandbox)
        agent2 = Agent(sandbox=sandbox)
        assert agent1.sandbox is agent2.sandbox

    @pytest.mark.asyncio
    async def test_shared_sandbox_working_dir_isolation(self, tmp_path):
        """Commands from both agents should execute in the same working directory."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))

        # Agent 1 creates a file
        await sandbox.write_file("from_agent1.txt", b"hello from 1")
        # Agent 2 should see it
        content = await sandbox.read_file("from_agent1.txt")
        assert content == b"hello from 1"

    @pytest.mark.asyncio
    async def test_shared_sandbox_stop_kills_both(self, tmp_path):
        """Stopping shared sandbox affects all agents using it."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.start()
        assert sandbox._started

        await sandbox.stop()
        assert not sandbox._started

        # Next call should auto-start again
        result = await sandbox.execute("echo recovered")
        assert result.exit_code == 0
        assert sandbox._started


class TestSandboxLifecycleEdgeCases:
    """Edge cases in start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_double_start(self, tmp_path):
        """Calling start() twice should be safe."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.start()
        await sandbox.start()  # Should not raise
        assert sandbox._started

    @pytest.mark.asyncio
    async def test_double_stop(self, tmp_path):
        """Calling stop() twice should be safe."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.start()
        await sandbox.stop()
        await sandbox.stop()  # Should not raise
        assert not sandbox._started

    @pytest.mark.asyncio
    async def test_stop_then_execute(self, tmp_path):
        """Executing after stop should auto-restart."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))
        await sandbox.start()
        await sandbox.stop()

        # Should auto-start
        result = await sandbox.execute("echo after_stop")
        assert result.exit_code == 0
        assert result.stdout.strip() == "after_stop"

    @pytest.mark.asyncio
    async def test_context_manager_reentry(self, tmp_path):
        """Using context manager twice should work."""
        sandbox = LocalSandbox(working_dir=str(tmp_path))

        async with sandbox:
            result = await sandbox.execute("echo first")
            assert result.stdout.strip() == "first"

        assert not sandbox._started

        async with sandbox:
            result = await sandbox.execute("echo second")
            assert result.stdout.strip() == "second"

    @pytest.mark.asyncio
    async def test_stop_during_execution(self, tmp_path):
        """stop() during execution should not crash or corrupt sandbox state.

        After the concurrent stop + execute settle, the sandbox should be
        in a consistent state: either stopped (and auto-restartable) or
        still running. No corruption.
        """
        sandbox = LocalSandbox(working_dir=str(tmp_path))

        async def long_running():
            return await sandbox.execute("sleep 5", timeout=10)

        async def stopper():
            await asyncio.sleep(0.1)
            await sandbox.stop()

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

        # After the dust settles, sandbox must be in a usable state:
        # auto-start should recover it for the next command
        recovery_result = await sandbox.execute("echo recovered")
        assert recovery_result.exit_code == 0
        assert recovery_result.stdout.strip() == "recovered"
        assert sandbox._started
