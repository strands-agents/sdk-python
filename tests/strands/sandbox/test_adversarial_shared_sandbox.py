"""Adversarial tests: Shared sandboxes and concurrent tool calls.

Tests what happens when:
- Multiple agents share the same sandbox instance
- Multiple concurrent execute() calls hit the same sandbox
- File operations overlap (write/read races)
- Lifecycle races (start/stop during execution)
"""

import asyncio

import pytest

from strands.sandbox.base import ExecutionResult
from strands.sandbox.host import HostSandbox


class TestSharedSandboxConcurrentExecution:
    """What happens when multiple coroutines call execute() on the same sandbox concurrently?"""

    @pytest.mark.asyncio
    async def test_concurrent_executes_same_sandbox(self, tmp_path):
        """Multiple concurrent execute() calls on same sandbox should not corrupt each other."""
        sandbox = HostSandbox(working_dir=str(tmp_path))

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
        sandbox = HostSandbox(working_dir=str(tmp_path))

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
        sandbox = HostSandbox(working_dir=str(tmp_path))

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
        sandbox = HostSandbox(working_dir=str(tmp_path))
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


class TestSharedSandboxBetweenAgents:
    """What happens when two Agent instances share the same sandbox?"""

    def test_two_agents_same_sandbox_instance(self):
        """Two agents sharing the same sandbox should reference the same object."""
        from strands import Agent

        sandbox = HostSandbox(working_dir="/tmp/shared")
        agent1 = Agent(sandbox=sandbox)
        agent2 = Agent(sandbox=sandbox)
        assert agent1.sandbox is agent2.sandbox

    @pytest.mark.asyncio
    async def test_shared_sandbox_working_dir_isolation(self, tmp_path):
        """Commands from both agents should execute in the same working directory."""
        sandbox = HostSandbox(working_dir=str(tmp_path))

        # Agent 1 creates a file
        await sandbox.write_file("from_agent1.txt", b"hello from 1")
        # Agent 2 should see it
        content = await sandbox.read_file("from_agent1.txt")
        assert content == b"hello from 1"
