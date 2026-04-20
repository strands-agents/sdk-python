"""Tests for the NoOpSandbox implementation."""

import pytest

from strands.sandbox.base import Sandbox
from strands.sandbox.noop import NoOpSandbox


class TestNoOpSandbox:
    def test_is_sandbox_instance(self) -> None:
        sandbox = NoOpSandbox()
        assert isinstance(sandbox, Sandbox)

    @pytest.mark.asyncio
    async def test_execute_raises_not_implemented(self) -> None:
        sandbox = NoOpSandbox()
        with pytest.raises(NotImplementedError, match="Sandbox is disabled"):
            await sandbox.execute("echo hello")

    @pytest.mark.asyncio
    async def test_execute_streaming_raises_not_implemented(self) -> None:
        sandbox = NoOpSandbox()
        with pytest.raises(NotImplementedError, match="Sandbox is disabled"):
            async for _ in sandbox.execute_streaming("echo hello"):
                pass

    @pytest.mark.asyncio
    async def test_execute_code_raises_not_implemented(self) -> None:
        sandbox = NoOpSandbox()
        with pytest.raises(NotImplementedError, match="Sandbox is disabled"):
            await sandbox.execute_code("print(1)", language="python")

    @pytest.mark.asyncio
    async def test_execute_code_streaming_raises_not_implemented(self) -> None:
        sandbox = NoOpSandbox()
        with pytest.raises(NotImplementedError, match="Sandbox is disabled"):
            async for _ in sandbox.execute_code_streaming("print(1)", language="python"):
                pass

    @pytest.mark.asyncio
    async def test_read_file_raises_not_implemented(self) -> None:
        sandbox = NoOpSandbox()
        with pytest.raises(NotImplementedError, match="Sandbox is disabled"):
            await sandbox.read_file("test.txt")

    @pytest.mark.asyncio
    async def test_write_file_raises_not_implemented(self) -> None:
        sandbox = NoOpSandbox()
        with pytest.raises(NotImplementedError, match="Sandbox is disabled"):
            await sandbox.write_file("test.txt", b"content")

    @pytest.mark.asyncio
    async def test_remove_file_raises_not_implemented(self) -> None:
        sandbox = NoOpSandbox()
        with pytest.raises(NotImplementedError, match="Sandbox is disabled"):
            await sandbox.remove_file("test.txt")

    @pytest.mark.asyncio
    async def test_list_files_raises_not_implemented(self) -> None:
        sandbox = NoOpSandbox()
        with pytest.raises(NotImplementedError, match="Sandbox is disabled"):
            await sandbox.list_files(".")

    @pytest.mark.asyncio
    async def test_read_text_raises_not_implemented(self) -> None:
        """read_text delegates to read_file, which raises."""
        sandbox = NoOpSandbox()
        with pytest.raises(NotImplementedError, match="Sandbox is disabled"):
            await sandbox.read_text("test.txt")

    @pytest.mark.asyncio
    async def test_write_text_raises_not_implemented(self) -> None:
        """write_text delegates to write_file, which raises."""
        sandbox = NoOpSandbox()
        with pytest.raises(NotImplementedError, match="Sandbox is disabled"):
            await sandbox.write_text("test.txt", "content")

    def test_agent_with_noop_sandbox(self) -> None:
        """Agent can be constructed with NoOpSandbox."""
        from strands.agent.agent import Agent

        sandbox = NoOpSandbox()
        agent = Agent(model="test", sandbox=sandbox)
        assert agent.sandbox is sandbox
        assert isinstance(agent.sandbox, NoOpSandbox)
