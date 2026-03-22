"""Unit tests for _ToolExecutor."""

import gc
import unittest.mock
import weakref

import pytest

from strands import Agent, tool
from strands.types._events import ToolResultEvent


class TestToolExecutor:
    """Test _ToolExecutor class."""

    def test_executor_is_callable(self):
        """Test tool executor is callable."""

        @tool
        def test_tool(x: int) -> int:
            return x * 2

        agent = Agent(tools=[test_tool])
        executor = agent.tool.test_tool

        # Should be callable
        result = executor(x=5)
        assert result["status"] == "success"

    def test_executor_has_streaming_methods(self):
        """Test executor has stream and stream_async methods."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])
        executor = agent.tool.test_tool

        assert hasattr(executor, "stream")
        assert hasattr(executor, "stream_async")
        assert callable(executor.stream)
        assert callable(executor.stream_async)

    def test_weakref_prevents_circular_reference(self):
        """Test weakref prevents agent from leaking."""

        @tool
        def test_tool(x: int) -> int:
            return x

        gc.disable()
        try:
            agent = Agent(tools=[test_tool])
            _ = agent.tool.test_tool  # Create executor to test weakref
            ref = weakref.ref(agent)

            del agent

            if ref() is not None:
                gc.collect()

            assert ref() is None
        finally:
            gc.enable()

    def test_executor_weakref_raises_on_deleted_agent(self):
        """Test accessing _agent property raises ReferenceError when agent deleted."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])
        executor = agent.tool.test_tool

        # Delete agent
        del agent
        gc.collect()

        # Accessing _agent should raise ReferenceError
        with pytest.raises(ReferenceError, match="Agent has been garbage collected"):
            _ = executor._agent

    @pytest.mark.asyncio
    async def test_stream_async_with_interrupt_raises(self):
        """Test stream_async raises when interrupt state activated."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])
        agent._interrupt_state.activate()

        with pytest.raises(RuntimeError, match="cannot directly call tool during interrupt"):
            async for _event in agent.tool.test_tool.stream_async(x=5):
                pass

    def test_find_normalized_tool_name_with_underscores(self):
        """Test tool name normalization replaces underscores with hyphens."""

        @tool(name="my-tool")
        def my_tool(x: int) -> int:
            return x

        agent = Agent(tools=[my_tool])
        executor = agent.tool.my_tool

        # Should find tool with hyphen name via underscore access
        result = executor(x=5)
        assert result["status"] == "success"

    def test_find_normalized_tool_name_not_found(self):
        """Test non-existent tool raises AttributeError."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])

        with pytest.raises(AttributeError, match="Tool 'nonexistent' not found"):
            _ = agent.tool.nonexistent(x=5)

    @pytest.mark.asyncio
    async def test_stream_async_basic(self, alist):
        """Test basic async streaming from direct tool call."""

        @tool
        def test_tool(x: int) -> int:
            return x * 2

        agent = Agent(tools=[test_tool])

        events = await alist(agent.tool.test_tool.stream_async(x=5))

        # Should yield at least one result event
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0]["tool_result"]["status"] == "success"
        assert "10" in result_events[0]["tool_result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_stream_async_with_error(self, alist):
        """Test async streaming handles tool errors."""

        @tool
        def error_tool(should_fail: bool) -> str:
            if should_fail:
                raise ValueError("Test error")
            return "success"

        agent = Agent(tools=[error_tool])

        events = await alist(agent.tool.error_tool.stream_async(should_fail=True))

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0]["tool_result"]["status"] == "error"
        assert "Test error" in result_events[0]["tool_result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_stream_async_tool_not_found(self):
        """Test stream_async with non-existent tool."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])

        with pytest.raises(AttributeError, match="Tool 'nonexistent' not found"):
            async for _ in agent.tool.nonexistent.stream_async(x=5):
                pass

    @pytest.mark.asyncio
    async def test_stream_async_with_interrupt_from_tool(self):
        """Test stream_async when tool raises interrupt."""

        @tool(context=True)
        def interrupt_tool(tool_context):
            tool_context.interrupt("test", reason="test")
            return "unreachable"

        agent = Agent(tools=[interrupt_tool])

        with pytest.raises(RuntimeError, match="cannot raise interrupt in direct tool call"):
            async for _ in agent.tool.interrupt_tool.stream_async():
                pass

        # Interrupt state should be deactivated
        assert not agent._interrupt_state.activated

    @pytest.mark.asyncio
    async def test_stream_async_after_agent_deleted(self):
        """Test stream_async raises when agent is garbage collected."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])
        executor = agent.tool.test_tool

        del agent
        gc.collect()

        with pytest.raises(ReferenceError, match="Agent has been garbage collected"):
            async for _ in executor.stream_async(x=5):
                pass

    @pytest.mark.asyncio
    async def test_stream_async_with_normalized_name(self, alist):
        """Test stream_async with underscore to hyphen normalization."""

        @tool(name="my-tool")
        def my_tool(x: int) -> int:
            return x * 3

        agent = Agent(tools=[my_tool])

        events = await alist(agent.tool.my_tool.stream_async(x=4))

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert "12" in result_events[0]["tool_result"]["content"][0]["text"]

    def test_stream_sync_basic(self):
        """Test synchronous streaming from direct tool call."""

        @tool
        def test_tool(x: int) -> int:
            return x * 2

        agent = Agent(tools=[test_tool])

        events = list(agent.tool.test_tool.stream(x=5))

        # Should yield at least one result event
        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0]["tool_result"]["status"] == "success"
        assert "10" in result_events[0]["tool_result"]["content"][0]["text"]

    def test_stream_sync_with_error(self):
        """Test sync streaming handles tool errors."""

        @tool
        def error_tool(should_fail: bool) -> str:
            if should_fail:
                raise ValueError("Sync error")
            return "success"

        agent = Agent(tools=[error_tool])

        events = list(agent.tool.error_tool.stream(should_fail=True))

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0]["tool_result"]["status"] == "error"
        assert "Sync error" in result_events[0]["tool_result"]["content"][0]["text"]

    def test_stream_sync_tool_not_found(self):
        """Test stream with non-existent tool."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])

        with pytest.raises(AttributeError, match="Tool 'nonexistent' not found"):
            list(agent.tool.nonexistent.stream(x=5))

    def test_stream_sync_with_interrupt_raises(self):
        """Test stream raises when interrupt state activated."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])
        agent._interrupt_state.activate()

        with pytest.raises(RuntimeError, match="cannot directly call tool during interrupt"):
            list(agent.tool.test_tool.stream(x=5))

    def test_stream_sync_with_interrupt_from_tool(self):
        """Test stream when tool raises interrupt."""

        @tool(context=True)
        def interrupt_tool(tool_context):
            tool_context.interrupt("test", reason="test")
            return "unreachable"

        agent = Agent(tools=[interrupt_tool])

        with pytest.raises(RuntimeError, match="cannot raise interrupt in direct tool call"):
            list(agent.tool.interrupt_tool.stream())

        # Interrupt state should be deactivated
        assert not agent._interrupt_state.activated

    def test_stream_sync_after_agent_deleted(self):
        """Test stream raises when agent is garbage collected."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])
        executor = agent.tool.test_tool

        del agent
        gc.collect()

        with pytest.raises(ReferenceError, match="Agent has been garbage collected"):
            list(executor.stream(x=5))

    def test_find_normalized_tool_name_exact_match(self):
        """Test _find_normalized_tool_name with exact match."""

        @tool(name="exact_name")
        def exact_tool(x: int) -> int:
            return x

        agent = Agent(tools=[exact_tool])
        executor = agent.tool.exact_name

        normalized = executor._find_normalized_tool_name("exact_name")
        assert normalized == "exact_name"

    @pytest.mark.asyncio
    async def test_stream_async_generates_unique_tool_ids(self, alist):
        """Test stream_async generates unique tool use IDs."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])

        with unittest.mock.patch("strands.tools._caller.random.randint") as mock_randint:
            mock_randint.side_effect = [111, 222, 333]

            tool_use_ids = []
            for _ in range(3):
                events = await alist(agent.tool.test_tool.stream_async(x=1))
                result_events = [e for e in events if isinstance(e, ToolResultEvent)]
                tool_use_ids.append(result_events[0]["tool_result"]["toolUseId"])

            # All IDs should be unique
            assert len(set(tool_use_ids)) == 3
            assert "tooluse_test_tool_111" in tool_use_ids
            assert "tooluse_test_tool_222" in tool_use_ids
            assert "tooluse_test_tool_333" in tool_use_ids
