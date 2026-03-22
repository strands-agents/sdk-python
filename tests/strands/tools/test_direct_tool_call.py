"""Unit tests for _DirectToolCall."""

import gc
import unittest.mock
import weakref

import pytest

from strands import Agent, tool
from strands.types._events import ToolResultEvent


class TestDirectToolCall:
    """Test _DirectToolCall class."""

    def test_is_callable(self):
        """Test direct tool call is callable."""

        @tool
        def test_tool(x: int) -> int:
            return x * 2

        agent = Agent(tools=[test_tool])
        direct_call = agent.tool.test_tool

        result = direct_call(x=5)
        assert result["status"] == "success"

    def test_has_streaming_methods(self):
        """Test direct tool call has stream and stream_async methods."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])
        direct_call = agent.tool.test_tool

        assert hasattr(direct_call, "stream")
        assert hasattr(direct_call, "stream_async")
        assert callable(direct_call.stream)
        assert callable(direct_call.stream_async)

    def test_weakref_prevents_circular_reference(self):
        """Test weakref prevents agent from leaking."""

        @tool
        def test_tool(x: int) -> int:
            return x

        gc.disable()
        try:
            agent = Agent(tools=[test_tool])
            _ = agent.tool.test_tool
            ref = weakref.ref(agent)

            del agent

            if ref() is not None:
                gc.collect()

            assert ref() is None
        finally:
            gc.enable()

    def test_weakref_raises_on_deleted_agent(self):
        """Test accessing _agent property raises ReferenceError when agent deleted."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])
        direct_call = agent.tool.test_tool

        del agent
        gc.collect()

        with pytest.raises(ReferenceError, match="Agent has been garbage collected"):
            _ = direct_call._agent

    def test_find_normalized_tool_name_with_underscores(self):
        """Test tool name normalization replaces underscores with hyphens."""

        @tool(name="my-tool")
        def my_tool(x: int) -> int:
            return x

        agent = Agent(tools=[my_tool])
        direct_call = agent.tool.my_tool

        result = direct_call(x=5)
        assert result["status"] == "success"

    def test_find_normalized_tool_name_not_found(self):
        """Test non-existent tool raises AttributeError."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])

        with pytest.raises(AttributeError, match="Tool 'nonexistent' not found"):
            _ = agent.tool.nonexistent(x=5)

    def test_find_normalized_tool_name_exact_match(self):
        """Test _find_normalized_tool_name with exact match."""

        @tool(name="exact_name")
        def exact_tool(x: int) -> int:
            return x

        agent = Agent(tools=[exact_tool])
        direct_call = agent.tool.exact_name

        normalized = direct_call._find_normalized_tool_name("exact_name")
        assert normalized == "exact_name"


class TestDirectToolCallStreamAsync:
    """Test _DirectToolCall.stream_async()."""

    @pytest.mark.asyncio
    async def test_basic(self, alist):
        """Test basic async streaming from direct tool call."""

        @tool
        def test_tool(x: int) -> int:
            return x * 2

        agent = Agent(tools=[test_tool])

        events = await alist(agent.tool.test_tool.stream_async(x=5))

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0]["tool_result"]["status"] == "success"
        assert "10" in result_events[0]["tool_result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_with_error(self, alist):
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
    async def test_tool_not_found(self):
        """Test stream_async with non-existent tool."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])

        with pytest.raises(AttributeError, match="Tool 'nonexistent' not found"):
            async for _ in agent.tool.nonexistent.stream_async(x=5):
                pass

    @pytest.mark.asyncio
    async def test_with_interrupt_state_raises(self):
        """Test stream_async raises when interrupt state activated."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])
        agent._interrupt_state.activate()

        with pytest.raises(RuntimeError, match="cannot directly call tool during interrupt"):
            async for _event in agent.tool.test_tool.stream_async(x=5):
                pass

    @pytest.mark.asyncio
    async def test_with_interrupt_from_tool(self):
        """Test stream_async when tool raises interrupt."""

        @tool(context=True)
        def interrupt_tool(tool_context):
            tool_context.interrupt("test", reason="test")
            return "unreachable"

        agent = Agent(tools=[interrupt_tool])

        with pytest.raises(RuntimeError, match="cannot raise interrupt in direct tool call"):
            async for _ in agent.tool.interrupt_tool.stream_async():
                pass

        assert not agent._interrupt_state.activated

    @pytest.mark.asyncio
    async def test_after_agent_deleted(self):
        """Test stream_async raises when agent is garbage collected."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])
        direct_call = agent.tool.test_tool

        del agent
        gc.collect()

        with pytest.raises(ReferenceError, match="Agent has been garbage collected"):
            async for _ in direct_call.stream_async(x=5):
                pass

    @pytest.mark.asyncio
    async def test_with_normalized_name(self, alist):
        """Test stream_async with underscore to hyphen normalization."""

        @tool(name="my-tool")
        def my_tool(x: int) -> int:
            return x * 3

        agent = Agent(tools=[my_tool])

        events = await alist(agent.tool.my_tool.stream_async(x=4))

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert "12" in result_events[0]["tool_result"]["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_generates_unique_tool_ids(self, alist):
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

            assert len(set(tool_use_ids)) == 3
            assert "tooluse_test_tool_111" in tool_use_ids
            assert "tooluse_test_tool_222" in tool_use_ids
            assert "tooluse_test_tool_333" in tool_use_ids


class TestDirectToolCallStream:
    """Test _DirectToolCall.stream() (sync streaming via thread+queue bridge)."""

    def test_basic(self):
        """Test synchronous streaming from direct tool call."""

        @tool
        def test_tool(x: int) -> int:
            return x * 2

        agent = Agent(tools=[test_tool])

        events = list(agent.tool.test_tool.stream(x=5))

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) == 1
        assert result_events[0]["tool_result"]["status"] == "success"
        assert "10" in result_events[0]["tool_result"]["content"][0]["text"]

    def test_with_error(self):
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

    def test_tool_not_found(self):
        """Test stream with non-existent tool."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])

        with pytest.raises(AttributeError, match="Tool 'nonexistent' not found"):
            list(agent.tool.nonexistent.stream(x=5))

    def test_with_interrupt_state_raises(self):
        """Test stream raises when interrupt state activated."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])
        agent._interrupt_state.activate()

        with pytest.raises(RuntimeError, match="cannot directly call tool during interrupt"):
            list(agent.tool.test_tool.stream(x=5))

    def test_with_interrupt_from_tool(self):
        """Test stream when tool raises interrupt."""

        @tool(context=True)
        def interrupt_tool(tool_context):
            tool_context.interrupt("test", reason="test")
            return "unreachable"

        agent = Agent(tools=[interrupt_tool])

        with pytest.raises(RuntimeError, match="cannot raise interrupt in direct tool call"):
            list(agent.tool.interrupt_tool.stream())

        assert not agent._interrupt_state.activated

    def test_after_agent_deleted(self):
        """Test stream raises when agent is garbage collected."""

        @tool
        def test_tool(x: int) -> int:
            return x

        agent = Agent(tools=[test_tool])
        direct_call = agent.tool.test_tool

        del agent
        gc.collect()

        with pytest.raises(ReferenceError, match="Agent has been garbage collected"):
            list(direct_call.stream(x=5))
