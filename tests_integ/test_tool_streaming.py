"""Integration tests for direct tool call streaming (Issue #1436)."""

import pytest

from strands import Agent, tool


@tool
def simple_tool(value: int) -> int:
    """Simple tool for testing."""
    return value * 2


class TestToolStreaming:
    """Test tool streaming methods."""

    @pytest.mark.asyncio
    async def test_async_streaming(self):
        """Test async streaming captures events."""
        agent = Agent(tools=[simple_tool])
        events = []

        async for event in agent.tool.simple_tool.stream_async(value=5):
            events.append(event)

        assert len(events) > 0
        assert any(e.get("type") == "tool_result" for e in events)

    def test_sync_streaming(self):
        """Test sync streaming captures events."""
        agent = Agent(tools=[simple_tool])
        events = []

        for event in agent.tool.simple_tool.stream(value=5):
            events.append(event)

        assert len(events) > 0
        assert any(e.get("type") == "tool_result" for e in events)

    def test_backward_compatibility(self):
        """Test existing sync API unchanged."""
        agent = Agent(tools=[simple_tool])
        result = agent.tool.simple_tool(value=5)

        assert result["status"] == "success"
        assert result["content"][0]["text"] == "10"

    @pytest.mark.asyncio
    async def test_tool_not_found(self):
        """Test non-existent tool raises AttributeError."""
        agent = Agent(tools=[])

        with pytest.raises(AttributeError, match="Tool 'fake' not found"):
            async for _event in agent.tool.fake.stream_async():
                pass

    @pytest.mark.asyncio
    async def test_tool_error_captured_in_result(self):
        """Test tool errors are captured in tool_result events."""

        @tool
        def error_tool() -> str:
            raise ValueError("Test error")

        agent = Agent(tools=[error_tool])
        events = []

        async for event in agent.tool.error_tool.stream_async():
            events.append(event)

        # Should have at least one event
        assert len(events) > 0

        # Final event should be tool_result with error status
        final_event = events[-1]
        assert final_event.get("type") == "tool_result"
        tool_result = final_event.get("tool_result", {})
        assert tool_result.get("status") == "error"
        assert "Test error" in tool_result.get("content", [{}])[0].get("text", "")
