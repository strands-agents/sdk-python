"""Tests for agent cancellation functionality using agent.cancel() API."""

import asyncio
import threading
import time

import pytest

from strands import Agent
from tests.fixtures.mocked_model_provider import MockedModelProvider

# Default agent response for simple tests
DEFAULT_RESPONSE = {
    "role": "assistant",
    "content": [{"text": "Hello! How can I help you?"}],
}


@pytest.mark.asyncio
async def test_agent_cancel_before_invocation():
    """Test agent.cancel() before invocation starts.

    Verifies that calling cancel() before invoke_async() results in
    immediate cancellation without any model calls.
    """
    agent = Agent(model=MockedModelProvider([DEFAULT_RESPONSE]))

    # Cancel before invocation
    agent.cancel()

    result = await agent.invoke_async("Hello")

    assert result.stop_reason == "cancelled"
    assert result.message == {"role": "assistant", "content": []}


@pytest.mark.asyncio
async def test_agent_cancel_during_execution():
    """Test agent.cancel() during execution.

    Verifies that calling cancel() while the agent is running
    stops execution at the next checkpoint.
    """

    # Create a model provider that simulates a delay
    class DelayedModelProvider(MockedModelProvider):
        async def stream(self, *args, **kwargs):
            # Add a small delay before streaming
            await asyncio.sleep(0.1)
            async for event in super().stream(*args, **kwargs):
                yield event

    agent = Agent(model=DelayedModelProvider([DEFAULT_RESPONSE]))

    # Cancel after a short delay (during execution)
    async def cancel_after_delay():
        await asyncio.sleep(0.05)
        agent.cancel()

    cancel_task = asyncio.create_task(cancel_after_delay())
    result = await agent.invoke_async("Hello")
    await cancel_task

    assert result.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_agent_cancel_with_tools():
    """Test agent.cancel() during tool execution.

    Verifies that cancellation works correctly when tools are being executed.
    """
    from strands import tool

    tool_executed = []

    @tool
    def slow_tool(x: int) -> int:
        """A slow tool that takes time to execute."""
        tool_executed.append(x)
        time.sleep(0.1)
        return x * 2

    # Create a response with tool use
    tool_use_response = {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "toolUseId": "tool_1",
                    "name": "slow_tool",
                    "input": {"x": 5},
                }
            }
        ],
    }

    agent = Agent(
        model=MockedModelProvider([tool_use_response]),
        tools=[slow_tool],
    )

    # Cancel during tool execution
    async def cancel_after_delay():
        await asyncio.sleep(0.05)
        agent.cancel()

    cancel_task = asyncio.create_task(cancel_after_delay())
    result = await agent.invoke_async("Use the tool")
    await cancel_task

    assert result.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_agent_without_cancellation():
    """Test that agent works normally without cancellation.

    Verifies that when cancel() is not called, the agent executes
    normally and completes successfully.
    """
    agent = Agent(model=MockedModelProvider([DEFAULT_RESPONSE]))

    result = await agent.invoke_async("Hello")

    assert result.stop_reason == "end_turn"
    assert result.message["role"] == "assistant"


@pytest.mark.asyncio
async def test_agent_cancel_idempotent():
    """Test that calling cancel() multiple times is safe.

    Verifies that multiple cancel() calls are idempotent and don't
    cause any issues.
    """
    agent = Agent(model=MockedModelProvider([DEFAULT_RESPONSE]))

    # Cancel multiple times
    agent.cancel()
    agent.cancel()
    agent.cancel()

    result = await agent.invoke_async("Hello")

    assert result.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_agent_cancel_from_thread():
    """Test agent.cancel() from another thread.

    Verifies thread-safety of the cancel() method when called
    from a background thread.
    """

    # Create a model provider with delay
    class DelayedModelProvider(MockedModelProvider):
        async def stream(self, *args, **kwargs):
            await asyncio.sleep(0.1)
            async for event in super().stream(*args, **kwargs):
                yield event

    agent = Agent(model=DelayedModelProvider([DEFAULT_RESPONSE]))

    # Cancel from another thread
    def cancel_from_thread():
        time.sleep(0.05)
        agent.cancel()

    thread = threading.Thread(target=cancel_from_thread)
    thread.start()

    result = await agent.invoke_async("Hello")
    thread.join()

    assert result.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_agent_cancel_streaming():
    """Test cancellation during streaming response.

    Verifies that cancellation works correctly when using
    the streaming API (stream_async).
    """

    # Create a model provider that streams slowly
    class SlowStreamingModelProvider(MockedModelProvider):
        async def stream(self, *args, **kwargs):
            # Stream with delays between chunks
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {}}}

            # Stream multiple chunks with delays
            for i in range(10):
                await asyncio.sleep(0.05)
                yield {"contentBlockDelta": {"delta": {"text": f"chunk {i} "}}}

            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}

    agent = Agent(model=SlowStreamingModelProvider([DEFAULT_RESPONSE]))

    # Cancel after receiving a few chunks
    async def cancel_after_delay():
        await asyncio.sleep(0.15)  # Let a few chunks through
        agent.cancel()

    cancel_task = asyncio.create_task(cancel_after_delay())

    events = []
    async for event in agent.stream_async("Hello"):
        events.append(event)
        if event.get("result"):
            break

    await cancel_task

    # Find the result event
    result_event = next((e for e in events if e.get("result")), None)
    assert result_event is not None
    assert result_event["result"].stop_reason == "cancelled"
