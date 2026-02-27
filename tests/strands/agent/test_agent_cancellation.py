"""Tests for agent cancellation functionality."""

import asyncio
import time

import pytest

from strands import Agent, CancellationToken
from tests.fixtures.mocked_model_provider import MockedModelProvider

# Default agent response for simple tests
DEFAULT_RESPONSE = {
    "role": "assistant",
    "content": [{"text": "Hello! How can I help you?"}],
}


@pytest.mark.asyncio
async def test_agent_cancellation_before_model_call():
    """Test cancellation before model call starts.

    This test verifies that when a cancellation token is cancelled before
    the agent starts processing, the agent immediately stops with a
    'cancelled' stop reason without making any model calls.
    """
    token = CancellationToken()
    agent = Agent(
        model=MockedModelProvider([DEFAULT_RESPONSE]),
        cancellation_token=token,
    )

    # Cancel immediately before invocation
    token.cancel()

    result = await agent.invoke_async("Hello")

    assert result.stop_reason == "cancelled"
    # When cancelled, we return an empty assistant message structure
    assert result.message == {"role": "assistant", "content": []}


@pytest.mark.asyncio
async def test_agent_cancellation_during_execution():
    """Test cancellation during agent execution.

    This test verifies that when a cancellation token is cancelled while
    the agent is executing, the agent detects the cancellation at the next
    checkpoint and stops gracefully with a 'cancelled' stop reason.
    """
    token = CancellationToken()

    # Create a model provider that simulates a delay
    class DelayedModelProvider(MockedModelProvider):
        async def stream(self, *args, **kwargs):
            # Add a small delay before streaming
            await asyncio.sleep(0.1)
            async for event in super().stream(*args, **kwargs):
                yield event

    agent = Agent(
        model=DelayedModelProvider([DEFAULT_RESPONSE]),
        cancellation_token=token,
    )

    # Cancel after a short delay (during execution)
    async def cancel_after_delay():
        await asyncio.sleep(0.05)
        token.cancel()

    cancel_task = asyncio.create_task(cancel_after_delay())
    result = await agent.invoke_async("Hello")
    await cancel_task

    assert result.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_agent_cancellation_with_tools():
    """Test cancellation during tool execution.

    This test verifies that when a cancellation token is cancelled while
    tools are being executed, the agent stops gracefully and doesn't
    execute remaining tools.
    """
    from strands import tool

    tool_executed = []

    @tool
    def slow_tool(x: int) -> int:
        """A slow tool that takes time to execute."""
        tool_executed.append(x)
        time.sleep(0.1)
        return x * 2

    token = CancellationToken()

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
        cancellation_token=token,
    )

    # Cancel during tool execution
    async def cancel_after_delay():
        await asyncio.sleep(0.05)
        token.cancel()

    cancel_task = asyncio.create_task(cancel_after_delay())
    result = await agent.invoke_async("Use the tool")
    await cancel_task

    assert result.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_agent_no_cancellation_token():
    """Test that agent works normally without cancellation token.

    This test verifies that when no cancellation token is provided,
    the agent executes normally and completes successfully.
    """
    agent = Agent(model=MockedModelProvider([DEFAULT_RESPONSE]))

    result = await agent.invoke_async("Hello")

    assert result.stop_reason == "end_turn"
    assert result.message["role"] == "assistant"


@pytest.mark.asyncio
async def test_agent_cancellation_idempotent():
    """Test that multiple cancellations are safe.

    This test verifies that calling cancel() multiple times on the same
    token doesn't cause any issues and the agent still stops gracefully.
    """
    token = CancellationToken()
    agent = Agent(
        model=MockedModelProvider([DEFAULT_RESPONSE]),
        cancellation_token=token,
    )

    # Cancel multiple times
    token.cancel()
    token.cancel()
    token.cancel()

    result = await agent.invoke_async("Hello")

    assert result.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_agent_cancellation_from_different_thread():
    """Test cancellation from a different thread.

    This test verifies that the cancellation token can be cancelled from
    a different thread (simulating a web request or external system) and
    the agent will detect it and stop gracefully.
    """
    import threading

    token = CancellationToken()

    # Create a model provider with delay
    class DelayedModelProvider(MockedModelProvider):
        async def stream(self, *args, **kwargs):
            await asyncio.sleep(0.1)
            async for event in super().stream(*args, **kwargs):
                yield event

    agent = Agent(
        model=DelayedModelProvider([DEFAULT_RESPONSE]),
        cancellation_token=token,
    )

    # Cancel from a different thread
    def cancel_from_thread():
        time.sleep(0.05)
        token.cancel()

    cancel_thread = threading.Thread(target=cancel_from_thread)
    cancel_thread.start()

    result = await agent.invoke_async("Hello")

    cancel_thread.join()

    assert result.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_agent_cancellation_shared_token():
    """Test that multiple agents can share the same cancellation token.

    This test verifies that when multiple agents share the same cancellation
    token, cancelling the token affects all agents using it.
    """
    token = CancellationToken()

    agent1 = Agent(
        model=MockedModelProvider([DEFAULT_RESPONSE]),
        cancellation_token=token,
    )

    agent2 = Agent(
        model=MockedModelProvider([DEFAULT_RESPONSE]),
        cancellation_token=token,
    )

    # Cancel the shared token
    token.cancel()

    result1 = await agent1.invoke_async("Hello from agent 1")
    result2 = await agent2.invoke_async("Hello from agent 2")

    assert result1.stop_reason == "cancelled"
    assert result2.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_agent_cancellation_streaming():
    """Test cancellation during streaming response.

    This test verifies that cancellation works correctly when using
    the streaming API (stream_async).
    """
    token = CancellationToken()

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

    agent = Agent(
        model=SlowStreamingModelProvider([DEFAULT_RESPONSE]),
        cancellation_token=token,
    )

    # Cancel after receiving a few chunks
    async def cancel_after_delay():
        await asyncio.sleep(0.15)  # Let a few chunks through
        token.cancel()

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
