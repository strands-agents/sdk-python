"""Tests for agent cancellation functionality using agent.cancel() API."""

import asyncio
import threading
from unittest.mock import ANY

import pytest

from strands import Agent, tool
from strands.hooks import AfterModelCallEvent
from tests.fixtures.mocked_model_provider import MockedModelProvider

# Default agent response for simple tests
DEFAULT_RESPONSE = {
    "role": "assistant",
    "content": [{"text": "Hello! How can I help you?"}],
}


class DelayedModelProvider(MockedModelProvider):
    """Model provider that blocks streaming until signaled, for cancel timing tests."""

    def __init__(self, responses: list) -> None:
        super().__init__(responses)
        self.streaming_started = asyncio.Event()
        self.cancel_ready = asyncio.Event()

    async def stream(self, *args, **kwargs):
        self.streaming_started.set()
        await self.cancel_ready.wait()
        async for event in super().stream(*args, **kwargs):
            yield event


@pytest.mark.asyncio
async def test_agent_cancel_before_invocation():
    """Test agent.cancel() before invocation starts.

    Verifies that calling cancel() before invoke_async() (agent runs) has no effect.
    """
    agent = Agent(model=MockedModelProvider([DEFAULT_RESPONSE]))

    # Cancel before invocation
    agent.cancel()

    result = await agent.invoke_async("Hello")

    assert result.stop_reason == "end_turn"
    assert result.message == {**DEFAULT_RESPONSE, "metadata": ANY}


@pytest.mark.asyncio
async def test_agent_cancel_during_execution():
    """Test agent.cancel() during execution.

    Verifies that calling cancel() while the agent is running
    stops execution at the next checkpoint.
    """
    model = DelayedModelProvider([DEFAULT_RESPONSE])
    agent = Agent(model=model)

    async def cancel_when_ready():
        await model.streaming_started.wait()
        agent.cancel()
        model.cancel_ready.set()

    cancel_task = asyncio.create_task(cancel_when_ready())
    result = await agent.invoke_async("Hello")
    await cancel_task

    assert result.stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_agent_cancel_with_tools():
    """Test agent.cancel() during tool execution.

    Verifies that cancellation works correctly when tools are being executed.
    Uses AfterModelCallEvent hook to cancel deterministically after model returns tool_use.
    """
    tool_executed = []

    @tool
    def slow_tool(x: int) -> int:
        """A tool for testing."""
        tool_executed.append(x)
        return x * 2

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
        model=MockedModelProvider([tool_use_response, DEFAULT_RESPONSE]),
        tools=[slow_tool],
    )

    # Cancel deterministically after model returns tool_use
    async def cancel_after_model(event: AfterModelCallEvent):
        if event.stop_response and event.stop_response.stop_reason == "tool_use":
            agent.cancel()

    agent.add_hook(cancel_after_model, AfterModelCallEvent)

    result = await agent.invoke_async("Use the tool")

    assert result.stop_reason == "cancelled"


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

    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_agent_cancel_from_thread():
    """Test agent.cancel() from another thread.

    Verifies thread-safety of the cancel() method when called
    from a background thread.
    """
    loop = asyncio.get_running_loop()

    model = DelayedModelProvider([DEFAULT_RESPONSE])
    agent = Agent(model=model)

    def cancel_from_thread():
        # Wait for streaming to start before cancelling
        asyncio.run_coroutine_threadsafe(model.streaming_started.wait(), loop).result()
        agent.cancel()
        loop.call_soon_threadsafe(model.cancel_ready.set)

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
    chunks_yielded = asyncio.Event()
    cancel_done = asyncio.Event()

    class SlowStreamingModelProvider(MockedModelProvider):
        async def stream(self, *args, **kwargs):
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {}}}

            for i in range(10):
                yield {"contentBlockDelta": {"delta": {"text": f"chunk {i} "}}}
                if i == 2:
                    # Signal after a few chunks so cancel can fire
                    chunks_yielded.set()
                    # Wait for cancel to complete before continuing
                    await cancel_done.wait()

            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}

    agent = Agent(model=SlowStreamingModelProvider([DEFAULT_RESPONSE]))

    async def cancel_after_chunks():
        await chunks_yielded.wait()
        agent.cancel()
        cancel_done.set()

    cancel_task = asyncio.create_task(cancel_after_chunks())

    events = []
    async for event in agent.stream_async("Hello"):
        events.append(event)
        if event.get("result"):
            break

    await cancel_task

    result_event = next((e for e in events if e.get("result")), None)
    assert result_event is not None
    assert result_event["result"].stop_reason == "cancelled"


@pytest.mark.asyncio
async def test_agent_cancel_before_tool_execution_adds_tool_results():
    """Test that cancelling before tool execution adds tool_result messages.

    Verifies that when cancellation occurs after model returns tool_use but before
    tools execute, proper tool_result messages are added to maintain valid conversation state.
    This prevents the "tool_use without tool_result" error on next invocation.
    """

    @tool
    def calculator(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    tool_use_response = {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "toolUseId": "tool_1",
                    "name": "calculator",
                    "input": {"x": 5, "y": 3},
                }
            }
        ],
    }

    agent = Agent(
        model=MockedModelProvider([tool_use_response, DEFAULT_RESPONSE]),
        tools=[calculator],
    )

    async def cancel_after_model(event: AfterModelCallEvent):
        if event.stop_response and event.stop_response.stop_reason == "tool_use":
            agent.cancel()

    agent.add_hook(cancel_after_model, AfterModelCallEvent)

    result = await agent.invoke_async("Calculate 5 + 3")

    assert result.stop_reason == "cancelled"

    # Should have: user message, assistant message with tool_use, user message with tool_result
    assert len(agent.messages) == 3
    assert agent.messages[0]["role"] == "user"
    assert agent.messages[1]["role"] == "assistant"
    assert agent.messages[2]["role"] == "user"

    tool_result_content = agent.messages[2]["content"]
    assert len(tool_result_content) == 1
    assert "toolResult" in tool_result_content[0]

    tool_result = tool_result_content[0]["toolResult"]
    assert tool_result["toolUseId"] == "tool_1"
    assert tool_result["status"] == "error"
    assert "cancelled" in tool_result["content"][0]["text"].lower()


@pytest.mark.asyncio
async def test_agent_cancel_continue_after():
    """Test that agent is reusable after cancellation.

    Verifies that the cancel signal is cleared after an invocation completes,
    allowing subsequent invocations to run normally.
    """
    model = DelayedModelProvider([DEFAULT_RESPONSE, DEFAULT_RESPONSE])
    agent = Agent(model=model)

    async def cancel_when_ready():
        await model.streaming_started.wait()
        agent.cancel()
        model.cancel_ready.set()

    cancel_task = asyncio.create_task(cancel_when_ready())
    result1 = await agent.invoke_async("Hello")
    await cancel_task
    assert result1.stop_reason == "cancelled"

    # Second invocation should work normally
    result2 = await agent.invoke_async("Hello again")
    assert result2.stop_reason == "end_turn"
