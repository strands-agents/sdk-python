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
    assert result.message == {"role": "assistant", "content": [{"text": "Cancelled by user"}]}


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


@pytest.mark.asyncio
async def test_agent_cancel_before_tool_execution_adds_tool_results():
    """Test that cancelling before tool execution adds tool_result messages.

    Verifies that when cancellation occurs after model returns tool_use but before
    tools execute, proper tool_result messages are added to maintain valid conversation state.
    This prevents the "tool_use without tool_result" error on next invocation.
    """
    from strands import tool

    @tool
    def calculator(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    # Create a response with tool use
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

    # Create a model that will return tool_use, then on next call return end_turn
    # This simulates: model returns tool_use -> tools cancelled -> model continues
    agent = Agent(
        model=MockedModelProvider([tool_use_response, DEFAULT_RESPONSE]),
        tools=[calculator],
    )

    # Use a hook to cancel via agent.cancel() which will be checked at checkpoint 4
    # We need to cancel AFTER model returns but BEFORE tool executor is called
    # The only way to do this reliably is to cancel in AfterModelCallEvent
    from strands.hooks import AfterModelCallEvent

    async def cancel_after_model(event: AfterModelCallEvent):
        # Only cancel if model returned tool_use
        if event.stop_response and event.stop_response.stop_reason == "tool_use":
            agent.cancel()

    agent.add_hook(cancel_after_model, AfterModelCallEvent)

    result = await agent.invoke_async("Calculate 5 + 3")

    # Verify cancellation occurred
    assert result.stop_reason == "cancelled"

    # Verify that tool_result message was added to conversation
    # Should have: user message, assistant message with tool_use, user message with tool_result
    assert len(agent.messages) == 3
    assert agent.messages[0]["role"] == "user"
    assert agent.messages[1]["role"] == "assistant"
    assert agent.messages[2]["role"] == "user"

    # Verify the tool_result message has the cancellation error
    tool_result_content = agent.messages[2]["content"]
    assert len(tool_result_content) == 1
    assert "toolResult" in tool_result_content[0]

    tool_result = tool_result_content[0]["toolResult"]
    assert tool_result["toolUseId"] == "tool_1"
    assert tool_result["status"] == "error"
    assert "cancelled" in tool_result["content"][0]["text"].lower()


@pytest.mark.asyncio
async def test_all_checkpoints_and_reinvocation():
    """Comprehensive test covering all 4 checkpoints and reinvocation after cancellation.

    This test verifies:
    1. Checkpoint 1: Cancellation at start of event loop cycle
    2. Checkpoint 2: Cancellation before model call
    3. Checkpoint 3: Cancellation during streaming
    4. Checkpoint 4: Cancellation before tool execution (with tool_result messages)
    5. Agent can be reinvoked after cancellation
    6. Conversation history is preserved correctly
    """
    from strands import tool
    from strands.hooks import AfterModelCallEvent

    @tool
    def calculator(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    # --- Test Checkpoint 1: Cancel before invocation ---
    agent1 = Agent(model=MockedModelProvider([DEFAULT_RESPONSE]))
    agent1.cancel()
    result1 = await agent1.invoke_async("Hello")
    assert result1.stop_reason == "cancelled"
    assert result1.message["content"] == [{"text": "Cancelled by user"}]

    # --- Test Checkpoint 2: Cancel before model call (during execution) ---
    class DelayedModelProvider(MockedModelProvider):
        async def stream(self, *args, **kwargs):
            await asyncio.sleep(0.1)
            async for event in super().stream(*args, **kwargs):
                yield event

    agent2 = Agent(model=DelayedModelProvider([DEFAULT_RESPONSE]))

    async def cancel_after_delay():
        await asyncio.sleep(0.05)
        agent2.cancel()

    cancel_task = asyncio.create_task(cancel_after_delay())
    result2 = await agent2.invoke_async("Hello")
    await cancel_task
    assert result2.stop_reason == "cancelled"
    assert result2.message["content"] == [{"text": "Cancelled by user"}]

    # --- Test Checkpoint 3: Cancel during streaming ---
    class SlowStreamingModelProvider(MockedModelProvider):
        async def stream(self, *args, **kwargs):
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {}}}
            for i in range(10):
                await asyncio.sleep(0.05)
                yield {"contentBlockDelta": {"delta": {"text": f"chunk {i} "}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}

    agent3 = Agent(model=SlowStreamingModelProvider([DEFAULT_RESPONSE]))

    async def cancel_during_stream():
        await asyncio.sleep(0.15)
        agent3.cancel()

    cancel_task = asyncio.create_task(cancel_during_stream())
    result3 = await agent3.invoke_async("Hello")
    await cancel_task
    assert result3.stop_reason == "cancelled"
    assert result3.message["content"] == [{"text": "Cancelled by user"}]

    # --- Test Checkpoint 4: Cancel before tool execution ---
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

    agent4 = Agent(
        model=MockedModelProvider([tool_use_response, DEFAULT_RESPONSE]),
        tools=[calculator],
    )

    async def cancel_after_model(event: AfterModelCallEvent):
        if event.stop_response and event.stop_response.stop_reason == "tool_use":
            agent4.cancel()

    agent4.add_hook(cancel_after_model, AfterModelCallEvent)
    result4 = await agent4.invoke_async("Calculate 5 + 3")

    assert result4.stop_reason == "cancelled"
    # Verify tool_result message was added
    assert len(agent4.messages) == 3
    assert agent4.messages[2]["role"] == "user"
    tool_result = agent4.messages[2]["content"][0]["toolResult"]
    assert tool_result["status"] == "error"
    assert "cancelled" in tool_result["content"][0]["text"].lower()

    # --- Test Reinvocation: Agent can be used again after cancellation ---
    # Reuse agent4 which has conversation history from checkpoint 4 test
    # The stop_signal is cleared at the end of each invocation
    result5 = await agent4.invoke_async("Continue")
    assert result5.stop_reason == "end_turn"
    assert result5.message["role"] == "assistant"
    # Verify conversation history was preserved:
    # 1. user: "Calculate 5 + 3"
    # 2. assistant: tool_use
    # 3. user: tool_result (cancelled)
    # 4. user: "Continue"
    # 5. assistant: response
    assert len(agent4.messages) == 5
