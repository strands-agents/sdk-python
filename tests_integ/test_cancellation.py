"""Integration tests for cancellation with real model providers.

These tests verify that cancellation works correctly with actual model providers
like Bedrock, Anthropic, OpenAI, etc. They require valid credentials and may
incur API costs.

To run these tests:
    hatch run test-integ tests_integ/test_cancellation.py
"""

import asyncio
import os
import threading
import time

import pytest

from strands import Agent, CancellationToken, tool

# Skip all tests if no model credentials are available
pytestmark = pytest.mark.skipif(
    not any(
        [
            os.getenv("AWS_REGION"),  # Bedrock
            os.getenv("ANTHROPIC_API_KEY"),  # Anthropic
            os.getenv("OPENAI_API_KEY"),  # OpenAI
        ]
    ),
    reason="No model provider credentials found",
)


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("AWS_REGION"), reason="AWS credentials not available")
async def test_cancellation_with_bedrock():
    """Test cancellation with Amazon Bedrock model.

    This test verifies that cancellation works correctly with a real Bedrock
    model. It starts a long-running request and cancels it mid-execution.
    """
    from strands.models import BedrockModel

    token = CancellationToken()
    agent = Agent(
        model=BedrockModel("anthropic.claude-3-haiku-20240307-v1:0"),
        cancellation_token=token,
    )

    # Cancel after 2 seconds
    async def cancel_after_delay():
        await asyncio.sleep(2.0)
        token.cancel()

    cancel_task = asyncio.create_task(cancel_after_delay())

    # Request a long response that should take more than 2 seconds
    result = await agent.invoke_async(
        "Write a detailed 1000-word essay about the history of space exploration, "
        "including major milestones, key figures, and technological breakthroughs."
    )

    await cancel_task

    assert result.stop_reason == "cancelled"
    # The message might be empty or partially complete
    assert result.message is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not available")
async def test_cancellation_with_anthropic():
    """Test cancellation with Anthropic Claude model.

    This test verifies that cancellation works correctly with the Anthropic
    API. It starts a long-running request and cancels it mid-execution.
    """
    from strands.models import AnthropicModel

    token = CancellationToken()
    agent = Agent(
        model=AnthropicModel("claude-3-haiku-20240307"),
        cancellation_token=token,
    )

    # Cancel after 2 seconds
    async def cancel_after_delay():
        await asyncio.sleep(2.0)
        token.cancel()

    cancel_task = asyncio.create_task(cancel_after_delay())

    # Request a long response
    result = await agent.invoke_async(
        "Write a detailed 1000-word essay about artificial intelligence, "
        "covering its history, current applications, and future potential."
    )

    await cancel_task

    assert result.stop_reason == "cancelled"
    assert result.message is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
async def test_cancellation_with_openai():
    """Test cancellation with OpenAI model.

    This test verifies that cancellation works correctly with the OpenAI
    API. It starts a long-running request and cancels it mid-execution.
    """
    from strands.models import OpenAIModel

    token = CancellationToken()
    agent = Agent(
        model=OpenAIModel("gpt-4o-mini"),
        cancellation_token=token,
    )

    # Cancel after 2 seconds
    async def cancel_after_delay():
        await asyncio.sleep(2.0)
        token.cancel()

    cancel_task = asyncio.create_task(cancel_after_delay())

    # Request a long response
    result = await agent.invoke_async(
        "Write a detailed 1000-word essay about quantum computing, "
        "explaining the principles, current state, and potential applications."
    )

    await cancel_task

    assert result.stop_reason == "cancelled"
    assert result.message is not None


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("AWS_REGION"), reason="AWS credentials not available")
async def test_cancellation_during_streaming_bedrock():
    """Test cancellation during streaming with Bedrock.

    This test verifies that cancellation works correctly when using the
    streaming API with a real Bedrock model.
    """
    from strands.models import BedrockModel

    token = CancellationToken()
    agent = Agent(
        model=BedrockModel("anthropic.claude-3-haiku-20240307-v1:0"),
        cancellation_token=token,
    )

    # Cancel after receiving some chunks
    async def cancel_after_delay():
        await asyncio.sleep(1.5)
        token.cancel()

    cancel_task = asyncio.create_task(cancel_after_delay())

    events = []
    async for event in agent.stream_async(
        "Write a detailed story about a space adventure. Make it at least 500 words long."
    ):
        events.append(event)
        if event.get("result"):
            break

    await cancel_task

    # Find the result event
    result_event = next((e for e in events if e.get("result")), None)
    assert result_event is not None
    assert result_event["result"].stop_reason == "cancelled"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("AWS_REGION"), reason="AWS credentials not available")
async def test_cancellation_with_tools_bedrock():
    """Test cancellation during tool execution with Bedrock.

    This test verifies that cancellation works correctly when the agent
    is executing tools with a real Bedrock model.
    """
    from strands.models import BedrockModel

    @tool
    def slow_calculation(x: int, y: int) -> int:
        """Perform a slow calculation that takes time.

        Args:
            x: First number
            y: Second number

        Returns:
            The sum of x and y
        """
        time.sleep(2)  # Simulate slow operation
        return x + y

    @tool
    def another_calculation(a: int, b: int) -> int:
        """Another slow calculation.

        Args:
            a: First number
            b: Second number

        Returns:
            The product of a and b
        """
        time.sleep(2)  # Simulate slow operation
        return a * b

    token = CancellationToken()
    agent = Agent(
        model=BedrockModel("anthropic.claude-3-haiku-20240307-v1:0"),
        tools=[slow_calculation, another_calculation],
        cancellation_token=token,
    )

    # Cancel after 3 seconds (should be during tool execution)
    async def cancel_after_delay():
        await asyncio.sleep(3.0)
        token.cancel()

    cancel_task = asyncio.create_task(cancel_after_delay())

    result = await agent.invoke_async(
        "Please use the slow_calculation tool to add 5 and 10, then use another_calculation to multiply 3 and 7."
    )

    await cancel_task

    assert result.stop_reason == "cancelled"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("AWS_REGION"), reason="AWS credentials not available")
async def test_cancellation_from_different_thread_bedrock():
    """Test cancellation from a different thread with Bedrock.

    This test simulates a real-world scenario where cancellation is triggered
    from a different thread (e.g., a web request handler) while the agent
    is executing in another thread.
    """
    from strands.models import BedrockModel

    token = CancellationToken()
    agent = Agent(
        model=BedrockModel("anthropic.claude-3-haiku-20240307-v1:0"),
        cancellation_token=token,
    )

    # Cancel from a different thread after 2 seconds
    def cancel_from_thread():
        time.sleep(2.0)
        token.cancel()

    cancel_thread = threading.Thread(target=cancel_from_thread)
    cancel_thread.start()

    result = await agent.invoke_async(
        "Write a comprehensive guide about machine learning, "
        "covering supervised learning, unsupervised learning, and deep learning. "
        "Make it at least 800 words."
    )

    cancel_thread.join()

    assert result.stop_reason == "cancelled"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not available")
async def test_cancellation_before_model_call_anthropic():
    """Test cancellation before model call with Anthropic.

    This test verifies that when cancellation is requested before the model
    is called, the agent stops immediately without making any API calls.
    """
    from strands.models import AnthropicModel

    token = CancellationToken()
    agent = Agent(
        model=AnthropicModel("claude-3-haiku-20240307"),
        cancellation_token=token,
    )

    # Cancel immediately before invocation
    token.cancel()

    result = await agent.invoke_async("Hello, how are you?")

    assert result.stop_reason == "cancelled"
    # Should not have made any API calls, so message should be empty
    assert result.message == {}


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not available")
async def test_cancellation_idempotent_openai():
    """Test that multiple cancellations are safe with OpenAI.

    This test verifies that calling cancel() multiple times doesn't cause
    any issues with a real model provider.
    """
    from strands.models import OpenAIModel

    token = CancellationToken()
    agent = Agent(
        model=OpenAIModel("gpt-4o-mini"),
        cancellation_token=token,
    )

    # Cancel multiple times
    token.cancel()
    token.cancel()
    token.cancel()

    result = await agent.invoke_async("Tell me a short joke.")

    assert result.stop_reason == "cancelled"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("AWS_REGION"), reason="AWS credentials not available")
async def test_cancellation_shared_token_bedrock():
    """Test that multiple agents can share the same cancellation token.

    This test verifies that when multiple agents share a cancellation token,
    cancelling it affects all agents using it.
    """
    from strands.models import BedrockModel

    token = CancellationToken()

    agent1 = Agent(
        model=BedrockModel("anthropic.claude-3-haiku-20240307-v1:0"),
        cancellation_token=token,
    )

    agent2 = Agent(
        model=BedrockModel("anthropic.claude-3-haiku-20240307-v1:0"),
        cancellation_token=token,
    )

    # Cancel the shared token
    token.cancel()

    result1 = await agent1.invoke_async("Write a short poem about the ocean.")
    result2 = await agent2.invoke_async("Write a short poem about mountains.")

    assert result1.stop_reason == "cancelled"
    assert result2.stop_reason == "cancelled"


@pytest.mark.asyncio
@pytest.mark.skipif(not os.getenv("AWS_REGION"), reason="AWS credentials not available")
async def test_agent_without_cancellation_token_bedrock():
    """Test that agent works normally without cancellation token.

    This test verifies that when no cancellation token is provided,
    the agent executes normally with a real model.
    """
    from strands.models import BedrockModel

    agent = Agent(
        model=BedrockModel("anthropic.claude-3-haiku-20240307-v1:0"),
    )

    result = await agent.invoke_async("Say hello in exactly 5 words.")

    assert result.stop_reason == "end_turn"
    assert result.message["role"] == "assistant"
    assert len(result.message["content"]) > 0
