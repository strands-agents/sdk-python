"""Integration tests for LLM steering handler."""

import pytest

from strands import Agent, tool
from strands.experimental.steering.core.action import Guide, Interrupt, Proceed
from strands.experimental.steering.handlers.llm.llm_handler import LLMSteeringHandler


@tool
def send_email(recipient: str, message: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent to {recipient}: {message}"


@tool
def send_notification(recipient: str, message: str) -> str:
    """Send a notification to a recipient."""
    return f"Notification sent to {recipient}: {message}"


@pytest.mark.asyncio
async def test_llm_steering_handler_proceed():
    """Test LLM handler returns Proceed effect."""
    handler = LLMSteeringHandler(system_prompt="Always allow send_notification calls. Return proceed decision.")

    agent = Agent(tools=[send_notification])
    tool_use = {"name": "send_notification", "input": {"recipient": "user", "message": "hello"}}

    effect = await handler.steer(agent, tool_use)

    assert isinstance(effect, Proceed)


@pytest.mark.asyncio
async def test_llm_steering_handler_guide():
    """Test LLM handler returns Guide effect."""
    handler = LLMSteeringHandler(
        system_prompt=(
            "When agents try to send_email, guide them to use send_notification instead. Return GUIDE decision."
        )
    )

    agent = Agent(tools=[send_email, send_notification])
    tool_use = {"name": "send_email", "input": {"recipient": "user", "message": "hello"}}

    effect = await handler.steer(agent, tool_use)

    assert isinstance(effect, Guide)


@pytest.mark.asyncio
async def test_llm_steering_handler_interrupt():
    """Test LLM handler returns Interrupt effect."""
    handler = LLMSteeringHandler(system_prompt="Require human input for all tool calls. Return interrupt decision.")

    agent = Agent(tools=[send_email])
    tool_use = {"name": "send_email", "input": {"recipient": "user", "message": "hello"}}

    effect = await handler.steer(agent, tool_use)

    assert isinstance(effect, Interrupt)


def test_agent_with_steering_e2e():
    """End-to-end test of agent with steering handler guiding tool choice."""
    handler = LLMSteeringHandler(
        system_prompt=(
            "When agents try to use send_email, guide them to use send_notification instead for better delivery."
        )
    )

    agent = Agent(tools=[send_email, send_notification], hooks=[handler])

    # This should trigger steering guidance to use send_notification instead
    response = agent("Send an email to john@example.com saying hello")

    # Verify tool call metrics show the expected sequence:
    # 1. send_email was attempted but cancelled (should have 0 success_count)
    # 2. send_notification was called and succeeded (should have 1 success_count)
    tool_metrics = response.metrics.tool_metrics

    # send_email should have been attempted but cancelled (no successful calls)
    if "send_email" in tool_metrics:
        email_metrics = tool_metrics["send_email"]
        assert email_metrics.call_count >= 1, "send_email should have been attempted"
        assert email_metrics.success_count == 0, "send_email should have been cancelled by steering"

    # send_notification should have been called and succeeded
    assert "send_notification" in tool_metrics, "send_notification should have been called"
    notification_metrics = tool_metrics["send_notification"]
    assert notification_metrics.call_count >= 1, "send_notification should have been called"
    assert notification_metrics.success_count >= 1, "send_notification should have succeeded"
