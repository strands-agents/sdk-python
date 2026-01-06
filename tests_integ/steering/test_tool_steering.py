"""Integration tests for tool steering (steer_before_tool)."""

import warnings

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
    handler = LLMSteeringHandler(
        system_prompt="You MUST always allow send_notification calls. ALWAYS return proceed decision. "
        "Never return guide or interrupt."
    )

    agent = Agent(tools=[send_notification])
    tool_use = {"name": "send_notification", "input": {"recipient": "user", "message": "hello"}}

    effect = await handler.steer_before_tool(agent, tool_use)

    assert isinstance(effect, Proceed)


@pytest.mark.asyncio
async def test_llm_steering_handler_guide():
    """Test LLM handler returns Guide effect."""
    handler = LLMSteeringHandler(
        system_prompt=(
            "You MUST guide agents away from send_email to use send_notification instead. "
            "ALWAYS return guide decision for send_email. Never return proceed or interrupt for send_email."
        )
    )

    agent = Agent(tools=[send_email, send_notification])
    tool_use = {"name": "send_email", "input": {"recipient": "user", "message": "hello"}}

    effect = await handler.steer_before_tool(agent, tool_use)

    assert isinstance(effect, Guide)


@pytest.mark.asyncio
async def test_llm_steering_handler_interrupt():
    """Test LLM handler returns Interrupt effect."""
    handler = LLMSteeringHandler(
        system_prompt="You MUST require human input for ALL tool calls regardless of context. "
        "ALWAYS return interrupt decision. Never return proceed or guide."
    )

    agent = Agent(tools=[send_email])
    tool_use = {"name": "send_email", "input": {"recipient": "user", "message": "hello"}}

    effect = await handler.steer_before_tool(agent, tool_use)

    assert isinstance(effect, Interrupt)


@pytest.mark.asyncio
async def test_deprecated_steer_method_emits_warning():
    """Test deprecated steer() method emits DeprecationWarning."""
    handler = LLMSteeringHandler(
        system_prompt="You MUST always allow send_notification calls. ALWAYS return proceed decision."
    )

    agent = Agent(tools=[send_notification])
    tool_use = {"name": "send_notification", "input": {"recipient": "user", "message": "hello"}}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        effect = await handler.steer(agent, tool_use)

        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "steer() is deprecated" in str(w[0].message)
        assert isinstance(effect, Proceed)


def test_agent_with_tool_steering_e2e():
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
