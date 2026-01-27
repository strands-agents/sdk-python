"""Integration tests for FallbackModel with real model providers."""

import os

import pytest

from strands import Agent
from strands.models import BedrockModel
from strands.models.anthropic import AnthropicModel
from strands.models.fallback import FallbackModel
from strands.models.openai import OpenAIModel
from tests_integ.models import providers


class TestFallbackModelIntegration:
    """Integration tests for FallbackModel with real model instances."""

    @providers.bedrock.mark
    @pytest.mark.asyncio
    async def test_same_provider_fallback_bedrock(self):
        """Test FallbackModel with two BedrockModel instances."""
        # Use different model IDs - opus as primary, haiku as fallback
        primary = BedrockModel(model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0", region_name="us-west-2")
        fallback = BedrockModel(model_id="us.anthropic.claude-3-haiku-20240307-v1:0", region_name="us-west-2")

        fallback_model = FallbackModel(
            primary=primary,
            fallback=fallback,
            circuit_failure_threshold=1,  # Open circuit quickly for testing
            circuit_time_window=60.0,
            circuit_cooldown_seconds=5,
        )

        # Test successful primary model usage
        messages = [{"role": "user", "content": [{"text": "Say 'Hello from primary model'"}]}]

        events = []
        async for event in fallback_model.stream(messages=messages):
            events.append(event)

        # Should have received events
        assert len(events) > 0

        # Check that primary was used (fallback_count should be 0)
        stats = fallback_model.get_stats()
        assert stats["fallback_count"] == 0
        assert not stats["using_fallback"]

    @pytest.mark.skipif(
        "OPENAI_API_KEY" not in os.environ or "AWS_ACCESS_KEY_ID" not in os.environ,
        reason="Both OPENAI_API_KEY and AWS credentials required for cross-provider test",
    )
    @pytest.mark.asyncio
    async def test_cross_provider_fallback_openai_bedrock(self):
        """Test FallbackModel with OpenAI primary and Bedrock fallback."""
        primary = OpenAIModel(
            model_id="gpt-4o",
            client_args={
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        )
        fallback = BedrockModel(model_id="us.anthropic.claude-3-haiku-20240307-v1:0", region_name="us-west-2")

        fallback_model = FallbackModel(
            primary=primary,
            fallback=fallback,
            circuit_failure_threshold=2,
            circuit_time_window=60.0,
            circuit_cooldown_seconds=10,
        )

        # Test successful cross-provider usage
        messages = [{"role": "user", "content": [{"text": "Respond with exactly: 'Cross-provider test successful'"}]}]

        events = []
        async for event in fallback_model.stream(messages=messages):
            events.append(event)

        # Should have received events
        assert len(events) > 0

        # Verify we can get configuration from both models
        config = fallback_model.get_config()
        assert "primary_config" in config
        assert "fallback_model_config" in config
        assert "fallback_config" in config
        assert "stats" in config

    @providers.bedrock.mark
    @pytest.mark.asyncio
    async def test_agent_integration_with_fallback(self):
        """Test FallbackModel used with Agent class."""
        # Create fallback model with two Bedrock instances
        primary = BedrockModel(model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0", region_name="us-west-2")
        fallback = BedrockModel(model_id="us.anthropic.claude-3-haiku-20240307-v1:0", region_name="us-west-2")

        fallback_model = FallbackModel(primary=primary, fallback=fallback, track_stats=True)

        # Create agent with fallback model
        agent = Agent(model=fallback_model, system_prompt="You are a helpful assistant. Keep responses brief.")

        # Send test message
        response = await agent.invoke_async("What is 2 + 2?")

        # Assert response received
        assert response is not None
        assert response.message is not None
        assert len(response.message["content"]) > 0
        assert response.message["content"][0]["text"] is not None

        # Check that the fallback model was used successfully
        stats = fallback_model.get_stats()
        assert isinstance(stats, dict)
        assert "fallback_count" in stats
        assert "primary_failures" in stats

    @pytest.mark.skipif(
        "ANTHROPIC_API_KEY" not in os.environ, reason="ANTHROPIC_API_KEY required for Anthropic provider test"
    )
    @pytest.mark.asyncio
    async def test_cross_provider_anthropic_bedrock(self):
        """Test FallbackModel with Anthropic primary and Bedrock fallback."""
        primary = AnthropicModel(
            client_args={
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            },
            model_id="claude-3-7-sonnet-20250219",
            max_tokens=512,
        )
        fallback = BedrockModel(model_id="us.anthropic.claude-3-haiku-20240307-v1:0", region_name="us-west-2")

        fallback_model = FallbackModel(primary=primary, fallback=fallback)

        # Test structured output
        from pydantic import BaseModel

        class TestResponse(BaseModel):
            message: str
            number: int

        messages = [{"role": "user", "content": [{"text": "Return a message 'test' and number 42"}]}]

        events = []
        async for event in fallback_model.structured_output(output_model=TestResponse, prompt=messages):
            events.append(event)

        # Should have received events
        assert len(events) > 0

        # Check final event has the structured output
        final_event = events[-1]
        if "output" in final_event:
            output = final_event["output"]
            assert hasattr(output, "message")
            assert hasattr(output, "number")

    @providers.bedrock.mark
    @pytest.mark.asyncio
    async def test_fallback_statistics_tracking(self):
        """Test that statistics are properly tracked during integration tests."""
        primary = BedrockModel(model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0", region_name="us-west-2")
        fallback = BedrockModel(model_id="us.anthropic.claude-3-haiku-20240307-v1:0", region_name="us-west-2")

        fallback_model = FallbackModel(primary=primary, fallback=fallback, track_stats=True)

        # Make a successful request
        messages = [{"role": "user", "content": [{"text": "Say hello"}]}]

        events = []
        async for event in fallback_model.stream(messages=messages):
            events.append(event)

        # Check statistics
        stats = fallback_model.get_stats()
        assert stats["fallback_count"] == 0  # No fallback should have occurred
        assert stats["primary_failures"] == 0  # No failures
        assert not stats["using_fallback"]  # Not using fallback
        assert not stats["circuit_open"]  # Circuit should be closed

        # Test configuration retrieval
        config = fallback_model.get_config()
        assert config["stats"] is not None
        assert config["fallback_config"]["track_stats"] is True

        # Test stats reset
        fallback_model.reset_stats()
        reset_stats = fallback_model.get_stats()
        assert reset_stats["fallback_count"] == 0
        assert reset_stats["primary_failures"] == 0

    @providers.bedrock.mark
    @pytest.mark.asyncio
    async def test_tool_calling_with_fallback_model(self):
        """Test that tool_specs and tool_choice parameters work with FallbackModel."""
        # Create fallback model with two Bedrock instances
        primary = BedrockModel(model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0", region_name="us-west-2")
        fallback = BedrockModel(model_id="us.anthropic.claude-3-haiku-20240307-v1:0", region_name="us-west-2")

        fallback_model = FallbackModel(primary=primary, fallback=fallback, track_stats=True)

        # Define a simple tool spec
        tool_specs = [
            {
                "name": "get_weather",
                "description": "Get weather information for a location",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "The location to get weather for"}
                        },
                        "required": ["location"],
                    }
                },
            }
        ]

        tool_choice = {"auto": {}}

        # Test message that might trigger tool use
        messages = [{"role": "user", "content": [{"text": "What's the weather in Seattle?"}]}]

        # Stream with tool parameters
        events = []
        async for event in fallback_model.stream(messages=messages, tool_specs=tool_specs, tool_choice=tool_choice):
            events.append(event)

        # Should have received events
        assert len(events) > 0

        # Verify primary was used (no fallback)
        stats = fallback_model.get_stats()
        assert stats["fallback_count"] == 0
        assert not stats["using_fallback"]

    @providers.bedrock.mark
    @pytest.mark.asyncio
    async def test_tool_calling_with_agent_and_fallback_model(self):
        """Test that FallbackModel works with Agent class when tools are provided."""
        # Create fallback model with two Bedrock instances
        primary = BedrockModel(model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0", region_name="us-west-2")
        fallback = BedrockModel(model_id="us.anthropic.claude-3-haiku-20240307-v1:0", region_name="us-west-2")

        fallback_model = FallbackModel(primary=primary, fallback=fallback, track_stats=True)

        # Create a simple tool using the strands tool decorator
        from strands import tool

        @tool
        def get_current_time(timezone: str = "UTC") -> dict:
            """Get the current time in a specific timezone."""
            return {"status": "success", "content": [{"text": f"Current time in {timezone}: 12:00 PM"}]}

        # Create agent with fallback model and tool
        agent = Agent(model=fallback_model, tools=[get_current_time], system_prompt="You are a helpful assistant.")

        # Send test message
        response = await agent.invoke_async("What time is it?")

        # Assert response received
        assert response is not None
        assert response.message is not None
        assert len(response.message["content"]) > 0

        # Verify primary was used (no fallback)
        stats = fallback_model.get_stats()
        assert stats["fallback_count"] == 0
        assert not stats["using_fallback"]
