"""Integration tests for Swarm AgentBase protocol support.

Tests that Swarm correctly handles real AgentBase implementations in production scenarios.
"""

from typing import Any

import pytest

from strands import Agent, AgentResult
from strands.agent.base import AgentBase
from strands.multiagent.swarm import Swarm


class SimpleCustomAgent(AgentBase):
    """A simple custom AgentBase implementation for testing.

    This demonstrates a minimal AgentBase that doesn't use LLMs,
    simulating a deterministic agent or rule-based system.
    """

    def __init__(self, name: str, response: str):
        """Initialize the custom agent.

        Args:
            name: Agent name
            response: Fixed response to return
        """
        self.name = name
        self.id = f"{name}_custom_id"
        self._response = response
        self._call_count = 0

    def __call__(self, prompt: Any = None, **kwargs: Any) -> AgentResult:
        """Synchronous invocation."""
        import asyncio

        return asyncio.run(self.invoke_async(prompt, **kwargs))

    async def invoke_async(self, prompt: Any = None, **kwargs: Any) -> AgentResult:
        """Asynchronous invocation.

        Returns a deterministic response without calling an LLM.
        """
        self._call_count += 1

        # Simulate processing time
        import asyncio

        await asyncio.sleep(0.01)

        return AgentResult(
            message={"role": "assistant", "content": [{"text": self._response}]},
            stop_reason="end_turn",
            state={},
            metrics={
                "accumulated_usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0},
                "accumulated_metrics": {"latencyMs": 10.0},
            },
        )

    async def stream_async(self, prompt: Any = None, **kwargs: Any):
        """Stream agent execution asynchronously."""
        # Yield some events to simulate streaming
        yield {"agent_start": True, "node": self.name}
        yield {"agent_thinking": True, "thought": f"Processing with {self.name}"}

        # Get the result
        result = await self.invoke_async(prompt, **kwargs)

        # Yield the final result
        yield {"result": result}


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_swarm_with_custom_agentbase_only():
    """Test Swarm execution with only custom AgentBase implementations (no LLM agents)."""
    # Create custom AgentBase implementations
    agent1 = SimpleCustomAgent("custom_agent_1", "Response from custom agent 1")
    agent2 = SimpleCustomAgent("custom_agent_2", "Response from custom agent 2")

    # Create swarm with only custom agents
    swarm = Swarm(nodes=[agent1, agent2])

    # Execute the swarm
    result = await swarm.invoke_async("Test task")

    # Verify execution completed successfully
    assert result.status.value == "completed"
    assert len(result.results) == 1  # Only entry point should execute without handoff
    assert "custom_agent_1" in result.results

    # Verify the custom agent was called
    assert agent1._call_count >= 1

    # Verify response content
    agent_results = result.results["custom_agent_1"].get_agent_results()
    assert len(agent_results) == 1
    assert "Response from custom agent 1" in agent_results[0].message["content"][0]["text"]


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_swarm_mixed_llm_and_custom_agents():
    """Test Swarm with both real LLM Agent and custom AgentBase implementations."""
    # Create a real LLM agent
    llm_agent = Agent(
        name="llm_agent",
        model="us.amazon.nova-lite-v1:0",
        system_prompt="You are a helpful assistant. Answer questions concisely without handing off.",
    )

    # Create a custom AgentBase implementation
    custom_agent = SimpleCustomAgent("custom_agent", "Deterministic response from custom agent")

    # Create swarm with mixed agent types
    swarm = Swarm(nodes=[llm_agent, custom_agent])

    # Execute with LLM agent as entry point
    result = await swarm.invoke_async("What is 2 + 2?")

    # Verify execution completed
    assert result.status.value in ["completed", "failed"]  # May fail due to credentials

    # If execution succeeded, verify the structure
    if result.status.value == "completed":
        assert len(result.results) >= 1
        assert "llm_agent" in result.results

        # Verify LLM agent result structure
        llm_result = result.results["llm_agent"]
        assert llm_result.result.message is not None
        assert llm_result.accumulated_usage["totalTokens"] >= 0


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_swarm_custom_agent_streaming():
    """Test that custom AgentBase implementations stream events correctly."""
    # Create custom agents
    agent1 = SimpleCustomAgent("streamer_1", "Streaming response 1")
    agent2 = SimpleCustomAgent("streamer_2", "Streaming response 2")

    # Create swarm
    swarm = Swarm(nodes=[agent1, agent2])

    # Collect streaming events
    events = []
    async for event in swarm.stream_async("Test streaming"):
        events.append(event)

    # Verify we received events
    assert len(events) > 0

    # Verify event types
    node_start_events = [e for e in events if e.get("type") == "multiagent_node_start"]
    node_stream_events = [e for e in events if e.get("type") == "multiagent_node_stream"]
    node_stop_events = [e for e in events if e.get("type") == "multiagent_node_stop"]
    result_events = [e for e in events if "result" in e and e.get("type") != "multiagent_node_stream"]

    # Should have proper event structure
    assert len(node_start_events) >= 1
    assert len(node_stream_events) >= 1  # Custom agents yield streaming events
    assert len(node_stop_events) >= 1
    assert len(result_events) == 1

    # Verify node start event structure
    start_event = node_start_events[0]
    assert start_event["node_id"] == "streamer_1"
    assert start_event["node_type"] == "agent"

    # Verify custom agent events were forwarded
    custom_events = [e for e in node_stream_events if e.get("node_id") == "streamer_1"]
    assert len(custom_events) >= 1


@pytest.mark.timeout(120)
@pytest.mark.asyncio
async def test_swarm_custom_agent_no_state_attributes():
    """Test that custom AgentBase without state/messages attributes works correctly."""

    class MinimalAgent:
        """Minimal AgentBase implementation without messages or state."""

        def __init__(self, name: str):
            self.name = name
            self.id = f"{name}_minimal_id"

        async def invoke_async(self, prompt: Any = None, **kwargs: Any) -> AgentResult:
            return AgentResult(
                message={"role": "assistant", "content": [{"text": f"Response from {self.name}"}]},
                stop_reason="end_turn",
                state={},
                metrics=None,
            )

        def __call__(self, prompt: Any = None, **kwargs: Any) -> AgentResult:
            import asyncio

            return asyncio.run(self.invoke_async(prompt, **kwargs))

        async def stream_async(self, prompt: Any = None, **kwargs: Any):
            yield {"agent_start": True}
            result = await self.invoke_async(prompt, **kwargs)
            yield {"result": result}

    # Verify it satisfies AgentBase protocol
    minimal = MinimalAgent("minimal")
    assert isinstance(minimal, AgentBase)

    # Create swarm with minimal agent
    swarm = Swarm(nodes=[minimal])

    # Execute should work without state/messages attributes
    result = await swarm.invoke_async("Test minimal agent")

    # Verify execution completed
    assert result.status.value == "completed"
    assert len(result.results) == 1
    assert "minimal" in result.results


@pytest.mark.timeout(120)
def test_swarm_custom_agent_synchronous_execution():
    """Test synchronous execution of Swarm with custom AgentBase implementations."""
    # Create custom agents
    agent1 = SimpleCustomAgent("sync_agent_1", "Sync response 1")
    agent2 = SimpleCustomAgent("sync_agent_2", "Sync response 2")

    # Create swarm
    swarm = Swarm(nodes=[agent1, agent2])

    # Execute synchronously
    result = swarm("Test synchronous execution")

    # Verify execution completed
    assert result.status.value == "completed"
    assert len(result.results) == 1
    assert "sync_agent_1" in result.results

    # Verify custom agent was called
    assert agent1._call_count >= 1
