import os
import subprocess
import time

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory

from strands.agent.a2a_agent import A2AAgent


@pytest.fixture
def a2a_server():
    """Start A2A server as subprocess fixture."""
    server_path = os.path.join(os.path.dirname(__file__), "a2a_server.py")
    process = subprocess.Popen(["python", server_path])
    time.sleep(5)  # Wait for A2A server to start

    yield "http://localhost:9000"

    # Cleanup
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def test_a2a_agent_invoke_sync(a2a_server):
    """Test synchronous invocation via __call__."""
    a2a_agent = A2AAgent(endpoint=a2a_server)
    result = a2a_agent("Hello there!")
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_a2a_agent_invoke_async(a2a_server):
    """Test async invocation."""
    a2a_agent = A2AAgent(endpoint=a2a_server)
    result = await a2a_agent.invoke_async("Hello there!")
    assert result.stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_a2a_agent_stream_async(a2a_server):
    """Test async streaming."""
    a2a_agent = A2AAgent(endpoint=a2a_server)

    events = []
    async for event in a2a_agent.stream_async("Hello there!"):
        events.append(event)

    # Should have at least one A2A stream event and one final result event
    assert len(events) >= 2
    assert events[0]["type"] == "a2a_stream"
    assert "result" in events[-1]
    assert events[-1]["result"].stop_reason == "end_turn"


@pytest.mark.asyncio
async def test_a2a_agent_context_manager(a2a_server):
    """Test async context manager for proper cleanup."""
    async with A2AAgent(endpoint=a2a_server) as agent:
        result = await agent.invoke_async("Hello there!")
        assert result.stop_reason == "end_turn"
    # Cleanup should happen automatically


@pytest.mark.asyncio
async def test_a2a_agent_with_streaming_client_config(a2a_server):
    """Test with streaming=True client configuration."""
    httpx_client = httpx.AsyncClient(timeout=300)
    config = ClientConfig(httpx_client=httpx_client, streaming=True)
    factory = ClientFactory(config)

    try:
        a2a_agent = A2AAgent(endpoint=a2a_server, a2a_client_factory=factory)

        events = []
        async for event in a2a_agent.stream_async("Hello there!"):
            events.append(event)

        assert len(events) >= 2
        assert events[0]["type"] == "a2a_stream"
        assert "result" in events[-1]
        assert events[-1]["result"].stop_reason == "end_turn"
    finally:
        await httpx_client.aclose()
