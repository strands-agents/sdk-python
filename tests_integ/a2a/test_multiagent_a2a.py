import os
import subprocess
import time

import httpx
import pytest
from a2a.client import ClientConfig, ClientFactory
from strands.agent.a2a_agent import A2AAgent


def _wait_for_server(endpoint: str, timeout: float = 30.0, poll_interval: float = 0.5) -> None:
    """Wait for the A2A server to be ready by polling the agent card endpoint.

    Args:
        endpoint: The server endpoint URL.
        timeout: Maximum time to wait in seconds.
        poll_interval: Time between polling attempts in seconds.

    Raises:
        RuntimeError: If server does not become ready within timeout.
    """
    agent_card_url = f"{endpoint}/.well-known/agent.json"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = httpx.get(agent_card_url, timeout=poll_interval)
            if response.status_code == 200:
                return  # Server is ready
        except (httpx.ConnectError, httpx.TimeoutException):
            pass  # Server not ready yet
        time.sleep(poll_interval)

    raise RuntimeError(f"A2A server at {endpoint} did not become ready within {timeout} seconds")


@pytest.fixture
def a2a_server():
    """Start A2A server as subprocess fixture."""
    server_path = os.path.join(os.path.dirname(__file__), "a2a_server.py")
    process = subprocess.Popen(["python", server_path])

    endpoint = "http://localhost:9000"
    try:
        _wait_for_server(endpoint)
    except RuntimeError:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise

    yield endpoint

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
async def test_a2a_agent_with_non_streaming_client_config(a2a_server):
    """Test with streaming=False client configuration (non-default)."""
    httpx_client = httpx.AsyncClient(timeout=300)
    config = ClientConfig(httpx_client=httpx_client, streaming=False)
    factory = ClientFactory(config)

    try:
        a2a_agent = A2AAgent(endpoint=a2a_server, a2a_client_factory=factory)
        result = await a2a_agent.invoke_async("Hello there!")
        assert result.stop_reason == "end_turn"
    finally:
        await httpx_client.aclose()
