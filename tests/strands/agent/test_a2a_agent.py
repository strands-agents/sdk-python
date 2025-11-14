"""Tests for A2AAgent class."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from a2a.types import AgentCard, Part, Role, TextPart
from a2a.types import Message as A2AMessage

from strands.agent.a2a_agent import A2AAgent
from strands.agent.agent_result import AgentResult


@pytest.fixture
def mock_agent_card():
    """Mock AgentCard for testing."""
    return AgentCard(
        name="test-agent",
        description="Test agent",
        url="http://localhost:8000",
        version="1.0.0",
        capabilities={},
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[],
    )


@pytest.fixture
def a2a_agent():
    """Create A2AAgent instance for testing."""
    return A2AAgent(endpoint="http://localhost:8000")


def test_init_with_defaults():
    """Test initialization with default parameters."""
    agent = A2AAgent(endpoint="http://localhost:8000")
    assert agent.endpoint == "http://localhost:8000"
    assert agent.timeout == 300
    assert agent._agent_card is None


def test_init_with_custom_timeout():
    """Test initialization with custom timeout."""
    agent = A2AAgent(endpoint="http://localhost:8000", timeout=600)
    assert agent.timeout == 600
    assert agent._httpx_client_args["timeout"] == 600


def test_init_with_httpx_client_args():
    """Test initialization with custom httpx client arguments."""
    agent = A2AAgent(
        endpoint="http://localhost:8000",
        httpx_client_args={"headers": {"Authorization": "Bearer token"}},
    )
    assert "headers" in agent._httpx_client_args
    assert agent._httpx_client_args["headers"]["Authorization"] == "Bearer token"


@pytest.mark.asyncio
async def test_get_agent_card(a2a_agent, mock_agent_card):
    """Test agent card discovery."""
    with patch("strands.agent.a2a_agent.A2ACardResolver") as mock_resolver_class:
        mock_resolver = AsyncMock()
        mock_resolver.get_agent_card = AsyncMock(return_value=mock_agent_card)
        mock_resolver_class.return_value = mock_resolver

        card = await a2a_agent._get_agent_card()

        assert card == mock_agent_card
        assert a2a_agent._agent_card == mock_agent_card


@pytest.mark.asyncio
async def test_get_agent_card_cached(a2a_agent, mock_agent_card):
    """Test that agent card is cached after first discovery."""
    a2a_agent._agent_card = mock_agent_card

    card = await a2a_agent._get_agent_card()

    assert card == mock_agent_card


@pytest.mark.asyncio
async def test_invoke_async_success(a2a_agent, mock_agent_card):
    """Test successful async invocation."""
    mock_response = A2AMessage(
        message_id=uuid4().hex,
        role=Role.agent,
        parts=[Part(TextPart(kind="text", text="Response"))],
    )

    async def mock_send_message(*args, **kwargs):
        yield mock_response

    with patch.object(a2a_agent, "_get_agent_card", return_value=mock_agent_card):
        with patch("strands.agent.a2a_agent.ClientFactory") as mock_factory_class:
            mock_client = AsyncMock()
            mock_client.send_message = mock_send_message
            mock_factory = MagicMock()
            mock_factory.create.return_value = mock_client
            mock_factory_class.return_value = mock_factory

            result = await a2a_agent.invoke_async("Hello")

            assert isinstance(result, AgentResult)
            assert result.message["content"][0]["text"] == "Response"


@pytest.mark.asyncio
async def test_invoke_async_no_prompt(a2a_agent):
    """Test that invoke_async raises ValueError when prompt is None."""
    with pytest.raises(ValueError, match="prompt is required"):
        await a2a_agent.invoke_async(None)


@pytest.mark.asyncio
async def test_invoke_async_no_response(a2a_agent, mock_agent_card):
    """Test that invoke_async raises RuntimeError when no response received."""

    async def mock_send_message(*args, **kwargs):
        return
        yield  # Make it an async generator

    with patch.object(a2a_agent, "_get_agent_card", return_value=mock_agent_card):
        with patch("strands.agent.a2a_agent.ClientFactory") as mock_factory_class:
            mock_client = AsyncMock()
            mock_client.send_message = mock_send_message
            mock_factory = MagicMock()
            mock_factory.create.return_value = mock_client
            mock_factory_class.return_value = mock_factory

            with pytest.raises(RuntimeError, match="No response received"):
                await a2a_agent.invoke_async("Hello")


def test_call_sync(a2a_agent):
    """Test synchronous call method."""
    mock_result = AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": "Response"}]},
        metrics=MagicMock(),
        state={},
    )

    with patch("strands.agent.a2a_agent.run_async") as mock_run_async:
        mock_run_async.return_value = mock_result

        result = a2a_agent("Hello")

        assert result == mock_result
        mock_run_async.assert_called_once()


@pytest.mark.asyncio
async def test_stream_async_success(a2a_agent, mock_agent_card):
    """Test successful async streaming."""
    mock_event1 = MagicMock()
    mock_event2 = MagicMock()

    async def mock_send_message(*args, **kwargs):
        yield mock_event1
        yield mock_event2

    with patch.object(a2a_agent, "_get_agent_card", return_value=mock_agent_card):
        with patch("strands.agent.a2a_agent.ClientFactory") as mock_factory_class:
            mock_client = AsyncMock()
            mock_client.send_message = mock_send_message
            mock_factory = MagicMock()
            mock_factory.create.return_value = mock_client
            mock_factory_class.return_value = mock_factory

            events = []
            async for event in a2a_agent.stream_async("Hello"):
                events.append(event)

            assert len(events) == 2
            assert events[0]["a2a_event"] == mock_event1
            assert events[1]["a2a_event"] == mock_event2


@pytest.mark.asyncio
async def test_stream_async_no_prompt(a2a_agent):
    """Test that stream_async raises ValueError when prompt is None."""
    with pytest.raises(ValueError, match="prompt is required"):
        async for _ in a2a_agent.stream_async(None):
            pass
