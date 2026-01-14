"""Tests for A2AAgent class."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from a2a.types import AgentCard, Message, Part, Role, TextPart

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
    assert agent.name is None
    assert agent.description is None


def test_init_with_name_and_description():
    """Test initialization with custom name and description."""
    agent = A2AAgent(endpoint="http://localhost:8000", name="my-agent", description="My custom agent")
    assert agent.name == "my-agent"
    assert agent.description == "My custom agent"


def test_init_with_custom_timeout():
    """Test initialization with custom timeout."""
    agent = A2AAgent(endpoint="http://localhost:8000", timeout=600)
    assert agent.timeout == 600


def test_init_with_external_a2a_client_factory():
    """Test initialization with external A2A client factory."""
    external_factory = MagicMock()
    agent = A2AAgent(endpoint="http://localhost:8000", a2a_client_factory=external_factory)
    assert agent._a2a_client_factory is external_factory
    assert not agent._owns_client


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
async def test_get_agent_card_populates_name_and_description(mock_agent_card):
    """Test that agent card populates name and description if not set."""
    agent = A2AAgent(endpoint="http://localhost:8000")

    with patch("strands.agent.a2a_agent.A2ACardResolver") as mock_resolver_class:
        mock_resolver = AsyncMock()
        mock_resolver.get_agent_card = AsyncMock(return_value=mock_agent_card)
        mock_resolver_class.return_value = mock_resolver

        await agent._get_agent_card()

        assert agent.name == mock_agent_card.name
        assert agent.description == mock_agent_card.description


@pytest.mark.asyncio
async def test_get_agent_card_preserves_custom_name_and_description(mock_agent_card):
    """Test that custom name and description are not overridden by agent card."""
    agent = A2AAgent(endpoint="http://localhost:8000", name="custom-name", description="Custom description")

    with patch("strands.agent.a2a_agent.A2ACardResolver") as mock_resolver_class:
        mock_resolver = AsyncMock()
        mock_resolver.get_agent_card = AsyncMock(return_value=mock_agent_card)
        mock_resolver_class.return_value = mock_resolver

        await agent._get_agent_card()

        assert agent.name == "custom-name"
        assert agent.description == "Custom description"


@pytest.mark.asyncio
async def test_invoke_async_success(a2a_agent, mock_agent_card):
    """Test successful async invocation."""
    mock_response = Message(
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
    mock_response = Message(
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

            events = []
            async for event in a2a_agent.stream_async("Hello"):
                events.append(event)

            assert len(events) == 2
            # First event is A2A stream event
            assert events[0]["type"] == "a2a_stream"
            assert events[0]["event"] == mock_response
            # Final event is AgentResult
            assert "result" in events[1]
            assert isinstance(events[1]["result"], AgentResult)
            assert events[1]["result"].message["content"][0]["text"] == "Response"


@pytest.mark.asyncio
async def test_stream_async_no_prompt(a2a_agent):
    """Test that stream_async raises ValueError when prompt is None."""
    with pytest.raises(ValueError, match="prompt is required"):
        async for _ in a2a_agent.stream_async(None):
            pass


def test_del_cleanup_owned_client():
    """Test that __del__ cleans up owned httpx client."""
    agent = A2AAgent(endpoint="http://localhost:8000")
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()
    agent._httpx_client = mock_client
    agent._owns_client = True

    with patch("strands.agent.a2a_agent.run_async") as mock_run_async:
        agent.__del__()
        mock_run_async.assert_called_once()


def test_del_no_cleanup_external_client():
    """Test that __del__ does not clean up external client."""
    external_factory = MagicMock()
    agent = A2AAgent(endpoint="http://localhost:8000", a2a_client_factory=external_factory)
    mock_client = MagicMock()
    agent._httpx_client = mock_client

    with patch("strands.agent.a2a_agent.run_async") as mock_run_async:
        agent.__del__()
        mock_run_async.assert_not_called()


def test_del_handles_exception():
    """Test that __del__ handles exceptions gracefully."""
    agent = A2AAgent(endpoint="http://localhost:8000")
    mock_client = MagicMock()
    agent._httpx_client = mock_client
    agent._owns_client = True

    with patch("strands.agent.a2a_agent.run_async", side_effect=RuntimeError("Event loop error")):
        # Should not raise - __del__ should catch exceptions
        agent.__del__()


def test_get_httpx_client_creates_client_with_timeout():
    """Test that _get_httpx_client creates client with configured timeout."""
    agent = A2AAgent(endpoint="http://localhost:8000", timeout=120)

    client = agent._get_httpx_client()

    assert client is not None
    assert agent._httpx_client is client
    assert client.timeout.connect == 120


def test_create_default_factory_uses_streaming():
    """Test _create_default_factory creates factory with streaming enabled."""
    agent = A2AAgent(endpoint="http://localhost:8000")

    with patch("strands.agent.a2a_agent.ClientConfig") as mock_config_class:
        with patch("strands.agent.a2a_agent.ClientFactory"):
            agent._create_default_factory()

            # Verify streaming=True is passed - this is the key behavior
            call_kwargs = mock_config_class.call_args[1]
            assert call_kwargs["streaming"] is True


@pytest.mark.asyncio
async def test_get_a2a_client_uses_provided_factory(mock_agent_card):
    """Test _get_a2a_client uses provided factory instead of creating default."""
    external_factory = MagicMock()
    mock_client = MagicMock()
    external_factory.create.return_value = mock_client

    agent = A2AAgent(endpoint="http://localhost:8000", a2a_client_factory=external_factory)

    with patch.object(agent, "_get_agent_card", return_value=mock_agent_card):
        client = await agent._get_a2a_client()

        assert client is mock_client
        external_factory.create.assert_called_once_with(mock_agent_card)


def test_is_complete_event_message(a2a_agent):
    """Test _is_complete_event returns True for Message."""
    mock_message = MagicMock(spec=Message)

    assert a2a_agent._is_complete_event(mock_message) is True


def test_is_complete_event_tuple_with_none_update(a2a_agent):
    """Test _is_complete_event returns True for tuple with None update event."""
    mock_task = MagicMock()

    assert a2a_agent._is_complete_event((mock_task, None)) is True


def test_is_complete_event_artifact_last_chunk(a2a_agent):
    """Test _is_complete_event handles TaskArtifactUpdateEvent last_chunk flag."""
    from a2a.types import TaskArtifactUpdateEvent

    mock_task = MagicMock()

    # last_chunk=True -> complete
    event_complete = MagicMock(spec=TaskArtifactUpdateEvent)
    event_complete.last_chunk = True
    assert a2a_agent._is_complete_event((mock_task, event_complete)) is True

    # last_chunk=False -> not complete
    event_incomplete = MagicMock(spec=TaskArtifactUpdateEvent)
    event_incomplete.last_chunk = False
    assert a2a_agent._is_complete_event((mock_task, event_incomplete)) is False

    # last_chunk=None -> not complete
    event_none = MagicMock(spec=TaskArtifactUpdateEvent)
    event_none.last_chunk = None
    assert a2a_agent._is_complete_event((mock_task, event_none)) is False


def test_is_complete_event_status_update(a2a_agent):
    """Test _is_complete_event handles TaskStatusUpdateEvent state."""
    from a2a.types import TaskState, TaskStatusUpdateEvent

    mock_task = MagicMock()

    # completed state -> complete
    event_completed = MagicMock(spec=TaskStatusUpdateEvent)
    event_completed.status = MagicMock()
    event_completed.status.state = TaskState.completed
    assert a2a_agent._is_complete_event((mock_task, event_completed)) is True

    # working state -> not complete
    event_working = MagicMock(spec=TaskStatusUpdateEvent)
    event_working.status = MagicMock()
    event_working.status.state = TaskState.working
    assert a2a_agent._is_complete_event((mock_task, event_working)) is False

    # no status -> not complete
    event_no_status = MagicMock(spec=TaskStatusUpdateEvent)
    event_no_status.status = None
    assert a2a_agent._is_complete_event((mock_task, event_no_status)) is False


def test_is_complete_event_unknown_type(a2a_agent):
    """Test _is_complete_event returns False for unknown event types."""
    assert a2a_agent._is_complete_event("unknown") is False


@pytest.mark.asyncio
async def test_stream_async_tracks_complete_events(a2a_agent, mock_agent_card):
    """Test stream_async uses last complete event for final result."""
    from a2a.types import TaskState, TaskStatusUpdateEvent

    mock_task = MagicMock()
    mock_task.artifacts = None

    # First event: incomplete
    incomplete_event = MagicMock(spec=TaskStatusUpdateEvent)
    incomplete_event.status = MagicMock()
    incomplete_event.status.state = TaskState.working
    incomplete_event.status.message = None

    # Second event: complete
    complete_event = MagicMock(spec=TaskStatusUpdateEvent)
    complete_event.status = MagicMock()
    complete_event.status.state = TaskState.completed
    complete_event.status.message = MagicMock()
    complete_event.status.message.parts = []

    async def mock_send_message(*args, **kwargs):
        yield (mock_task, incomplete_event)
        yield (mock_task, complete_event)

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

            # Should have 2 stream events + 1 result event
            assert len(events) == 3
            assert "result" in events[2]


@pytest.mark.asyncio
async def test_stream_async_falls_back_to_last_event(a2a_agent, mock_agent_card):
    """Test stream_async falls back to last event when no complete event."""
    from a2a.types import TaskState, TaskStatusUpdateEvent

    mock_task = MagicMock()
    mock_task.artifacts = None

    incomplete_event = MagicMock(spec=TaskStatusUpdateEvent)
    incomplete_event.status = MagicMock()
    incomplete_event.status.state = TaskState.working
    incomplete_event.status.message = None

    async def mock_send_message(*args, **kwargs):
        yield (mock_task, incomplete_event)

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

            # Should have 1 stream event + 1 result event (falls back to last)
            assert len(events) == 2
            assert "result" in events[1]
