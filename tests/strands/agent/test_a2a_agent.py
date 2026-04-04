"""Tests for A2AAgent class."""

import warnings
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from a2a.client import ClientConfig
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


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.AsyncClient that works as async context manager."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


# --- Initialization tests ---


def test_init_with_defaults():
    """Test initialization with default parameters."""
    agent = A2AAgent(endpoint="http://localhost:8000")
    assert agent.endpoint == "http://localhost:8000"
    assert agent.timeout == 300
    assert agent._agent_card is None
    assert agent._a2a_client is None
    assert agent._client_config is None
    assert agent._a2a_client_factory is None
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


def test_init_with_client_config():
    """Test initialization with client_config."""
    config = ClientConfig()
    agent = A2AAgent(endpoint="http://localhost:8000", client_config=config)
    assert agent._client_config is config
    assert agent._a2a_client_factory is None


def test_init_with_factory_emits_deprecation_warning():
    """Test that passing a2a_client_factory emits a DeprecationWarning."""
    factory = MagicMock()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        agent = A2AAgent(endpoint="http://localhost:8000", a2a_client_factory=factory)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "a2a_client_factory is deprecated" in str(w[0].message)
    assert agent._a2a_client_factory is factory


# --- Card resolution tests ---


@pytest.mark.asyncio
async def test_get_agent_card_no_config(a2a_agent, mock_agent_card, mock_httpx_client):
    """Test agent card discovery without config uses transient httpx client."""
    with patch("strands.agent.a2a_agent.httpx.AsyncClient", return_value=mock_httpx_client):
        with patch("strands.agent.a2a_agent.A2ACardResolver") as mock_resolver_class:
            mock_resolver = AsyncMock()
            mock_resolver.get_agent_card = AsyncMock(return_value=mock_agent_card)
            mock_resolver_class.return_value = mock_resolver

            card = await a2a_agent.get_agent_card()

            assert card == mock_agent_card
            assert a2a_agent._agent_card == mock_agent_card


@pytest.mark.asyncio
async def test_get_agent_card_with_client_config():
    """Test agent card discovery with client_config uses its httpx client."""
    mock_httpx = MagicMock()
    config = ClientConfig(httpx_client=mock_httpx)
    agent = A2AAgent(endpoint="http://localhost:8000", client_config=config)

    mock_card = MagicMock(spec=AgentCard)
    mock_card.name = "test"
    mock_card.description = "desc"

    with patch("strands.agent.a2a_agent.A2ACardResolver") as mock_resolver_class:
        mock_resolver = AsyncMock()
        mock_resolver.get_agent_card = AsyncMock(return_value=mock_card)
        mock_resolver_class.return_value = mock_resolver

        card = await agent.get_agent_card()

        # Should use the config's httpx client, not create a new one
        mock_resolver_class.assert_called_once_with(httpx_client=mock_httpx, base_url="http://localhost:8000")
        assert card == mock_card


@pytest.mark.asyncio
async def test_get_agent_card_with_factory_uses_factory_config(mock_agent_card):
    """Test agent card discovery with deprecated factory extracts its config for auth."""
    mock_httpx = MagicMock()
    mock_config = MagicMock(spec=ClientConfig)
    mock_config.httpx_client = mock_httpx

    mock_factory = MagicMock()
    mock_factory._config = mock_config

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        agent = A2AAgent(endpoint="http://localhost:8000", a2a_client_factory=mock_factory)

    with patch("strands.agent.a2a_agent.A2ACardResolver") as mock_resolver_class:
        mock_resolver = AsyncMock()
        mock_resolver.get_agent_card = AsyncMock(return_value=mock_agent_card)
        mock_resolver_class.return_value = mock_resolver

        card = await agent.get_agent_card()

        # Should use the factory's httpx client for card resolution
        mock_resolver_class.assert_called_once_with(httpx_client=mock_httpx, base_url="http://localhost:8000")
        assert card == mock_agent_card


@pytest.mark.asyncio
async def test_get_agent_card_client_config_takes_precedence_over_factory(mock_agent_card):
    """Test that client_config is preferred over factory config for card resolution."""
    explicit_httpx = MagicMock()
    explicit_config = ClientConfig(httpx_client=explicit_httpx)

    factory_httpx = MagicMock()
    factory_config = MagicMock(spec=ClientConfig)
    factory_config.httpx_client = factory_httpx
    mock_factory = MagicMock()
    mock_factory._config = factory_config

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        agent = A2AAgent(
            endpoint="http://localhost:8000",
            client_config=explicit_config,
            a2a_client_factory=mock_factory,
        )

    with patch("strands.agent.a2a_agent.A2ACardResolver") as mock_resolver_class:
        mock_resolver = AsyncMock()
        mock_resolver.get_agent_card = AsyncMock(return_value=mock_agent_card)
        mock_resolver_class.return_value = mock_resolver

        await agent.get_agent_card()

        # Should use explicit client_config's httpx, not factory's
        mock_resolver_class.assert_called_once_with(httpx_client=explicit_httpx, base_url="http://localhost:8000")


@pytest.mark.asyncio
async def test_get_agent_card_factory_without_config_attr(mock_agent_card, mock_httpx_client):
    """Test fallback when factory has no _config attribute."""
    mock_factory = MagicMock(spec=[])  # No _config attribute

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        agent = A2AAgent(endpoint="http://localhost:8000", a2a_client_factory=mock_factory)

    with patch("strands.agent.a2a_agent.httpx.AsyncClient", return_value=mock_httpx_client):
        with patch("strands.agent.a2a_agent.A2ACardResolver") as mock_resolver_class:
            mock_resolver = AsyncMock()
            mock_resolver.get_agent_card = AsyncMock(return_value=mock_agent_card)
            mock_resolver_class.return_value = mock_resolver

            card = await agent.get_agent_card()

            # Should fall back to transient httpx client
            assert card == mock_agent_card


@pytest.mark.asyncio
async def test_get_agent_card_cached(a2a_agent, mock_agent_card):
    """Test that agent card is cached after first discovery."""
    a2a_agent._agent_card = mock_agent_card

    card = await a2a_agent.get_agent_card()

    assert card == mock_agent_card


@pytest.mark.asyncio
async def test_get_agent_card_populates_name_and_description(mock_agent_card, mock_httpx_client):
    """Test that agent card populates name and description if not set."""
    agent = A2AAgent(endpoint="http://localhost:8000")

    with patch("strands.agent.a2a_agent.httpx.AsyncClient", return_value=mock_httpx_client):
        with patch("strands.agent.a2a_agent.A2ACardResolver") as mock_resolver_class:
            mock_resolver = AsyncMock()
            mock_resolver.get_agent_card = AsyncMock(return_value=mock_agent_card)
            mock_resolver_class.return_value = mock_resolver

            await agent.get_agent_card()

            assert agent.name == mock_agent_card.name
            assert agent.description == mock_agent_card.description


@pytest.mark.asyncio
async def test_get_agent_card_preserves_custom_name_and_description(mock_agent_card, mock_httpx_client):
    """Test that custom name and description are not overridden by agent card."""
    agent = A2AAgent(endpoint="http://localhost:8000", name="custom-name", description="Custom description")

    with patch("strands.agent.a2a_agent.httpx.AsyncClient", return_value=mock_httpx_client):
        with patch("strands.agent.a2a_agent.A2ACardResolver") as mock_resolver_class:
            mock_resolver = AsyncMock()
            mock_resolver.get_agent_card = AsyncMock(return_value=mock_agent_card)
            mock_resolver_class.return_value = mock_resolver

            await agent.get_agent_card()

            assert agent.name == "custom-name"
            assert agent.description == "Custom description"


# --- Client creation tests ---


@pytest.mark.asyncio
async def test_get_or_create_client_with_client_config(mock_agent_card):
    """Test _get_or_create_client with client_config uses ClientFactory.connect() and caches."""
    config = ClientConfig()
    agent = A2AAgent(endpoint="http://localhost:8000", client_config=config)

    mock_client = AsyncMock()

    with patch("strands.agent.a2a_agent.ClientFactory") as mock_factory_class:
        mock_factory_class.connect = AsyncMock(return_value=mock_client)

        client1 = await agent._get_or_create_client()
        client2 = await agent._get_or_create_client()

        # Should connect once and cache
        mock_factory_class.connect.assert_called_once_with("http://localhost:8000", client_config=config)
        assert client1 is client2
        assert client1 is mock_client


@pytest.mark.asyncio
async def test_get_or_create_client_with_factory_uses_factory_create(mock_agent_card):
    """Test _get_or_create_client with deprecated factory uses factory.create()."""
    mock_factory = MagicMock()
    mock_factory._config = MagicMock(spec=ClientConfig)
    mock_factory._config.httpx_client = None
    mock_created_client = MagicMock()
    mock_factory.create.return_value = mock_created_client

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        agent = A2AAgent(endpoint="http://localhost:8000", a2a_client_factory=mock_factory)

    # Pre-set the agent card to avoid card resolution complexity
    agent._agent_card = mock_agent_card

    client = await agent._get_or_create_client()

    # Should use factory.create() with the agent card
    mock_factory.create.assert_called_once_with(mock_agent_card)
    assert client is mock_created_client


@pytest.mark.asyncio
async def test_get_or_create_client_factory_caches():
    """Test _get_or_create_client caches the client when factory is provided."""
    mock_factory = MagicMock()
    mock_factory._config = MagicMock(spec=ClientConfig)
    mock_factory._config.httpx_client = None
    mock_created_client = MagicMock()
    mock_factory.create.return_value = mock_created_client

    mock_card = MagicMock(spec=AgentCard)
    mock_card.name = "test"
    mock_card.description = "desc"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        agent = A2AAgent(endpoint="http://localhost:8000", a2a_client_factory=mock_factory)

    agent._agent_card = mock_card

    client1 = await agent._get_or_create_client()
    client2 = await agent._get_or_create_client()

    # factory.create() should only be called once
    mock_factory.create.assert_called_once()
    assert client1 is client2


@pytest.mark.asyncio
async def test_get_or_create_client_transient_without_config():
    """Test _get_or_create_client creates transient clients when no config or factory."""
    agent = A2AAgent(endpoint="http://localhost:8000")

    mock_client_1 = AsyncMock()
    mock_client_2 = AsyncMock()

    with patch("strands.agent.a2a_agent.ClientFactory") as mock_factory_class:
        mock_factory_class.connect = AsyncMock(side_effect=[mock_client_1, mock_client_2])

        client1 = await agent._get_or_create_client()
        client2 = await agent._get_or_create_client()

        # Should connect each time (transient)
        assert mock_factory_class.connect.call_count == 2
        assert client1 is mock_client_1
        assert client2 is mock_client_2


# --- Invocation tests ---


@pytest.mark.asyncio
async def test_invoke_async_success(a2a_agent, mock_agent_card):
    """Test successful async invocation."""
    mock_response = Message(
        message_id=uuid4().hex,
        role=Role.agent,
        parts=[Part(TextPart(kind="text", text="Response"))],
    )

    mock_client = AsyncMock()

    async def mock_send_message(*args, **kwargs):
        yield mock_response

    mock_client.send_message = mock_send_message

    with patch.object(a2a_agent, "_get_or_create_client", return_value=mock_client):
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
    mock_client = AsyncMock()

    async def mock_send_message(*args, **kwargs):
        return
        yield  # Make it an async generator

    mock_client.send_message = mock_send_message

    with patch.object(a2a_agent, "_get_or_create_client", return_value=mock_client):
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


# --- Streaming tests ---


@pytest.mark.asyncio
async def test_stream_async_success(a2a_agent, mock_agent_card):
    """Test successful async streaming."""
    mock_response = Message(
        message_id=uuid4().hex,
        role=Role.agent,
        parts=[Part(TextPart(kind="text", text="Response"))],
    )

    mock_client = AsyncMock()

    async def mock_send_message(*args, **kwargs):
        yield mock_response

    mock_client.send_message = mock_send_message

    with patch.object(a2a_agent, "_get_or_create_client", return_value=mock_client):
        events = []
        async for event in a2a_agent.stream_async("Hello"):
            events.append(event)

        assert len(events) == 2
        assert events[0]["type"] == "a2a_stream"
        assert events[0]["event"] == mock_response
        assert "result" in events[1]
        assert isinstance(events[1]["result"], AgentResult)
        assert events[1]["result"].message["content"][0]["text"] == "Response"


@pytest.mark.asyncio
async def test_stream_async_no_prompt(a2a_agent):
    """Test that stream_async raises ValueError when prompt is None."""
    with pytest.raises(ValueError, match="prompt is required"):
        async for _ in a2a_agent.stream_async(None):
            pass


# --- _is_complete_event tests ---


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


# --- Complete event tracking tests ---


@pytest.mark.asyncio
async def test_stream_async_tracks_complete_events(a2a_agent, mock_agent_card):
    """Test stream_async uses last complete event for final result."""
    from a2a.types import TaskState, TaskStatusUpdateEvent

    mock_task = MagicMock()
    mock_task.artifacts = None

    incomplete_event = MagicMock(spec=TaskStatusUpdateEvent)
    incomplete_event.status = MagicMock()
    incomplete_event.status.state = TaskState.working
    incomplete_event.status.message = None

    complete_event = MagicMock(spec=TaskStatusUpdateEvent)
    complete_event.status = MagicMock()
    complete_event.status.state = TaskState.completed
    complete_event.status.message = MagicMock()
    complete_event.status.message.parts = []

    mock_client = AsyncMock()

    async def mock_send_message(*args, **kwargs):
        yield (mock_task, incomplete_event)
        yield (mock_task, complete_event)

    mock_client.send_message = mock_send_message

    with patch.object(a2a_agent, "_get_or_create_client", return_value=mock_client):
        events = []
        async for event in a2a_agent.stream_async("Hello"):
            events.append(event)

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

    mock_client = AsyncMock()

    async def mock_send_message(*args, **kwargs):
        yield (mock_task, incomplete_event)

    mock_client.send_message = mock_send_message

    with patch.object(a2a_agent, "_get_or_create_client", return_value=mock_client):
        events = []
        async for event in a2a_agent.stream_async("Hello"):
            events.append(event)

        assert len(events) == 2
        assert "result" in events[1]
