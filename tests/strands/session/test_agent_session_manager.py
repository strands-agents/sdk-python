"""Tests for AgentSessionManager."""

from unittest.mock import MagicMock

import pytest

from strands.agent.agent import Agent
from strands.hooks.events import AgentInitializedEvent, MessageAddedEvent
from strands.session.agent_session_manager import AgentSessionManager
from strands.types.content import ContentBlock
from strands.types.session import Session, SessionAgent, SessionMessage, SessionType
from tests.fixtures.mock_session_repository import MockedSessionRepository


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    return MockedSessionRepository()


@pytest.fixture
def session_manager(mock_repository):
    """Create a session manager with mock repository."""
    return AgentSessionManager(session_id="test-session", session_repository=mock_repository)


@pytest.fixture
def agent():
    """Create a mock agent."""
    agent = MagicMock(spec=Agent)
    agent.agent_id = None
    agent.messages = [{"role": "user", "content": [{"text": "Hello!"}]}]
    agent.state = MagicMock()
    agent.event_loop_metrics = MagicMock()
    agent.event_loop_metrics.to_dict.return_value = {}
    return agent


def test_init_creates_session_if_not_exists(mock_repository):
    """Test that init creates a session if it doesn't exist."""
    # Session doesn't exist yet
    assert mock_repository.read_session("test-session") is None

    # Creating manager should create session
    AgentSessionManager(session_id="test-session", session_repository=mock_repository)

    # Verify session created
    session = mock_repository.read_session("test-session")
    assert session is not None
    assert session["session_id"] == "test-session"
    assert session["session_type"] == SessionType.AGENT


def test_init_uses_existing_session(mock_repository):
    """Test that init uses existing session if it exists."""
    # Create session first
    session = Session(session_id="test-session", session_type=SessionType.AGENT)
    mock_repository.create_session(session)

    # Creating manager should use existing session
    manager = AgentSessionManager(session_id="test-session", session_repository=mock_repository)

    # Verify session used
    assert manager.session == session


def test_initialize_with_no_agent_id(session_manager, agent):
    """Test initializing an agent with no agent_id."""
    # Agent has no ID
    assert agent.agent_id is None

    # Initialize agent
    event = AgentInitializedEvent(agent=agent)
    session_manager.initialize(event)

    # Verify agent ID set to default
    assert agent.agent_id == "default"

    # Verify agent created in repository
    agent_data = session_manager.session_repository.read_agent("test-session", "default")
    assert agent_data is not None
    assert agent_data["agent_id"] == "default"


def test_initialize_with_existing_agent_id(session_manager, agent):
    """Test initializing an agent with existing agent_id."""
    # Set agent ID
    agent.agent_id = "custom-agent"

    # Initialize agent
    event = AgentInitializedEvent(agent=agent)
    session_manager.initialize(event)

    # Verify agent created in repository
    agent_data = session_manager.session_repository.read_agent("test-session", "custom-agent")
    assert agent_data is not None
    assert agent_data["agent_id"] == "custom-agent"


def test_initialize_multiple_agents_without_id(session_manager, agent):
    """Test initializing multiple agents without IDs."""
    # First agent initialization works
    event1 = AgentInitializedEvent(agent=agent)
    session_manager.initialize(event1)

    # Second agent without ID should fail
    agent2 = MagicMock(spec=Agent)
    agent2.agent_id = None
    event2 = AgentInitializedEvent(agent=agent2)

    with pytest.raises(ValueError, match="only one agent with no `agent_id`"):
        session_manager.initialize(event2)


def test_initialize_restores_existing_agent(session_manager, agent):
    """Test that initializing an existing agent restores its state."""
    # Set agent ID
    agent.agent_id = "existing-agent"

    # Create agent in repository first
    session_agent = SessionAgent(
        agent_id="existing-agent",
        session_id="test-session",
        state={"key": "value"},
        event_loop_metrics={},
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
    )
    session_manager.session_repository.create_agent("test-session", session_agent)

    # Create some messages
    message = SessionMessage(
        role="user",
        content=[ContentBlock(text="Hello")],
        message_id="test-message",
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
    )
    session_manager.session_repository.create_message("test-session", "existing-agent", message)

    # Initialize agent
    event = AgentInitializedEvent(agent=agent)
    session_manager.initialize(event)

    # Verify agent state restored
    assert agent.state.get("key") == "value"
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "user"
    assert agent.messages[0]["content"][0]["text"] == "Hello"


def test_append_message(session_manager, agent):
    """Test appending a message to an agent's session."""
    # Set agent ID
    agent.agent_id = "test-agent"

    # Create agent in repository
    session_agent = SessionAgent(
        agent_id="test-agent",
        session_id="test-session",
        state={},
        event_loop_metrics={},
        created_at="2025-01-01T00:00:00Z",
        updated_at="2025-01-01T00:00:00Z",
    )
    session_manager.session_repository.create_agent("test-session", session_agent)

    # Create message
    message = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}

    # Append message
    event = MessageAddedEvent(agent=agent, message=message)
    session_manager.append_message(event)

    # Verify message created in repository
    messages = session_manager.session_repository.list_messages("test-session", "test-agent")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["text"] == "Hello"


def test_append_message_without_agent_id(session_manager, agent):
    """Test appending a message to an agent without ID."""
    # Agent has no ID
    agent.agent_id = None

    # Create message
    message = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}

    # Append message should fail
    event = MessageAddedEvent(agent=agent, message=message)
    with pytest.raises(ValueError, match="`agent.agent_id` must be set"):
        session_manager.append_message(event)
