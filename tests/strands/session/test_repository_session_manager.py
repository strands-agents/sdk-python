"""Tests for AgentSessionManager."""

import pytest

from strands.agent.agent import Agent
from strands.session.repository_session_manager import RepositorySessionManager
from strands.types.content import ContentBlock
from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage, SessionType
from tests.fixtures.mock_session_repository import MockedSessionRepository


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    return MockedSessionRepository()


@pytest.fixture
def session_manager(mock_repository):
    """Create a session manager with mock repository."""
    return RepositorySessionManager(session_id="test-session", session_repository=mock_repository)


@pytest.fixture
def agent():
    """Create a mock agent."""
    return Agent(messages=[{"role": "user", "content": [{"text": "Hello!"}]}])


def test_init_creates_session_if_not_exists(mock_repository):
    """Test that init creates a session if it doesn't exist."""
    # Session doesn't exist yet
    assert mock_repository.read_session("test-session") is None

    # Creating manager should create session
    RepositorySessionManager(session_id="test-session", session_repository=mock_repository)

    # Verify session created
    session = mock_repository.read_session("test-session")
    assert session is not None
    assert session.session_id == "test-session"
    assert session.session_type == SessionType.AGENT


def test_init_uses_existing_session(mock_repository):
    """Test that init uses existing session if it exists."""
    # Create session first
    session = Session(session_id="test-session", session_type=SessionType.AGENT)
    mock_repository.create_session(session)

    # Creating manager should use existing session
    manager = RepositorySessionManager(session_id="test-session", session_repository=mock_repository)

    # Verify session used
    assert manager.session == session


def test_initialize_with_existing_agent_id(session_manager, agent):
    """Test initializing an agent with existing agent_id."""
    # Set agent ID
    agent.agent_id = "custom-agent"

    # Initialize agent
    session_manager.initialize(agent)

    # Verify agent created in repository
    agent_data = session_manager.session_repository.read_agent("test-session", "custom-agent")
    assert agent_data is not None
    assert agent_data.agent_id == "custom-agent"


def test_initialize_multiple_agents_without_id(session_manager, agent):
    """Test initializing multiple agents with same ID."""
    # First agent initialization works
    agent.agent_id = "custom-agent"
    session_manager.initialize(agent)

    # Second agent with no set agent_id should fail
    agent2 = Agent(agent_id="custom-agent")

    with pytest.raises(SessionException, match="The `agent_id` of an agent must be unique in a session."):
        session_manager.initialize(agent2)


def test_initialize_restores_existing_agent(session_manager, agent):
    """Test that initializing an existing agent restores its state."""
    # Set agent ID
    agent.agent_id = "existing-agent"

    # Create agent in repository first
    session_agent = SessionAgent(agent_id="existing-agent", state={"key": "value"})
    session_manager.session_repository.create_agent("test-session", session_agent)

    # Create some messages
    message = SessionMessage(
        message={
            "role": "user",
            "content": [ContentBlock(text="Hello")],
        }
    )
    session_manager.session_repository.create_message("test-session", "existing-agent", message)

    # Initialize agent
    session_manager.initialize(agent)

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
        state={},
    )
    session_manager.session_repository.create_agent("test-session", session_agent)

    # Create message
    message = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}

    # Append message
    session_manager.append_message(message, agent)

    # Verify message created in repository
    messages = session_manager.session_repository.list_messages("test-session", "test-agent")
    assert len(messages) == 1
    assert messages[0].message["role"] == "user"
    assert messages[0].message["content"][0]["text"] == "Hello"
