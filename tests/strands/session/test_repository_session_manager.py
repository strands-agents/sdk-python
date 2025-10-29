"""Tests for AgentSessionManager."""

from unittest.mock import Mock

import pytest

from strands.agent.agent import Agent
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands.agent.conversation_manager.summarizing_conversation_manager import SummarizingConversationManager
from strands.agent.interrupt import InterruptState
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


@pytest.fixture
def mock_multi_agent():
    """Create mock multi-agent for testing."""

    mock = Mock()
    mock.id = "test-multi-agent"
    mock.serialize_state.return_value = {"id": "test-multi-agent", "state": {"key": "value"}}
    mock.deserialize_state = Mock()
    return mock


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
    session_agent = SessionAgent(
        agent_id="existing-agent",
        state={"key": "value"},
        conversation_manager_state=SlidingWindowConversationManager().get_state(),
        _internal_state={"interrupt_state": {"interrupts": {}, "context": {"test": "init"}, "activated": False}},
    )
    session_manager.session_repository.create_agent("test-session", session_agent)

    # Create some messages
    message = SessionMessage(
        message={
            "role": "user",
            "content": [ContentBlock(text="Hello")],
        },
        message_id=0,
    )
    session_manager.session_repository.create_message("test-session", "existing-agent", message)

    # Initialize agent
    session_manager.initialize(agent)

    # Verify agent state restored
    assert agent.state.get("key") == "value"
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "user"
    assert agent.messages[0]["content"][0]["text"] == "Hello"
    assert agent._interrupt_state == InterruptState(interrupts={}, context={"test": "init"}, activated=False)


def test_initialize_restores_existing_agent_with_summarizing_conversation_manager(session_manager):
    """Test that initializing an existing agent restores its state."""
    conversation_manager = SummarizingConversationManager()
    conversation_manager.removed_message_count = 1
    conversation_manager._summary_message = {"role": "assistant", "content": [{"text": "summary"}]}

    # Create agent in repository first
    session_agent = SessionAgent(
        agent_id="existing-agent",
        state={"key": "value"},
        conversation_manager_state=conversation_manager.get_state(),
    )
    session_manager.session_repository.create_agent("test-session", session_agent)

    # Create some messages
    message = SessionMessage(
        message={
            "role": "user",
            "content": [ContentBlock(text="Hello")],
        },
        message_id=0,
    )
    # Create two messages as one will be removed by the conversation manager
    session_manager.session_repository.create_message("test-session", "existing-agent", message)
    message.message_id = 1
    session_manager.session_repository.create_message("test-session", "existing-agent", message)

    # Initialize agent
    agent = Agent(agent_id="existing-agent", conversation_manager=SummarizingConversationManager())
    session_manager.initialize(agent)

    # Verify agent state restored
    assert agent.state.get("key") == "value"
    # The session message plus the summary message
    assert len(agent.messages) == 2
    assert agent.messages[1]["role"] == "user"
    assert agent.messages[1]["content"][0]["text"] == "Hello"
    assert agent.conversation_manager.removed_message_count == 1


def test_append_message(session_manager):
    """Test appending a message to an agent's session."""
    # Set agent ID and session manager
    agent = Agent(agent_id="test-agent", session_manager=session_manager)

    # Create message
    message = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}

    # Append message
    session_manager.append_message(message, agent)

    # Verify message created in repository
    messages = session_manager.session_repository.list_messages("test-session", "test-agent")
    assert len(messages) == 1
    assert messages[0].message["role"] == "user"
    assert messages[0].message["content"][0]["text"] == "Hello"


def test_sync_multi_agent(session_manager, mock_multi_agent):
    """Test syncing multi-agent state."""
    # Create multi-agent first
    session_manager.session_repository.create_multi_agent("test-session", mock_multi_agent)

    # Sync multi-agent
    session_manager.sync_multi_agent(mock_multi_agent)

    # Verify repository update_multi_agent was called
    state = session_manager.session_repository.read_multi_agent("test-session", mock_multi_agent.id)
    assert state["id"] == "test-multi-agent"
    assert state["state"] == {"key": "value"}


def test_initialize_multi_agent_new(session_manager, mock_multi_agent):
    """Test initializing new multi-agent state."""
    session_manager.initialize_multi_agent(mock_multi_agent)

    # Verify multi-agent was created
    state = session_manager.session_repository.read_multi_agent("test-session", mock_multi_agent.id)
    assert state["id"] == "test-multi-agent"
    assert state["state"] == {"key": "value"}


def test_initialize_multi_agent_existing(session_manager, mock_multi_agent):
    """Test initializing existing multi-agent state."""
    # Create existing state first
    session_manager.session_repository.create_multi_agent("test-session", mock_multi_agent)

    # Create a mock with updated state for the update call
    updated_mock = Mock()
    updated_mock.id = "test-multi-agent"
    existing_state = {"id": "test-multi-agent", "state": {"restored": "data"}}
    updated_mock.serialize_state.return_value = existing_state
    session_manager.session_repository.update_multi_agent("test-session", updated_mock)

    # Initialize multi-agent
    session_manager.initialize_multi_agent(mock_multi_agent)

    # Verify deserialize_state was called with existing state
    mock_multi_agent.deserialize_state.assert_called_once_with(existing_state)
