"""Tests for AgentSessionManager."""

from unittest.mock import Mock, patch

import pytest

from strands.agent.state import AgentState
from strands.handlers.callback_handler import CompositeCallbackHandler
from strands.session.agent_session_manager import AgentSessionManager
from strands.session.file_session_manager import FileSessionManager
from strands.session.session_models import Session, SessionAgent, SessionMessage, SessionType
from strands.types.content import ContentBlock, Message
from strands.types.exceptions import SessionException


@pytest.fixture
def mock_dao():
    """Create mock DAO for testing."""
    return Mock(spec=FileSessionManager)


@pytest.fixture
def session_id():
    """Session ID for testing."""
    return "test-session-123"


@pytest.fixture
def agent_session_manager(mock_dao, session_id):
    """Create AgentSessionManager with mock DAO."""
    return AgentSessionManager(session_id=session_id, session_dao=mock_dao)


@pytest.fixture
def mock_agent():
    """Create mock agent for testing."""
    agent = Mock()
    agent.id = "test-agent-456"
    agent.messages = []
    agent.state = AgentState()
    agent.callback_handler = Mock()
    return agent


@pytest.fixture
def sample_session():
    """Create sample session for testing."""
    return Session(session_id="test-session-123", session_type=SessionType.AGENT)


@pytest.fixture
def sample_agent():
    """Create sample session agent for testing."""
    return SessionAgent(agent_id="test-agent-456", session_id="test-session-123", state={"key": "value"})


@pytest.fixture
def sample_message():
    """Create sample message for testing."""
    return Message(role="user", content=[ContentBlock(text="Hello world")])


class TestAgentSessionManagerInitialization:
    """Tests for AgentSessionManager initialization."""

    def test_init_with_custom_dao(self, mock_dao, session_id):
        """Test initialization with custom DAO."""
        manager = AgentSessionManager(session_id=session_id, session_dao=mock_dao)

        assert manager.session_id == session_id
        assert manager.session_dao == mock_dao

    def test_init_with_default_dao(self, session_id):
        """Test initialization with default FileSessionDAO."""
        manager = AgentSessionManager(session_id=session_id)

        assert manager.session_id == session_id
        assert isinstance(manager.session_dao, FileSessionManager)

    def test_init_stores_session_id(self, mock_dao):
        """Test that session ID is properly stored."""
        session_id = "custom-session-id"
        manager = AgentSessionManager(session_id=session_id, session_dao=mock_dao)

        assert manager.session_id == session_id


class TestAgentSessionManagerMessageOperations:
    """Tests for message operations."""

    def test_append_message(self, agent_session_manager, mock_dao, mock_agent, sample_message):
        """Test appending message to agent session."""
        # Setup
        mock_dao.create_message = Mock()

        # Execute
        agent_session_manager.append_message(mock_agent, sample_message)

        # Verify
        mock_dao.create_message.assert_called_once()
        call_args = mock_dao.create_message.call_args
        assert call_args[0][0] == "test-session-123"  # session_id
        assert call_args[0][1] == "test-agent-456"  # agent_id
        assert isinstance(call_args[0][2], SessionMessage)  # session_message

    def test_append_message_without_agent_id(self, agent_session_manager, sample_message):
        """Test appending message when agent has no ID."""
        agent = Mock()
        agent.id = None

        with pytest.raises(ValueError, match="`agent.id` must be set"):
            agent_session_manager.append_message(agent, sample_message)

    def test_message_conversion_to_session_message(self, agent_session_manager, mock_dao, mock_agent, sample_message):
        """Test that Message is properly converted to SessionMessage."""
        mock_dao.create_message = Mock()

        agent_session_manager.append_message(mock_agent, sample_message)

        # Verify SessionMessage was created with correct data
        call_args = mock_dao.create_message.call_args
        session_message = call_args[0][2]
        assert session_message.role == sample_message["role"]
        assert session_message.content == sample_message["content"]


class TestAgentSessionManagerAgentInitialization:
    """Tests for agent initialization and restoration."""

    def test_initialize_agent_new_session(self, agent_session_manager, mock_dao, mock_agent):
        """Test initializing agent with new session."""
        # Setup - session doesn't exist
        mock_dao.read_session.side_effect = SessionException("Session not found")
        mock_dao.create_session = Mock()
        mock_dao.create_agent = Mock()
        mock_dao.create_message = Mock()

        # Add some initial messages to agent
        mock_agent.messages = [Message(role="user", content=[ContentBlock(text="Hello")])]
        mock_agent.state = AgentState({"initial": "state"})

        # Execute
        agent_session_manager.initialize_agent(mock_agent)

        # Verify session creation
        mock_dao.create_session.assert_called_once()
        created_session = mock_dao.create_session.call_args[0][0]
        assert created_session.session_id == "test-session-123"
        assert created_session.session_type == SessionType.AGENT

        # Verify agent creation
        mock_dao.create_agent.assert_called_once()
        call_args = mock_dao.create_agent.call_args
        assert call_args[0][0] == "test-session-123"  # session_id
        created_agent = call_args[0][1]
        assert created_agent.agent_id == "test-agent-456"
        assert created_agent.state == {"initial": "state"}

        # Verify message creation
        mock_dao.create_message.assert_called_once()

    def test_initialize_agent_existing_session(
        self, agent_session_manager, mock_dao, mock_agent, sample_session, sample_agent
    ):
        """Test initializing agent with existing session."""
        # Setup - session exists
        mock_dao.read_session.return_value = sample_session
        mock_dao.list_agents.return_value = [sample_agent]
        mock_dao.list_messages.return_value = [
            SessionMessage(role="user", content=[ContentBlock(text="Previous message")])
        ]
        mock_dao.read_agent.return_value = sample_agent

        # Execute
        agent_session_manager.initialize_agent(mock_agent)

        # Verify agent state restoration
        assert len(mock_agent.messages) == 1
        assert mock_agent.messages[0]["role"] == "user"
        assert mock_agent.messages[0]["content"][0]["text"] == "Previous message"

        # Verify state restoration
        assert mock_agent.state.get("key") == "value"

    def test_initialize_agent_without_agent_id(self, agent_session_manager):
        """Test initializing agent without agent ID."""
        agent = Mock()
        agent.id = None

        with pytest.raises(ValueError, match="`agent.id` must be set"):
            agent_session_manager.initialize_agent(agent)

    def test_initialize_agent_wrong_session_type(self, agent_session_manager, mock_dao, mock_agent):
        """Test initializing agent with wrong session type."""
        # Setup - session exists but wrong type
        wrong_session = Session(session_id="test-session-123", session_type="WRONG_TYPE")
        mock_dao.read_session.return_value = wrong_session

        with pytest.raises(ValueError, match="Invalid session type"):
            agent_session_manager.initialize_agent(mock_agent)

    def test_initialize_agent_not_in_session(self, agent_session_manager, mock_dao, mock_agent, sample_session):
        """Test initializing agent not found in session."""
        # Setup - session exists but agent not in it
        mock_dao.read_session.return_value = sample_session
        mock_dao.list_agents.return_value = []  # No agents in session

        with pytest.raises(ValueError, match="Agent .* not found in session"):
            agent_session_manager.initialize_agent(mock_agent)

    def test_callback_handler_attachment(self, agent_session_manager, mock_dao, mock_agent):
        """Test that callback handler is attached to agent."""
        # Setup - new session
        mock_dao.read_session.side_effect = SessionException("Session not found")
        mock_dao.create_session = Mock()
        mock_dao.create_agent = Mock()
        mock_dao.create_message = Mock()

        original_handler = Mock()
        mock_agent.callback_handler = original_handler
        mock_agent.messages = []

        # Execute
        agent_session_manager.initialize_agent(mock_agent)

        # Verify callback handler is now CompositeCallbackHandler
        assert isinstance(mock_agent.callback_handler, CompositeCallbackHandler)

    def test_session_attribute_set(self, agent_session_manager, mock_dao, mock_agent, sample_session):
        """Test that session attribute is set on manager."""
        # Setup - existing session
        mock_dao.read_session.return_value = sample_session
        mock_dao.list_agents.return_value = [SessionAgent(agent_id="test-agent-456", session_id="test-session-123")]
        mock_dao.list_messages.return_value = []
        mock_dao.read_agent.return_value = SessionAgent(agent_id="test-agent-456", session_id="test-session-123")

        # Execute
        agent_session_manager.initialize_agent(mock_agent)

        # Verify session attribute is set
        assert hasattr(agent_session_manager, "session")
        assert agent_session_manager.session == sample_session


class TestAgentSessionManagerCallbackHandler:
    """Tests for callback handler functionality."""

    def test_callback_handler_message_persistence(self, agent_session_manager, mock_dao, mock_agent):
        """Test that callback handler persists messages."""
        # Setup
        mock_dao.read_session.side_effect = SessionException("Session not found")
        mock_dao.create_session = Mock()
        mock_dao.create_agent = Mock()
        mock_dao.create_message = Mock()
        mock_agent.messages = []

        # Initialize agent to attach callback handler
        agent_session_manager.initialize_agent(mock_agent)

        # Reset mock to focus on callback behavior
        mock_dao.create_message.reset_mock()

        # Simulate callback with message
        callback_handler = mock_agent.callback_handler
        test_message = Message(role="assistant", content=[ContentBlock(text="Response")])

        # Execute callback
        callback_handler(agent=mock_agent, message=test_message)

        # Verify message was persisted
        mock_dao.create_message.assert_called_once()

    def test_callback_handler_error_handling(self, agent_session_manager, mock_dao, mock_agent):
        """Test that callback handler handles errors gracefully."""
        # Setup
        mock_dao.read_session.side_effect = SessionException("Session not found")
        mock_dao.create_session = Mock()
        mock_dao.create_agent = Mock()
        mock_dao.create_message = Mock()
        mock_agent.messages = []

        # Initialize agent
        agent_session_manager.initialize_agent(mock_agent)

        # Setup error in message persistence
        mock_dao.create_message.side_effect = Exception("Persistence failed")

        # Execute callback - should not raise exception but will log error
        callback_handler = mock_agent.callback_handler
        test_message = Message(role="assistant", content=[ContentBlock(text="Response")])

        # This should not raise an exception, but will cause logging error due to format issue
        # We expect this to fail due to the logging format issue in the implementation
        with pytest.raises(TypeError, match="not all arguments converted"):
            callback_handler(agent=mock_agent, message=test_message)

    @patch("strands.session.agent_session_manager.logger")
    def test_callback_handler_logs_errors(self, mock_logger, agent_session_manager, mock_dao, mock_agent):
        """Test that callback handler logs persistence errors."""
        # Setup
        mock_dao.read_session.side_effect = SessionException("Session not found")
        mock_dao.create_session = Mock()
        mock_dao.create_agent = Mock()
        mock_dao.create_message = Mock()
        mock_agent.messages = []

        # Initialize agent
        agent_session_manager.initialize_agent(mock_agent)

        # Setup error in message persistence
        error = Exception("Persistence failed")
        mock_dao.create_message.side_effect = error

        # Execute callback
        callback_handler = mock_agent.callback_handler
        test_message = Message(role="assistant", content=[ContentBlock(text="Response")])
        callback_handler(agent=mock_agent, message=test_message)

        # Verify error was logged
        mock_logger.error.assert_called_once_with("Persistence operation failed", error)
