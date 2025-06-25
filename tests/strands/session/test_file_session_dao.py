"""Tests for FileSessionDAO."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from strands.session.exceptions import SessionException
from strands.session.file_session_dao import FileSessionDAO
from strands.session.session_models import Session, SessionAgent, SessionMessage, SessionType
from strands.types.content import ContentBlock


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def file_dao(temp_dir):
    """Create FileSessionDAO with temporary directory."""
    return FileSessionDAO(storage_dir=temp_dir)


@pytest.fixture
def sample_session():
    """Create sample session for testing."""
    return Session(session_id="test-session-123", session_type=SessionType.AGENT)


@pytest.fixture
def sample_agent():
    """Create sample agent for testing."""
    return SessionAgent(agent_id="test-agent-456", session_id="test-session-123", state={"key": "value"})


@pytest.fixture
def sample_message():
    """Create sample message for testing."""
    return SessionMessage(role="user", content=[ContentBlock(text="Hello world")], message_id="test-message-789")


class TestFileSessionDAOInitialization:
    """Tests for FileSessionDAO initialization."""

    def test_init_with_custom_directory(self, temp_dir):
        """Test initialization with custom storage directory."""
        dao = FileSessionDAO(storage_dir=temp_dir)

        assert dao.storage_dir == temp_dir
        assert os.path.exists(temp_dir)

    def test_init_with_default_directory(self):
        """Test initialization with default temp directory."""
        dao = FileSessionDAO()

        assert dao.storage_dir is not None
        assert "strands/sessions" in dao.storage_dir
        assert os.path.exists(dao.storage_dir)

    def test_directory_creation(self, temp_dir):
        """Test that storage directory is created if it doesn't exist."""
        storage_path = os.path.join(temp_dir, "custom", "sessions")
        FileSessionDAO(storage_dir=storage_path)

        assert os.path.exists(storage_path)


class TestFileSessionDAOSessionOperations:
    """Tests for session CRUD operations."""

    def test_create_session(self, file_dao, sample_session):
        """Test creating a new session."""
        result = file_dao.create_session(sample_session)

        assert result == sample_session

        # Verify directory structure created
        session_path = file_dao._get_session_path(sample_session.session_id)
        assert os.path.exists(session_path)

        # Verify session file created
        session_file = os.path.join(session_path, "session.json")
        assert os.path.exists(session_file)

        # Verify content
        with open(session_file, "r") as f:
            data = json.load(f)
            assert data["session_id"] == sample_session.session_id
            assert data["session_type"] == sample_session.session_type.value

    def test_read_session(self, file_dao, sample_session):
        """Test reading an existing session."""
        # Create session first
        file_dao.create_session(sample_session)

        # Read it back
        result = file_dao.read_session(sample_session.session_id)

        assert result.session_id == sample_session.session_id
        assert result.session_type == sample_session.session_type

    def test_read_nonexistent_session(self, file_dao):
        """Test reading a session that doesn't exist."""
        with pytest.raises(SessionException, match="does not exist"):
            file_dao.read_session("nonexistent-session")

    def test_update_session(self, file_dao, sample_session):
        """Test updating an existing session."""
        # Create session first
        file_dao.create_session(sample_session)

        # Get original updated_at
        original_updated_at = sample_session.updated_at

        # Update session (the implementation automatically updates the timestamp)
        file_dao.update_session(sample_session)

        # Verify update (timestamp should be different)
        result = file_dao.read_session(sample_session.session_id)
        assert result.updated_at != original_updated_at

    def test_list_sessions(self, file_dao):
        """Test listing all sessions."""
        # Create multiple sessions
        session1 = Session(session_id="session-1", session_type=SessionType.AGENT)
        session2 = Session(session_id="session-2", session_type=SessionType.AGENT)

        file_dao.create_session(session1)
        file_dao.create_session(session2)

        # List sessions
        sessions = file_dao.list_sessions()

        assert len(sessions) == 2
        session_ids = [s.session_id for s in sessions]
        assert "session-1" in session_ids
        assert "session-2" in session_ids

    def test_delete_session(self, file_dao, sample_session):
        """Test deleting a session."""
        # Create session first
        file_dao.create_session(sample_session)
        session_path = file_dao._get_session_path(sample_session.session_id)
        assert os.path.exists(session_path)

        # Delete session
        file_dao.delete_session(sample_session.session_id)

        # Verify deletion
        assert not os.path.exists(session_path)

    def test_delete_nonexistent_session(self, file_dao):
        """Test deleting a session that doesn't exist."""
        # Should raise an error according to the implementation
        with pytest.raises(SessionException, match="does not exist"):
            file_dao.delete_session("nonexistent-session")


class TestFileSessionDAOAgentOperations:
    """Tests for agent CRUD operations."""

    def test_create_agent(self, file_dao, sample_session, sample_agent):
        """Test creating an agent in a session."""
        # Create session first
        file_dao.create_session(sample_session)

        # Create agent
        file_dao.create_agent(sample_session.session_id, sample_agent)

        # Verify directory structure
        agent_path = file_dao._get_agent_path(sample_session.session_id, sample_agent.agent_id)
        assert os.path.exists(agent_path)

        # Verify agent file
        agent_file = os.path.join(agent_path, "agent.json")
        assert os.path.exists(agent_file)

        # Verify content
        with open(agent_file, "r") as f:
            data = json.load(f)
            assert data["agent_id"] == sample_agent.agent_id
            assert data["state"] == sample_agent.state

    def test_read_agent(self, file_dao, sample_session, sample_agent):
        """Test reading an agent from a session."""
        # Create session and agent
        file_dao.create_session(sample_session)
        file_dao.create_agent(sample_session.session_id, sample_agent)

        # Read agent
        result = file_dao.read_agent(sample_session.session_id, sample_agent.agent_id)

        assert result.agent_id == sample_agent.agent_id
        assert result.session_id == sample_agent.session_id
        assert result.state == sample_agent.state

    def test_update_agent(self, file_dao, sample_session, sample_agent):
        """Test updating an agent."""
        # Create session and agent
        file_dao.create_session(sample_session)
        file_dao.create_agent(sample_session.session_id, sample_agent)

        # Update agent
        sample_agent.state = {"updated": "value"}
        file_dao.update_agent(sample_session.session_id, sample_agent)

        # Verify update
        result = file_dao.read_agent(sample_session.session_id, sample_agent.agent_id)
        assert result.state == {"updated": "value"}

    def test_list_agents(self, file_dao, sample_session):
        """Test listing agents in a session."""
        # Create session
        file_dao.create_session(sample_session)

        # Create multiple agents
        agent1 = SessionAgent(agent_id="agent-1", session_id=sample_session.session_id)
        agent2 = SessionAgent(agent_id="agent-2", session_id=sample_session.session_id)

        file_dao.create_agent(sample_session.session_id, agent1)
        file_dao.create_agent(sample_session.session_id, agent2)

        # List agents
        agents = file_dao.list_agents(sample_session.session_id)

        assert len(agents) == 2
        agent_ids = [a.agent_id for a in agents]
        assert "agent-1" in agent_ids
        assert "agent-2" in agent_ids

    def test_delete_agent(self, file_dao, sample_session, sample_agent):
        """Test deleting an agent."""
        # Create session and agent
        file_dao.create_session(sample_session)
        file_dao.create_agent(sample_session.session_id, sample_agent)

        agent_path = file_dao._get_agent_path(sample_session.session_id, sample_agent.agent_id)
        assert os.path.exists(agent_path)

        # Delete agent
        file_dao.delete_agent(sample_session.session_id, sample_agent.agent_id)

        # Verify deletion
        assert not os.path.exists(agent_path)


class TestFileSessionDAOMessageOperations:
    """Tests for message CRUD operations."""

    def test_create_message(self, file_dao, sample_session, sample_agent, sample_message):
        """Test creating a message for an agent."""
        # Create session and agent
        file_dao.create_session(sample_session)
        file_dao.create_agent(sample_session.session_id, sample_agent)

        # Create message
        file_dao.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

        # Verify message file
        message_path = file_dao._get_message_path(
            sample_session.session_id, sample_agent.agent_id, sample_message.message_id
        )
        assert os.path.exists(message_path)

        # Verify content
        with open(message_path, "r") as f:
            data = json.load(f)
            assert data["message_id"] == sample_message.message_id
            assert data["role"] == sample_message.role

    def test_read_message(self, file_dao, sample_session, sample_agent, sample_message):
        """Test reading a message."""
        # Create session, agent, and message
        file_dao.create_session(sample_session)
        file_dao.create_agent(sample_session.session_id, sample_agent)
        file_dao.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

        # Read message
        result = file_dao.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)

        assert result.message_id == sample_message.message_id
        assert result.role == sample_message.role
        assert result.content == sample_message.content

    def test_list_messages_all(self, file_dao, sample_session, sample_agent):
        """Test listing all messages for an agent."""
        # Create session and agent
        file_dao.create_session(sample_session)
        file_dao.create_agent(sample_session.session_id, sample_agent)

        # Create multiple messages
        messages = []
        for i in range(5):
            message = SessionMessage(
                role="user", content=[ContentBlock(text=f"Message {i}")], message_id=f"message-{i}"
            )
            messages.append(message)
            file_dao.create_message(sample_session.session_id, sample_agent.agent_id, message)

        # List all messages
        result = file_dao.list_messages(sample_session.session_id, sample_agent.agent_id)

        assert len(result) == 5
        message_ids = [m.message_id for m in result]
        for i in range(5):
            assert f"message-{i}" in message_ids

    def test_list_messages_with_limit(self, file_dao, sample_session, sample_agent):
        """Test listing messages with limit."""
        # Create session and agent
        file_dao.create_session(sample_session)
        file_dao.create_agent(sample_session.session_id, sample_agent)

        # Create multiple messages
        for i in range(10):
            message = SessionMessage(
                role="user", content=[ContentBlock(text=f"Message {i}")], message_id=f"message-{i}"
            )
            file_dao.create_message(sample_session.session_id, sample_agent.agent_id, message)

        # List with limit
        result = file_dao.list_messages(sample_session.session_id, sample_agent.agent_id, limit=3)

        assert len(result) == 3

    def test_list_messages_with_offset(self, file_dao, sample_session, sample_agent):
        """Test listing messages with offset."""
        # Create session and agent
        file_dao.create_session(sample_session)
        file_dao.create_agent(sample_session.session_id, sample_agent)

        # Create multiple messages
        for i in range(10):
            message = SessionMessage(
                role="user", content=[ContentBlock(text=f"Message {i}")], message_id=f"message-{i}"
            )
            file_dao.create_message(sample_session.session_id, sample_agent.agent_id, message)

        # List with offset
        result = file_dao.list_messages(sample_session.session_id, sample_agent.agent_id, offset=5)

        assert len(result) == 5

    def test_update_message(self, file_dao, sample_session, sample_agent, sample_message):
        """Test updating a message."""
        # Create session, agent, and message
        file_dao.create_session(sample_session)
        file_dao.create_agent(sample_session.session_id, sample_agent)
        file_dao.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

        # Update message
        sample_message.content = [ContentBlock(text="Updated content")]
        file_dao.update_message(sample_session.session_id, sample_agent.agent_id, sample_message)

        # Verify update
        result = file_dao.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)
        # Content is stored as dict, so access it as such
        assert result.content[0]["text"] == "Updated content"

    def test_delete_message(self, file_dao, sample_session, sample_agent, sample_message):
        """Test deleting a message."""
        # Create session, agent, and message
        file_dao.create_session(sample_session)
        file_dao.create_agent(sample_session.session_id, sample_agent)
        file_dao.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

        message_path = file_dao._get_message_path(
            sample_session.session_id, sample_agent.agent_id, sample_message.message_id
        )
        assert os.path.exists(message_path)

        # Delete message
        file_dao.delete_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)

        # Verify deletion
        assert not os.path.exists(message_path)


class TestFileSessionDAOErrorHandling:
    """Tests for error handling scenarios."""

    def test_corrupted_json_file(self, file_dao, temp_dir):
        """Test handling of corrupted JSON files."""
        # Create a corrupted session file
        session_path = os.path.join(temp_dir, "session_test")
        os.makedirs(session_path, exist_ok=True)
        session_file = os.path.join(session_path, "session.json")

        with open(session_file, "w") as f:
            f.write("invalid json content")

        # Should raise SessionException
        with pytest.raises(SessionException, match="Invalid JSON"):
            file_dao.read_session("test")

    def test_permission_error_handling(self, file_dao):
        """Test handling of permission errors."""
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            session = Session(session_id="test", session_type=SessionType.AGENT)

            with pytest.raises(SessionException):
                file_dao.create_session(session)
