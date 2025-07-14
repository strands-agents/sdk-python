"""Tests for FileSessionManager."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest

from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.session.file_session_manager import FileSessionManager
from strands.types.content import ContentBlock
from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage, SessionType


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def file_manager(temp_dir):
    """Create FileSessionManager for testing."""
    return FileSessionManager(session_id="test", storage_dir=temp_dir)


@pytest.fixture
def sample_session():
    """Create sample session for testing."""
    return Session(session_id="test-session", session_type=SessionType.AGENT)


@pytest.fixture
def sample_agent():
    """Create sample agent for testing."""
    return SessionAgent(
        agent_id="test-agent", state={"key": "value"}, conversation_manager_state=NullConversationManager().get_state()
    )


@pytest.fixture
def sample_message():
    """Create sample message for testing."""
    return SessionMessage.from_message(
        message={
            "role": "user",
            "content": [ContentBlock(text="Hello world")],
        },
        index=0,
    )


class TestFileSessionManagerSessionOperations:
    """Tests for session operations."""

    def test_create_session(self, file_manager, sample_session):
        """Test creating a session."""
        file_manager.create_session(sample_session)

        # Verify directory structure created
        session_path = file_manager._get_session_path(sample_session.session_id)
        assert os.path.exists(session_path)

        # Verify session file created
        session_file = os.path.join(session_path, "session.json")
        assert os.path.exists(session_file)

        # Verify content
        with open(session_file, "r") as f:
            data = json.load(f)
            assert data["session_id"] == sample_session.session_id
            assert data["session_type"] == sample_session.session_type

    def test_read_session(self, file_manager, sample_session):
        """Test reading an existing session."""
        # Create session first
        file_manager.create_session(sample_session)

        # Read it back
        result = file_manager.read_session(sample_session.session_id)

        assert result.session_id == sample_session.session_id
        assert result.session_type == sample_session.session_type

    def test_read_nonexistent_session(self, file_manager):
        """Test reading a session that doesn't exist."""
        result = file_manager.read_session("nonexistent-session")
        assert result is None

    def test_delete_session(self, file_manager, sample_session):
        """Test deleting a session."""
        # Create session first
        file_manager.create_session(sample_session)
        session_path = file_manager._get_session_path(sample_session.session_id)
        assert os.path.exists(session_path)

        # Delete session
        file_manager.delete_session(sample_session.session_id)

        # Verify deletion
        assert not os.path.exists(session_path)

    def test_delete_nonexistent_session(self, file_manager):
        """Test deleting a session that doesn't exist."""
        # Should raise an error according to the implementation
        with pytest.raises(SessionException, match="does not exist"):
            file_manager.delete_session("nonexistent-session")


class TestFileSessionManagerAgentOperations:
    """Tests for agent operations."""

    def test_create_agent(self, file_manager, sample_session, sample_agent):
        """Test creating an agent in a session."""
        # Create session first
        file_manager.create_session(sample_session)

        # Create agent
        file_manager.create_agent(sample_session.session_id, sample_agent)

        # Verify directory structure
        agent_path = file_manager._get_agent_path(sample_session.session_id, sample_agent.agent_id)
        assert os.path.exists(agent_path)

        # Verify agent file
        agent_file = os.path.join(agent_path, "agent.json")
        assert os.path.exists(agent_file)

        # Verify content
        with open(agent_file, "r") as f:
            data = json.load(f)
            assert data["agent_id"] == sample_agent.agent_id
            assert data["state"] == sample_agent.state

    def test_read_agent(self, file_manager, sample_session, sample_agent):
        """Test reading an agent from a session."""
        # Create session and agent
        file_manager.create_session(sample_session)
        file_manager.create_agent(sample_session.session_id, sample_agent)

        # Read agent
        result = file_manager.read_agent(sample_session.session_id, sample_agent.agent_id)

        assert result.agent_id == sample_agent.agent_id
        assert result.state == sample_agent.state

    def test_read_nonexistent_agent(self, file_manager, sample_session):
        """Test reading an agent that doesn't exist."""
        result = file_manager.read_agent(sample_session.session_id, "nonexistent_agent")
        assert result is None

    def test_update_agent(self, file_manager, sample_session, sample_agent):
        """Test updating an agent."""
        # Create session and agent
        file_manager.create_session(sample_session)
        file_manager.create_agent(sample_session.session_id, sample_agent)

        # Update agent
        sample_agent.state = {"updated": "value"}
        file_manager.update_agent(sample_session.session_id, sample_agent)

        # Verify update
        result = file_manager.read_agent(sample_session.session_id, sample_agent.agent_id)
        assert result.state == {"updated": "value"}

    def test_update_nonexistent_agent(self, file_manager, sample_session, sample_agent):
        """Test updating an agent."""
        # Create session and agent
        file_manager.create_session(sample_session)

        # Update agent
        with pytest.raises(SessionException):
            file_manager.update_agent(sample_session.session_id, sample_agent)


class TestFileSessionManagerMessageOperations:
    """Tests for message operations."""

    def test_create_message(self, file_manager, sample_session, sample_agent, sample_message):
        """Test creating a message for an agent."""
        # Create session and agent
        file_manager.create_session(sample_session)
        file_manager.create_agent(sample_session.session_id, sample_agent)

        # Create message
        file_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

        # Verify message file
        message_path = file_manager._get_message_path(
            sample_session.session_id, sample_agent.agent_id, sample_message.message_id
        )
        assert os.path.exists(message_path)

        # Verify content
        with open(message_path, "r") as f:
            data = json.load(f)
            assert data["message_id"] == sample_message.message_id

    def test_read_message(self, file_manager, sample_session, sample_agent, sample_message):
        """Test reading a message."""
        # Create session, agent, and message
        file_manager.create_session(sample_session)
        file_manager.create_agent(sample_session.session_id, sample_agent)
        file_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

        # Create multiple messages when reading
        sample_message.message_id = sample_message.message_id + 1
        file_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

        # Read message
        result = file_manager.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)

        assert result.message_id == sample_message.message_id
        assert result.message["role"] == sample_message.message["role"]
        assert result.message["content"] == sample_message.message["content"]

    def test_read_messages_with_new_agent(self, file_manager, sample_session, sample_agent):
        """Test reading a message with with a new agent."""
        # Create session and agent
        file_manager.create_session(sample_session)
        file_manager.create_agent(sample_session.session_id, sample_agent)

        result = file_manager.read_message(sample_session.session_id, sample_agent.agent_id, "nonexistent_message")

        assert result is None

    def test_read_nonexistent_message(self, file_manager, sample_session, sample_agent):
        """Test reading a message that doesnt exist."""
        result = file_manager.read_message(sample_session.session_id, sample_agent.agent_id, "nonexistent_message")
        assert result is None

    def test_list_messages_all(self, file_manager, sample_session, sample_agent):
        """Test listing all messages for an agent."""
        # Create session and agent
        file_manager.create_session(sample_session)
        file_manager.create_agent(sample_session.session_id, sample_agent)

        # Create multiple messages
        messages = []
        for i in range(5):
            message = SessionMessage(
                message={
                    "role": "user",
                    "content": [ContentBlock(text=f"Message {i}")],
                },
                message_id=i,
            )
            messages.append(message)
            file_manager.create_message(sample_session.session_id, sample_agent.agent_id, message)

        # List all messages
        result = file_manager.list_messages(sample_session.session_id, sample_agent.agent_id)

        assert len(result) == 5

    def test_list_messages_with_limit(self, file_manager, sample_session, sample_agent):
        """Test listing messages with limit."""
        # Create session and agent
        file_manager.create_session(sample_session)
        file_manager.create_agent(sample_session.session_id, sample_agent)

        # Create multiple messages
        for i in range(10):
            message = SessionMessage(
                message={
                    "role": "user",
                    "content": [ContentBlock(text=f"Message {i}")],
                },
                message_id=i,
            )
            file_manager.create_message(sample_session.session_id, sample_agent.agent_id, message)

        # List with limit
        result = file_manager.list_messages(sample_session.session_id, sample_agent.agent_id, limit=3)

        assert len(result) == 3

    def test_list_messages_with_offset(self, file_manager, sample_session, sample_agent):
        """Test listing messages with offset."""
        # Create session and agent
        file_manager.create_session(sample_session)
        file_manager.create_agent(sample_session.session_id, sample_agent)

        # Create multiple messages
        for i in range(10):
            message = SessionMessage(
                message={
                    "role": "user",
                    "content": [ContentBlock(text=f"Message {i}")],
                },
                message_id=i,
            )
            file_manager.create_message(sample_session.session_id, sample_agent.agent_id, message)

        # List with offset
        result = file_manager.list_messages(sample_session.session_id, sample_agent.agent_id, offset=5)

        assert len(result) == 5

    def test_list_messages_with_new_agent(self, file_manager, sample_session, sample_agent):
        """Test listing messages with new agent."""
        # Create session and agent
        file_manager.create_session(sample_session)
        file_manager.create_agent(sample_session.session_id, sample_agent)

        result = file_manager.list_messages(sample_session.session_id, sample_agent.agent_id)

        assert len(result) == 0

    def test_update_message(self, file_manager, sample_session, sample_agent, sample_message):
        """Test updating a message."""
        # Create session, agent, and message
        file_manager.create_session(sample_session)
        file_manager.create_agent(sample_session.session_id, sample_agent)
        file_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

        # Update message
        sample_message.message["content"] = [ContentBlock(text="Updated content")]
        file_manager.update_message(sample_session.session_id, sample_agent.agent_id, sample_message)

        # Verify update
        result = file_manager.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)
        assert result.message["content"][0]["text"] == "Updated content"

    def test_update_nonexistent_message(self, file_manager, sample_session, sample_agent, sample_message):
        """Test updating a message."""
        # Create session, agent, and message
        file_manager.create_session(sample_session)
        file_manager.create_agent(sample_session.session_id, sample_agent)

        # Update nonexistent message
        with pytest.raises(SessionException):
            file_manager.update_message(sample_session.session_id, sample_agent.agent_id, sample_message)


class TestFileSessionManagerErrorHandling:
    """Tests for error handling scenarios."""

    def test_corrupted_json_file(self, file_manager, temp_dir):
        """Test handling of corrupted JSON files."""
        # Create a corrupted session file
        session_path = os.path.join(temp_dir, "session_test")
        os.makedirs(session_path, exist_ok=True)
        session_file = os.path.join(session_path, "session.json")

        with open(session_file, "w") as f:
            f.write("invalid json content")

        # Should raise SessionException
        with pytest.raises(SessionException, match="Invalid JSON"):
            file_manager._read_file(session_file)

    def test_permission_error_handling(self, file_manager):
        """Test handling of permission errors."""
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            session = Session(session_id="test", session_type=SessionType.AGENT)

            with pytest.raises(SessionException):
                file_manager.create_session(session)
