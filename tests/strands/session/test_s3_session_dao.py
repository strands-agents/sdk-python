"""Tests for S3SessionDAO."""

import json
from unittest.mock import Mock, patch

import boto3
import pytest
from moto import mock_aws

from strands.session.s3_session_manager import S3SessionDAO
from strands.session.session_models import Session, SessionAgent, SessionMessage, SessionType
from strands.types.content import ContentBlock
from strands.types.exceptions import SessionException


@pytest.fixture
def s3_bucket():
    """S3 bucket name for testing."""
    return "test-session-bucket"


@pytest.fixture
def s3_prefix():
    """S3 prefix for testing."""
    return "sessions/"


@pytest.fixture
def s3_dao(s3_bucket, s3_prefix):
    """Create S3SessionDAO with mocked S3."""
    with mock_aws():
        # Create the bucket
        s3_client = boto3.client("s3", region_name="us-west-2")
        s3_client.create_bucket(Bucket=s3_bucket, CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

        yield S3SessionDAO(bucket=s3_bucket, prefix=s3_prefix, region_name="us-west-2")


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


def test_init_with_required_params():
    """Test initialization with required parameters."""
    dao = S3SessionDAO(bucket="test-bucket")

    assert dao.bucket == "test-bucket"
    assert dao.prefix == ""


def test_init_with_all_params():
    """Test initialization with all parameters."""
    dao = S3SessionDAO(bucket="test-bucket", prefix="sessions/", region_name="us-east-1")

    assert dao.bucket == "test-bucket"
    assert dao.prefix == "sessions/"


def test_user_agent_configuration():
    """Test that strands-agents is added to user agent."""
    with patch("boto3.Session") as mock_session_class:
        mock_session = Mock()
        mock_session_class.return_value = mock_session

        S3SessionDAO(bucket="test-bucket")

        # Verify client was called with user agent config
        mock_session.client.assert_called_once()
        call_args = mock_session.client.call_args
        assert "config" in call_args[1]


def test_create_session(s3_dao, sample_session):
    """Test creating a session in S3."""
    result = s3_dao.create_session(sample_session)

    assert result == sample_session

    # Verify S3 object created
    key = f"{s3_dao._get_session_path(sample_session.session_id)}session.json"
    response = s3_dao.client.get_object(Bucket=s3_dao.bucket, Key=key)
    data = json.loads(response["Body"].read().decode("utf-8"))

    assert data["session_id"] == sample_session.session_id
    assert data["session_type"] == sample_session.session_type.value


def test_read_session(s3_dao, sample_session):
    """Test reading a session from S3."""
    # Create session first
    s3_dao.create_session(sample_session)

    # Read it back
    result = s3_dao.read_session(sample_session.session_id)

    assert result.session_id == sample_session.session_id
    assert result.session_type == sample_session.session_type


def test_read_nonexistent_session(s3_dao):
    """Test reading a session that doesn't exist in S3."""
    with pytest.raises(SessionException, match="not found"):
        s3_dao.read_session("nonexistent-session")


def test_update_session(s3_dao, sample_session):
    """Test updating a session in S3."""
    # Create session first
    s3_dao.create_session(sample_session)

    # Get original updated_at
    original_updated_at = sample_session.updated_at

    # Update session (the implementation automatically updates the timestamp)
    s3_dao.update_session(sample_session)

    # Verify update (timestamp should be different)
    result = s3_dao.read_session(sample_session.session_id)
    assert result.updated_at != original_updated_at


def test_list_sessions(s3_dao):
    """Test listing all sessions from S3."""
    # Create multiple sessions
    session1 = Session(session_id="session-1", session_type=SessionType.AGENT)
    session2 = Session(session_id="session-2", session_type=SessionType.AGENT)

    s3_dao.create_session(session1)
    s3_dao.create_session(session2)

    # List sessions
    sessions = s3_dao.list_sessions()

    assert len(sessions) == 2
    session_ids = [s.session_id for s in sessions]
    assert "session-1" in session_ids
    assert "session-2" in session_ids


def test_create_agent(s3_dao, sample_session, sample_agent):
    """Test creating an agent in S3."""
    # Create session first
    s3_dao.create_session(sample_session)

    # Create agent
    s3_dao.create_agent(sample_session.session_id, sample_agent)

    # Verify S3 object created
    key = f"{s3_dao._get_agent_path(sample_session.session_id, sample_agent.agent_id)}agent.json"
    response = s3_dao.client.get_object(Bucket=s3_dao.bucket, Key=key)
    data = json.loads(response["Body"].read().decode("utf-8"))

    assert data["agent_id"] == sample_agent.agent_id
    assert data["state"] == sample_agent.state


def test_read_agent(s3_dao, sample_session, sample_agent):
    """Test reading an agent from S3."""
    # Create session and agent
    s3_dao.create_session(sample_session)
    s3_dao.create_agent(sample_session.session_id, sample_agent)

    # Read agent
    result = s3_dao.read_agent(sample_session.session_id, sample_agent.agent_id)

    assert result.agent_id == sample_agent.agent_id
    assert result.session_id == sample_agent.session_id
    assert result.state == sample_agent.state


def test_update_agent(s3_dao, sample_session, sample_agent):
    """Test updating an agent in S3."""
    # Create session and agent
    s3_dao.create_session(sample_session)
    s3_dao.create_agent(sample_session.session_id, sample_agent)

    # Update agent
    sample_agent.state = {"updated": "value"}
    s3_dao.update_agent(sample_session.session_id, sample_agent)

    # Verify update
    result = s3_dao.read_agent(sample_session.session_id, sample_agent.agent_id)
    assert result.state == {"updated": "value"}


def test_list_agents(s3_dao, sample_session):
    """Test listing agents in S3."""
    # Create session
    s3_dao.create_session(sample_session)

    # Create multiple agents
    agent1 = SessionAgent(agent_id="agent-1", session_id=sample_session.session_id)
    agent2 = SessionAgent(agent_id="agent-2", session_id=sample_session.session_id)

    s3_dao.create_agent(sample_session.session_id, agent1)
    s3_dao.create_agent(sample_session.session_id, agent2)

    # List agents
    agents = s3_dao.list_agents(sample_session.session_id)

    assert len(agents) == 2
    agent_ids = [a.agent_id for a in agents]
    assert "agent-1" in agent_ids
    assert "agent-2" in agent_ids


def test_create_message(s3_dao, sample_session, sample_agent, sample_message):
    """Test creating a message in S3."""
    # Create session and agent
    s3_dao.create_session(sample_session)
    s3_dao.create_agent(sample_session.session_id, sample_agent)

    # Create message
    s3_dao.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    # Verify S3 object created
    key = s3_dao._get_message_path(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)
    response = s3_dao.client.get_object(Bucket=s3_dao.bucket, Key=key)
    data = json.loads(response["Body"].read().decode("utf-8"))

    assert data["message_id"] == sample_message.message_id
    assert data["role"] == sample_message.role


def test_read_message(s3_dao, sample_session, sample_agent, sample_message):
    """Test reading a message from S3."""
    # Create session, agent, and message
    s3_dao.create_session(sample_session)
    s3_dao.create_agent(sample_session.session_id, sample_agent)
    s3_dao.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    # Read message
    result = s3_dao.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)

    assert result.message_id == sample_message.message_id
    assert result.role == sample_message.role
    assert result.content == sample_message.content


def test_list_messages_all(s3_dao, sample_session, sample_agent):
    """Test listing all messages from S3."""
    # Create session and agent
    s3_dao.create_session(sample_session)
    s3_dao.create_agent(sample_session.session_id, sample_agent)

    # Create multiple messages
    messages = []
    for i in range(5):
        message = SessionMessage(role="user", content=[ContentBlock(text=f"Message {i}")], message_id=f"message-{i}")
        messages.append(message)
        s3_dao.create_message(sample_session.session_id, sample_agent.agent_id, message)

    # List all messages
    result = s3_dao.list_messages(sample_session.session_id, sample_agent.agent_id)

    assert len(result) == 5
    message_ids = [m.message_id for m in result]
    for i in range(5):
        assert f"message-{i}" in message_ids


def test_list_messages_with_pagination(s3_dao, sample_session, sample_agent):
    """Test listing messages with pagination in S3."""
    # Create session and agent
    s3_dao.create_session(sample_session)
    s3_dao.create_agent(sample_session.session_id, sample_agent)

    # Create multiple messages
    for i in range(10):
        message = SessionMessage(
            role="user",
            content=[ContentBlock(text=f"Message {i}")],
            message_id=f"message-{i:02d}",  # Zero-padded for consistent ordering
        )
        s3_dao.create_message(sample_session.session_id, sample_agent.agent_id, message)

    # List with limit
    result = s3_dao.list_messages(sample_session.session_id, sample_agent.agent_id, limit=3)
    assert len(result) == 3

    # List with offset
    result = s3_dao.list_messages(sample_session.session_id, sample_agent.agent_id, offset=5)
    assert len(result) == 5


def test_update_message(s3_dao, sample_session, sample_agent, sample_message):
    """Test updating a message in S3."""
    # Create session, agent, and message
    s3_dao.create_session(sample_session)
    s3_dao.create_agent(sample_session.session_id, sample_agent)
    s3_dao.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    # Update message
    sample_message.content = [ContentBlock(text="Updated content")]
    s3_dao.update_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    # Verify update
    result = s3_dao.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)
    # Content is stored as dict, so access it as such
    assert result.content[0]["text"] == "Updated content"


def test_bucket_not_found_error():
    """Test handling of bucket not found errors."""
    dao = S3SessionDAO(bucket="nonexistent-bucket")
    session = Session(session_id="test", session_type=SessionType.AGENT)

    with pytest.raises(SessionException):
        dao.create_session(session)


def test_access_denied_error():
    """Test handling of access denied errors."""
    from botocore.exceptions import ClientError

    dao = S3SessionDAO(bucket="test-bucket")

    # Mock the client after creation
    with patch.object(dao, "client") as mock_client:
        mock_client.put_object.side_effect = ClientError({"Error": {"Code": "AccessDenied"}}, "PutObject")

        session = Session(session_id="test", session_type=SessionType.AGENT)

        with pytest.raises(SessionException):
            dao.create_session(session)


def test_network_error_handling():
    """Test handling of network errors."""
    from botocore.exceptions import EndpointConnectionError

    dao = S3SessionDAO(bucket="test-bucket")

    # Mock the client after creation
    with patch.object(dao, "client") as mock_client:
        mock_client.put_object.side_effect = EndpointConnectionError(endpoint_url="https://s3.amazonaws.com")

        session = Session(session_id="test", session_type=SessionType.AGENT)

        with pytest.raises(SessionException):
            dao.create_session(session)
