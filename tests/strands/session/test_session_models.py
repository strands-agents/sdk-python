"""Tests for session models."""

from datetime import datetime
from uuid import UUID

from strands.session.session_models import Session, SessionAgent, SessionMessage, SessionType
from strands.types.content import ContentBlock


def test_create_message_with_defaults():
    """Test creating message with auto-generated fields."""
    content = [ContentBlock(text="Hello world")]
    message = SessionMessage(role="user", content=content)

    assert message.role == "user"
    assert message.content == content
    assert message.message_id is not None
    assert UUID(message.message_id)  # Validates UUID format
    assert message.created_at is not None
    assert message.updated_at is not None

    # Validate timestamp format
    datetime.fromisoformat(message.created_at.replace("Z", "+00:00"))
    datetime.fromisoformat(message.updated_at.replace("Z", "+00:00"))


def test_create_message_with_explicit_values():
    """Test creating message with explicit field values."""
    content = [ContentBlock(text="Test message")]
    message_id = "test-message-123"
    created_at = "2025-01-01T00:00:00+00:00"
    updated_at = "2025-01-01T01:00:00+00:00"

    message = SessionMessage(
        role="assistant", content=content, message_id=message_id, created_at=created_at, updated_at=updated_at
    )

    assert message.role == "assistant"
    assert message.content == content
    assert message.message_id == message_id
    assert message.created_at == created_at
    assert message.updated_at == updated_at


def test_message_to_dict():
    """Test converting message to dictionary."""
    content = [ContentBlock(text="Test")]
    message = SessionMessage(role="user", content=content)

    result = message.to_dict()

    assert isinstance(result, dict)
    assert result["role"] == "user"
    assert result["content"] == content
    assert "message_id" in result
    assert "created_at" in result
    assert "updated_at" in result


def test_message_from_dict():
    """Test creating message from dictionary."""
    data = {
        "role": "user",
        "content": [ContentBlock(text="Hello")],
        "message_id": "test-123",
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
    }

    message = SessionMessage.from_dict(data)

    assert message.role == "user"
    assert message.content == data["content"]
    assert message.message_id == "test-123"
    assert message.created_at == "2025-01-01T00:00:00+00:00"
    assert message.updated_at == "2025-01-01T00:00:00+00:00"


def test_message_to_message_conversion():
    """Test converting SessionMessage to Message type."""
    content = [ContentBlock(text="Test")]
    session_message = SessionMessage(role="user", content=content)

    message = session_message.to_message()

    assert message["role"] == "user"
    assert message["content"] == content


def test_create_agent_with_required_fields():
    """Test creating agent with required fields."""
    agent = SessionAgent(agent_id="agent-123", session_id="session-456")

    assert agent.agent_id == "agent-123"
    assert agent.session_id == "session-456"
    assert agent.state == {}
    assert agent.created_at is not None
    assert agent.updated_at is not None

    # Validate timestamp format
    datetime.fromisoformat(agent.created_at.replace("Z", "+00:00"))
    datetime.fromisoformat(agent.updated_at.replace("Z", "+00:00"))


def test_create_agent_with_state():
    """Test creating agent with state data."""
    state = {"key1": "value1", "key2": 42}
    agent = SessionAgent(agent_id="agent-123", session_id="session-456", state=state)

    assert agent.state == state


def test_agent_to_dict():
    """Test converting agent to dictionary."""
    state = {"test": "data"}
    agent = SessionAgent(agent_id="agent-123", session_id="session-456", state=state)

    result = agent.to_dict()

    assert isinstance(result, dict)
    assert result["agent_id"] == "agent-123"
    assert result["session_id"] == "session-456"
    assert result["state"] == state
    assert "created_at" in result
    assert "updated_at" in result


def test_agent_from_dict():
    """Test creating agent from dictionary."""
    data = {
        "agent_id": "agent-123",
        "session_id": "session-456",
        "state": {"key": "value"},
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
    }

    agent = SessionAgent.from_dict(data)

    assert agent.agent_id == "agent-123"
    assert agent.session_id == "session-456"
    assert agent.state == {"key": "value"}
    assert agent.created_at == "2025-01-01T00:00:00+00:00"
    assert agent.updated_at == "2025-01-01T00:00:00+00:00"


def test_create_session_with_required_fields():
    """Test creating session with required fields."""
    session = Session(session_id="session-123", session_type=SessionType.AGENT)

    assert session.session_id == "session-123"
    assert session.session_type == SessionType.AGENT
    assert session.created_at is not None
    assert session.updated_at is not None

    # Validate timestamp format
    datetime.fromisoformat(session.created_at.replace("Z", "+00:00"))
    datetime.fromisoformat(session.updated_at.replace("Z", "+00:00"))


def test_session_type_enum():
    """Test session type enumeration."""
    session = Session(session_id="test", session_type=SessionType.AGENT)

    assert session.session_type == SessionType.AGENT
    assert session.session_type.value == "AGENT"


def test_session_to_dict():
    """Test converting session to dictionary."""
    session = Session(session_id="session-123", session_type=SessionType.AGENT)

    result = session.to_dict()

    assert isinstance(result, dict)
    assert result["session_id"] == "session-123"
    assert result["session_type"] == SessionType.AGENT
    assert "created_at" in result
    assert "updated_at" in result


def test_session_from_dict():
    """Test creating session from dictionary."""
    data = {
        "session_id": "session-123",
        "session_type": SessionType.AGENT,
        "created_at": "2025-01-01T00:00:00+00:00",
        "updated_at": "2025-01-01T00:00:00+00:00",
    }

    session = Session.from_dict(data)

    assert session.session_id == "session-123"
    assert session.session_type == SessionType.AGENT
    assert session.created_at == "2025-01-01T00:00:00+00:00"
    assert session.updated_at == "2025-01-01T00:00:00+00:00"
