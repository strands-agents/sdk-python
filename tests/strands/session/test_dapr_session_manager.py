"""Tests for DaprSessionManager."""

from typing import Any, Optional
from unittest.mock import Mock

import pytest
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.session.dapr_session_manager import (
    DAPR_CONSISTENCY_EVENTUAL,
    DAPR_CONSISTENCY_STRONG,
    DaprSessionManager,
)
from strands.types.content import ContentBlock
from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage, SessionType


class FakeDaprClient:
    """Sync fake Dapr client for testing."""

    def __init__(self) -> None:
        """Initialize fake client with in-memory state."""
        self._state: dict[str, bytes] = {}
        self._closed = False

    def get_state(
        self,
        store_name: str,
        key: str,
        state_metadata: Optional[dict[str, str]] = None,
        **kwargs: Any,
    ) -> Mock:
        """Get state from in-memory store."""
        response = Mock()
        response.data = self._state.get(key)
        return response

    def save_state(
        self,
        store_name: str,
        key: str,
        value: str | bytes,
        state_metadata: Optional[dict[str, str]] = None,
        options: Any = None,
    ) -> None:
        """Save state to in-memory store."""
        if isinstance(value, str):
            self._state[key] = value.encode("utf-8")
        else:
            self._state[key] = value

    def delete_state(
        self,
        store_name: str,
        key: str,
        state_metadata: Optional[dict[str, str]] = None,
        options: Any = None,
    ) -> None:
        """Delete state from in-memory store."""
        self._state.pop(key, None)

    def close(self) -> None:
        """Close the client."""
        self._closed = True


@pytest.fixture
def fake_dapr_client() -> FakeDaprClient:
    """Create fake Dapr client for testing."""
    return FakeDaprClient()


@pytest.fixture
def dapr_manager(fake_dapr_client: FakeDaprClient) -> DaprSessionManager:
    """Create DaprSessionManager for testing."""
    return DaprSessionManager(
        session_id="test", state_store_name="statestore", dapr_client=fake_dapr_client, consistency=DAPR_CONSISTENCY_EVENTUAL
    )


@pytest.fixture
def sample_session() -> Session:
    """Create sample session for testing."""
    return Session(session_id="test-session", session_type=SessionType.AGENT)


@pytest.fixture
def sample_agent() -> SessionAgent:
    """Create sample agent for testing."""
    return SessionAgent(
        agent_id="test-agent", state={"key": "value"}, conversation_manager_state=NullConversationManager().get_state()
    )


@pytest.fixture
def sample_message() -> SessionMessage:
    """Create sample message for testing."""
    return SessionMessage.from_message(
        message={
            "role": "user",
            "content": [ContentBlock(text="Hello world")],
        },
        index=0,
    )


def test_consistency_constants():
    """Test consistency constants are properly defined."""
    assert DAPR_CONSISTENCY_EVENTUAL == "eventual"
    assert DAPR_CONSISTENCY_STRONG == "strong"


def test_messages_shape_non_list_handling(dapr_manager: DaprSessionManager, sample_session: Session, sample_agent: SessionAgent):
    """Seed a non-list messages payload and verify graceful handling and overwrite by create_message."""
    dapr_manager.create_session(sample_session)
    dapr_manager.create_agent(sample_session.session_id, sample_agent)

    # Manually write an invalid messages payload (object instead of list)
    messages_key = dapr_manager._get_messages_key(sample_session.session_id, sample_agent.agent_id)
    # Directly mutate internal client state for test
    assert hasattr(dapr_manager._dapr_client, "_state")
    dapr_manager._dapr_client._state[messages_key] = b'{"messages": {}}'

    # list_messages should return empty
    assert dapr_manager.list_messages(sample_session.session_id, sample_agent.agent_id) == []

    # Now create a message; this should overwrite messages with a proper list
    new_msg = SessionMessage.from_message({"role": "user", "content": [ContentBlock(text="こんにちは 😃")]}, 0)
    dapr_manager.create_message(sample_session.session_id, sample_agent.agent_id, new_msg)

    listed = dapr_manager.list_messages(sample_session.session_id, sample_agent.agent_id)
    assert len(listed) == 1
    assert "こんにちは" in str(listed[0].message["content"])  # unicode round-trip


def test_init_with_consistency_levels(fake_dapr_client: FakeDaprClient):
    """Test initialization with different consistency levels."""
    # Test eventual consistency
    manager_eventual = DaprSessionManager(
        session_id="test",
        state_store_name="statestore",
        dapr_client=fake_dapr_client,
        consistency=DAPR_CONSISTENCY_EVENTUAL,
    )
    assert manager_eventual._consistency == DAPR_CONSISTENCY_EVENTUAL

    # Test strong consistency
    manager_strong = DaprSessionManager(
        session_id="test",
        state_store_name="statestore",
        dapr_client=fake_dapr_client,
        consistency=DAPR_CONSISTENCY_STRONG,
    )
    assert manager_strong._consistency == DAPR_CONSISTENCY_STRONG


def test_init_with_ttl(fake_dapr_client: FakeDaprClient):
    """Test initialization with TTL."""
    manager = DaprSessionManager(
        session_id="test", state_store_name="statestore", dapr_client=fake_dapr_client, ttl=3600
    )
    assert manager._ttl == 3600


def test_create_session(dapr_manager: DaprSessionManager, sample_session: Session):
    """Test creating a session."""
    dapr_manager.create_session(sample_session)

    # Verify session was stored
    result = dapr_manager.read_session(sample_session.session_id)
    assert result is not None
    assert result.session_id == sample_session.session_id
    assert result.session_type == sample_session.session_type


def test_create_session_already_exists(dapr_manager: DaprSessionManager, sample_session: Session):
    """Test creating a session that already exists."""
    dapr_manager.create_session(sample_session)

    # Try to create again
    with pytest.raises(SessionException, match="already exists"):
        dapr_manager.create_session(sample_session)


def test_read_session(dapr_manager: DaprSessionManager, sample_session: Session):
    """Test reading a session."""
    dapr_manager.create_session(sample_session)
    result = dapr_manager.read_session(sample_session.session_id)

    assert result is not None
    assert result.session_id == sample_session.session_id
    assert result.session_type == sample_session.session_type


def test_read_nonexistent_session(dapr_manager: DaprSessionManager):
    """Test reading a session that doesn't exist."""
    result = dapr_manager.read_session("nonexistent-session")
    assert result is None


def test_create_agent(dapr_manager: DaprSessionManager, sample_session: Session, sample_agent: SessionAgent):
    """Test creating an agent in a session."""
    dapr_manager.create_session(sample_session)
    dapr_manager.create_agent(sample_session.session_id, sample_agent)

    # Verify agent was stored
    result = dapr_manager.read_agent(sample_session.session_id, sample_agent.agent_id)
    assert result is not None
    assert result.agent_id == sample_agent.agent_id
    assert result.state == sample_agent.state


def test_read_agent(dapr_manager: DaprSessionManager, sample_session: Session, sample_agent: SessionAgent):
    """Test reading an agent from a session."""
    dapr_manager.create_session(sample_session)
    dapr_manager.create_agent(sample_session.session_id, sample_agent)

    result = dapr_manager.read_agent(sample_session.session_id, sample_agent.agent_id)

    assert result is not None
    assert result.agent_id == sample_agent.agent_id
    assert result.state == sample_agent.state


def test_read_nonexistent_agent(dapr_manager: DaprSessionManager, sample_session: Session):
    """Test reading an agent that doesn't exist."""
    result = dapr_manager.read_agent(sample_session.session_id, "nonexistent_agent")
    assert result is None


def test_update_agent(dapr_manager: DaprSessionManager, sample_session: Session, sample_agent: SessionAgent):
    """Test updating an agent."""
    dapr_manager.create_session(sample_session)
    dapr_manager.create_agent(sample_session.session_id, sample_agent)

    # Update agent
    sample_agent.state = {"updated": "value"}
    dapr_manager.update_agent(sample_session.session_id, sample_agent)

    # Verify update
    result = dapr_manager.read_agent(sample_session.session_id, sample_agent.agent_id)
    assert result is not None
    assert result.state == {"updated": "value"}


def test_update_nonexistent_agent(
    dapr_manager: DaprSessionManager, sample_session: Session, sample_agent: SessionAgent
):
    """Test updating an agent that doesn't exist."""
    dapr_manager.create_session(sample_session)

    # Try to update non-existent agent
    with pytest.raises(SessionException, match="does not exist"):
        dapr_manager.update_agent(sample_session.session_id, sample_agent)


def test_create_message(
    dapr_manager: DaprSessionManager,
    sample_session: Session,
    sample_agent: SessionAgent,
    sample_message: SessionMessage,
):
    """Test creating a message for an agent."""
    dapr_manager.create_session(sample_session)
    dapr_manager.create_agent(sample_session.session_id, sample_agent)

    dapr_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    # Verify message was stored
    result = dapr_manager.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)
    assert result is not None
    assert result.message_id == sample_message.message_id


def test_read_message(
    dapr_manager: DaprSessionManager,
    sample_session: Session,
    sample_agent: SessionAgent,
    sample_message: SessionMessage,
):
    """Test reading a message."""
    dapr_manager.create_session(sample_session)
    dapr_manager.create_agent(sample_session.session_id, sample_agent)
    dapr_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    # Create additional message
    sample_message_2 = SessionMessage.from_message(
        message={
            "role": "assistant",
            "content": [ContentBlock(text="Hi there")],
        },
        index=1,
    )
    dapr_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message_2)

    # Read specific message
    result = dapr_manager.read_message(sample_session.session_id, sample_agent.agent_id, 1)

    assert result is not None
    assert result.message_id == 1
    assert result.message["role"] == "assistant"


def test_read_nonexistent_message(
    dapr_manager: DaprSessionManager, sample_session: Session, sample_agent: SessionAgent
):
    """Test reading a message that doesn't exist."""
    dapr_manager.create_session(sample_session)
    dapr_manager.create_agent(sample_session.session_id, sample_agent)

    result = dapr_manager.read_message(sample_session.session_id, sample_agent.agent_id, 999)
    assert result is None


def test_list_messages_all(dapr_manager: DaprSessionManager, sample_session: Session, sample_agent: SessionAgent):
    """Test listing all messages for an agent."""
    dapr_manager.create_session(sample_session)
    dapr_manager.create_agent(sample_session.session_id, sample_agent)

    # Create multiple messages
    for i in range(5):
        message = SessionMessage(
            message={
                "role": "user",
                "content": [ContentBlock(text=f"Message {i}")],
            },
            message_id=i,
        )
        dapr_manager.create_message(sample_session.session_id, sample_agent.agent_id, message)

    # List all messages
    result = dapr_manager.list_messages(sample_session.session_id, sample_agent.agent_id)

    assert len(result) == 5


def test_list_messages_with_limit(
    dapr_manager: DaprSessionManager, sample_session: Session, sample_agent: SessionAgent
):
    """Test listing messages with limit."""
    dapr_manager.create_session(sample_session)
    dapr_manager.create_agent(sample_session.session_id, sample_agent)

    # Create multiple messages
    for i in range(10):
        message = SessionMessage(
            message={
                "role": "user",
                "content": [ContentBlock(text=f"Message {i}")],
            },
            message_id=i,
        )
        dapr_manager.create_message(sample_session.session_id, sample_agent.agent_id, message)

    # List with limit
    result = dapr_manager.list_messages(sample_session.session_id, sample_agent.agent_id, limit=3)

    assert len(result) == 3


def test_list_messages_with_offset(
    dapr_manager: DaprSessionManager, sample_session: Session, sample_agent: SessionAgent
):
    """Test listing messages with offset."""
    dapr_manager.create_session(sample_session)
    dapr_manager.create_agent(sample_session.session_id, sample_agent)

    # Create multiple messages
    for i in range(10):
        message = SessionMessage(
            message={
                "role": "user",
                "content": [ContentBlock(text=f"Message {i}")],
            },
            message_id=i,
        )
        dapr_manager.create_message(sample_session.session_id, sample_agent.agent_id, message)

    # List with offset
    result = dapr_manager.list_messages(sample_session.session_id, sample_agent.agent_id, offset=5)

    assert len(result) == 5
    assert result[0].message_id == 5


def test_list_messages_empty(dapr_manager: DaprSessionManager, sample_session: Session, sample_agent: SessionAgent):
    """Test listing messages when none exist."""
    dapr_manager.create_session(sample_session)
    dapr_manager.create_agent(sample_session.session_id, sample_agent)

    result = dapr_manager.list_messages(sample_session.session_id, sample_agent.agent_id)

    assert len(result) == 0


def test_update_message(
    dapr_manager: DaprSessionManager,
    sample_session: Session,
    sample_agent: SessionAgent,
    sample_message: SessionMessage,
):
    """Test updating a message."""
    dapr_manager.create_session(sample_session)
    dapr_manager.create_agent(sample_session.session_id, sample_agent)
    dapr_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    # Update message
    sample_message.message["content"] = [ContentBlock(text="Updated content")]
    dapr_manager.update_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    # Verify update
    result = dapr_manager.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)
    assert result is not None
    assert result.message["content"][0]["text"] == "Updated content"


def test_update_nonexistent_message(
    dapr_manager: DaprSessionManager,
    sample_session: Session,
    sample_agent: SessionAgent,
    sample_message: SessionMessage,
):
    """Test updating a message that doesn't exist."""
    dapr_manager.create_session(sample_session)
    dapr_manager.create_agent(sample_session.session_id, sample_agent)

    # Try to update non-existent message
    with pytest.raises(SessionException, match="does not exist"):
        dapr_manager.update_message(sample_session.session_id, sample_agent.agent_id, sample_message)


def test_corrupted_json(dapr_manager: DaprSessionManager, fake_dapr_client: FakeDaprClient):
    """Test handling of corrupted JSON data."""
    # Store invalid JSON (use string key to match how _get_session_key works)
    fake_dapr_client._state["test:session"] = b"invalid json content"

    # Should raise SessionException
    with pytest.raises(SessionException, match="Invalid JSON"):
        dapr_manager.read_session("test")


@pytest.mark.parametrize(
    "session_id",
    [
        "a/../b",
        "a/b",
    ],
)
def test_invalid_session_id(session_id: str, fake_dapr_client: FakeDaprClient):
    """Test that session IDs with path separators are rejected."""
    manager = DaprSessionManager(session_id="test", state_store_name="statestore", dapr_client=fake_dapr_client)

    with pytest.raises(ValueError, match=f"session_id={session_id} | id cannot contain path separators"):
        manager._get_session_key(session_id)


@pytest.mark.parametrize(
    "agent_id",
    [
        "a/../b",
        "a/b",
    ],
)
def test_invalid_agent_id(agent_id: str, dapr_manager: DaprSessionManager):
    """Test that agent IDs with path separators are rejected."""
    with pytest.raises(ValueError, match=f"agent_id={agent_id} | id cannot contain path separators"):
        dapr_manager._get_agent_key("session1", agent_id)


@pytest.mark.parametrize(
    "message_id",
    [
        "../../../secret",
        "../../attack",
        "../escape",
        "path/traversal",
        "not_an_int",
        None,
        [],
    ],
)
def test_invalid_message_id(message_id: Any, dapr_manager: DaprSessionManager, sample_session: Session):
    """Test that non-integer message IDs are rejected."""
    with pytest.raises(ValueError, match=r"message_id=<.*> \| message id must be an integer"):
        dapr_manager.read_message(sample_session.session_id, "agent1", message_id)


def test_client_ownership(fake_dapr_client: FakeDaprClient):
    """Test client ownership tracking."""
    # Manager created with external client
    manager = DaprSessionManager(session_id="test", state_store_name="statestore", dapr_client=fake_dapr_client)
    assert manager._owns_client is False

    # Manager created with from_address
    # We can't test actual client creation without mocking, but we can verify the flag
    assert hasattr(manager, "_owns_client")


def test_close_owned_client(fake_dapr_client: FakeDaprClient):
    """Test closing owned client."""
    # Create manager with existing session
    fake_dapr_client._state["test-close:session"] = b'{"session_id": "test-close", "session_type": "AGENT"}'
    manager = DaprSessionManager(session_id="test-close", state_store_name="statestore", dapr_client=fake_dapr_client)
    manager._owns_client = True

    manager.close()
    assert fake_dapr_client._closed is True


def test_close_unowned_client(fake_dapr_client: FakeDaprClient):
    """Test not closing unowned client."""
    # Create manager with existing session
    fake_dapr_client._state["test-close2:session"] = b'{"session_id": "test-close2", "session_type": "AGENT"}'
    manager = DaprSessionManager(session_id="test-close2", state_store_name="statestore", dapr_client=fake_dapr_client)
    manager._owns_client = False

    manager.close()
    assert fake_dapr_client._closed is False


def test_delete_session_parity(fake_dapr_client: FakeDaprClient):
    """Test delete_session removes session, agents, messages and manifest."""
    manager = DaprSessionManager(session_id="sess", state_store_name="statestore", dapr_client=fake_dapr_client)
    # Create agent and messages (session already created by __init__ if needed)
    agent = SessionAgent(agent_id="a1", state={}, conversation_manager_state={})
    manager.create_agent("sess", agent)
    manager.create_message(
        "sess", "a1", SessionMessage.from_message({"role": "user", "content": [ContentBlock(text="hi")]}, 0)
    )

    # Sanity
    assert manager.read_session("sess") is not None
    assert manager.read_agent("sess", "a1") is not None
    assert len(manager.list_messages("sess", "a1")) == 1

    # Delete
    manager.delete_session("sess")

    # All gone
    assert manager.read_session("sess") is None
    assert manager.read_agent("sess", "a1") is None
    assert manager.list_messages("sess", "a1") == []
