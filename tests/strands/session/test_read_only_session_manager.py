"""Tests for ReadOnlySessionManager wrapper."""

from unittest.mock import Mock, patch

import pytest

from strands.agent.agent import Agent
from strands.session.read_only_session_manager import ReadOnlySessionManager
from strands.session.repository_session_manager import RepositorySessionManager
from strands.types.content import ContentBlock
from strands.types.session import Session, SessionAgent, SessionMessage, SessionType
from tests.fixtures.mock_session_repository import MockedSessionRepository


@pytest.fixture
def mock_repository():
    """Create a mock repository."""
    return MockedSessionRepository()


@pytest.fixture
def inner_session_manager(mock_repository):
    """Create an inner read-write session manager."""
    return RepositorySessionManager(
        session_id="test-session",
        session_repository=mock_repository,
    )


@pytest.fixture
def read_only_session_manager(inner_session_manager):
    """Create a read-only wrapper around the inner session manager."""
    return ReadOnlySessionManager(inner_session_manager)


@pytest.fixture
def existing_read_only_session_manager(mock_repository):
    """Create a read-only wrapper with a pre-existing session."""
    session = Session(session_id="test-session", session_type=SessionType.AGENT)
    mock_repository.create_session(session)
    inner = RepositorySessionManager(
        session_id="test-session",
        session_repository=mock_repository,
    )
    return ReadOnlySessionManager(inner)


def test_initialize_delegates_to_inner(existing_read_only_session_manager, mock_repository):
    """Test that initialize restores agent state from the inner session manager."""
    session_agent = SessionAgent(
        agent_id="test-agent",
        state={"key": "value"},
        conversation_manager_state={
            "__name__": "SlidingWindowConversationManager",
            "removed_message_count": 0,
        },
    )
    mock_repository.create_agent("test-session", session_agent)
    mock_repository.create_message(
        "test-session",
        "test-agent",
        SessionMessage(message={"role": "user", "content": [ContentBlock(text="Hello")]}, message_id=0),
    )

    agent = Agent(agent_id="test-agent")
    existing_read_only_session_manager.initialize(agent)

    assert agent.state.get("key") == "value"
    assert len(agent.messages) == 1
    assert agent.messages[0]["content"][0]["text"] == "Hello"


def test_write_methods_are_noop(read_only_session_manager):
    """Test that all write methods are no-ops and don't raise."""
    agent = Mock(agent_id="test-agent")
    source = Mock()

    read_only_session_manager.append_message({"role": "user", "content": []}, agent)
    read_only_session_manager.redact_latest_message({"role": "user", "content": []}, agent)
    read_only_session_manager.sync_agent(agent)
    read_only_session_manager.sync_multi_agent(source)
    read_only_session_manager.append_bidi_message({"role": "user", "content": []}, agent)
    read_only_session_manager.sync_bidi_agent(agent)


def test_hooks_do_not_call_inner_write_methods(inner_session_manager):
    """Test that hooks fire the wrapper's no-ops, not the inner's write methods."""
    with (
        patch.object(inner_session_manager, "append_message") as mock_append,
        patch.object(inner_session_manager, "sync_agent") as mock_sync,
    ):
        ro = ReadOnlySessionManager(inner_session_manager)
        Agent(agent_id="test-agent", session_manager=ro)

        mock_append.assert_not_called()
        mock_sync.assert_not_called()


def test_messages_not_persisted_via_hooks(read_only_session_manager, mock_repository):
    """Test that messages are not persisted when hooks fire through the wrapper."""
    Agent(agent_id="test-agent", session_manager=read_only_session_manager)

    messages = mock_repository.list_messages("test-session", "test-agent")
    assert len(messages) == 0


def test_direct_write_calls_are_noop(read_only_session_manager, mock_repository):
    """Test that direct calls to write methods don't persist."""
    agent = Agent(agent_id="test-agent", session_manager=read_only_session_manager)

    agent.messages.append({"role": "user", "content": [{"text": "test"}]})
    read_only_session_manager.sync_agent(agent)

    session_agent = mock_repository.read_agent("test-session", "test-agent")
    assert session_agent.state == {}


def test_multi_agent_initialize_delegates(read_only_session_manager):
    """Test that multi-agent initialize delegates to inner."""
    mock_multi_agent = Mock()
    mock_multi_agent.id = "test-multi-agent"
    mock_multi_agent.serialize_state.return_value = {"id": "test-multi-agent", "state": {}}

    read_only_session_manager.initialize_multi_agent(mock_multi_agent)


def test_getattr_forwards_to_inner(read_only_session_manager, inner_session_manager):
    """Test that attribute access is forwarded to the inner session manager."""
    assert read_only_session_manager.session_id == inner_session_manager.session_id
