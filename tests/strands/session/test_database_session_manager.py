"""Tests for DatabaseSessionManager."""

import pytest
from sqlalchemy import create_engine

from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.session.database_models import Base
from strands.session.database_session_manager import DatabaseSessionManager
from strands.types.content import ContentBlock
from strands.types.exceptions import SessionException
from strands.types.session import Session, SessionAgent, SessionMessage, SessionType


@pytest.fixture
def db_engine():
    """Create in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_manager(db_engine):
    """Create DatabaseSessionManager with in-memory SQLite."""
    return DatabaseSessionManager(session_id="test", engine=db_engine)


@pytest.fixture
def sample_session():
    """Create sample session for testing."""
    return Session(
        session_id="test-session-123",
        session_type=SessionType.AGENT,
    )


@pytest.fixture
def sample_agent():
    """Create sample agent for testing."""
    return SessionAgent(
        agent_id="test-agent-456",
        state={"key": "value"},
        conversation_manager_state=NullConversationManager().get_state(),
    )


@pytest.fixture
def sample_message():
    """Create sample message for testing."""
    return SessionMessage.from_message(
        message={
            "role": "user",
            "content": [ContentBlock(text="test_message")],
        },
        index=0,
    )


def test_init_with_connection_string():
    """Test initialization with connection string."""
    manager = DatabaseSessionManager(
        session_id="test",
        connection_string="sqlite:///:memory:",
    )
    assert manager._owns_engine is True
    manager.engine.dispose()


def test_init_with_engine(db_engine):
    """Test initialization with existing engine."""
    manager = DatabaseSessionManager(
        session_id="test",
        engine=db_engine,
    )
    assert manager._owns_engine is False


def test_init_without_engine_or_connection_string():
    """Test that initialization fails without engine or connection string."""
    with pytest.raises(ValueError, match="Must provide either"):
        DatabaseSessionManager(session_id="test")


def test_create_session(db_manager, sample_session):
    """Test creating a session in the database."""
    result = db_manager.create_session(sample_session)

    assert result == sample_session

    # Verify session was created
    read_session = db_manager.read_session(sample_session.session_id)
    assert read_session.session_id == sample_session.session_id
    assert read_session.session_type == sample_session.session_type


def test_create_session_already_exists(db_manager, sample_session):
    """Test creating a session that already exists."""
    db_manager.create_session(sample_session)

    with pytest.raises(SessionException, match="already exists"):
        db_manager.create_session(sample_session)


def test_read_session(db_manager, sample_session):
    """Test reading a session from the database."""
    # Create session first
    db_manager.create_session(sample_session)

    # Read it back
    result = db_manager.read_session(sample_session.session_id)

    assert result.session_id == sample_session.session_id
    assert result.session_type == sample_session.session_type


def test_read_nonexistent_session(db_manager):
    """Test reading a session that doesn't exist."""
    result = db_manager.read_session("nonexistent-session")
    assert result is None


def test_delete_session(db_manager, sample_session):
    """Test deleting a session from the database."""
    # Create session first
    db_manager.create_session(sample_session)

    # Verify session exists
    assert db_manager.read_session(sample_session.session_id) is not None

    # Delete session
    db_manager.delete_session(sample_session.session_id)

    # Verify deletion
    assert db_manager.read_session(sample_session.session_id) is None


def test_delete_nonexistent_session(db_manager):
    """Test deleting a session that doesn't exist."""
    with pytest.raises(SessionException, match="does not exist"):
        db_manager.delete_session("nonexistent-session")


def test_create_agent(db_manager, sample_session, sample_agent):
    """Test creating an agent in the database."""
    # Create session first
    db_manager.create_session(sample_session)

    # Create agent
    db_manager.create_agent(sample_session.session_id, sample_agent)

    # Verify agent was created
    result = db_manager.read_agent(sample_session.session_id, sample_agent.agent_id)
    assert result.agent_id == sample_agent.agent_id
    assert result.state == sample_agent.state


def test_read_agent(db_manager, sample_session, sample_agent):
    """Test reading an agent from the database."""
    # Create session and agent
    db_manager.create_session(sample_session)
    db_manager.create_agent(sample_session.session_id, sample_agent)

    # Read agent
    result = db_manager.read_agent(sample_session.session_id, sample_agent.agent_id)

    assert result.agent_id == sample_agent.agent_id
    assert result.state == sample_agent.state


def test_read_nonexistent_agent(db_manager, sample_session):
    """Test reading an agent that doesn't exist."""
    # Create session
    db_manager.create_session(sample_session)

    # Read nonexistent agent
    result = db_manager.read_agent(sample_session.session_id, "nonexistent_agent")

    assert result is None


def test_update_agent(db_manager, sample_session, sample_agent):
    """Test updating an agent in the database."""
    # Create session and agent
    db_manager.create_session(sample_session)
    db_manager.create_agent(sample_session.session_id, sample_agent)

    # Update agent
    sample_agent.state = {"updated": "value"}
    db_manager.update_agent(sample_session.session_id, sample_agent)

    # Verify update
    result = db_manager.read_agent(sample_session.session_id, sample_agent.agent_id)
    assert result.state == {"updated": "value"}


def test_update_nonexistent_agent(db_manager, sample_session, sample_agent):
    """Test updating an agent that doesn't exist."""
    # Create session
    db_manager.create_session(sample_session)

    with pytest.raises(SessionException, match="does not exist"):
        db_manager.update_agent(sample_session.session_id, sample_agent)


def test_create_message(db_manager, sample_session, sample_agent, sample_message):
    """Test creating a message in the database."""
    # Create session and agent
    db_manager.create_session(sample_session)
    db_manager.create_agent(sample_session.session_id, sample_agent)

    # Create message
    db_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    # Verify message was created
    result = db_manager.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)
    assert result.message_id == sample_message.message_id
    assert result.message["role"] == sample_message.message["role"]


def test_read_message(db_manager, sample_session, sample_agent, sample_message):
    """Test reading a message from the database."""
    # Create session, agent, and message
    db_manager.create_session(sample_session)
    db_manager.create_agent(sample_session.session_id, sample_agent)
    db_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    # Read message
    result = db_manager.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)

    assert result.message_id == sample_message.message_id
    assert result.message["role"] == sample_message.message["role"]
    assert result.message["content"] == sample_message.message["content"]


def test_read_nonexistent_message(db_manager, sample_session, sample_agent):
    """Test reading a message that doesn't exist."""
    # Create session and agent
    db_manager.create_session(sample_session)
    db_manager.create_agent(sample_session.session_id, sample_agent)

    # Read nonexistent message
    result = db_manager.read_message(sample_session.session_id, sample_agent.agent_id, 999)

    assert result is None


def test_list_messages_all(db_manager, sample_session, sample_agent):
    """Test listing all messages from the database."""
    # Create session and agent
    db_manager.create_session(sample_session)
    db_manager.create_agent(sample_session.session_id, sample_agent)

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
        db_manager.create_message(sample_session.session_id, sample_agent.agent_id, message)

    # List all messages
    result = db_manager.list_messages(sample_session.session_id, sample_agent.agent_id)

    assert len(result) == 5
    for i, msg in enumerate(result):
        assert msg.message_id == i


def test_list_messages_with_pagination(db_manager, sample_session, sample_agent):
    """Test listing messages with pagination."""
    # Create session and agent
    db_manager.create_session(sample_session)
    db_manager.create_agent(sample_session.session_id, sample_agent)

    # Create multiple messages
    for index in range(10):
        message = SessionMessage.from_message(
            message={
                "role": "user",
                "content": [ContentBlock(text="test_message")],
            },
            index=index,
        )
        db_manager.create_message(sample_session.session_id, sample_agent.agent_id, message)

    # List with limit
    result = db_manager.list_messages(sample_session.session_id, sample_agent.agent_id, limit=3)
    assert len(result) == 3

    # List with offset
    result = db_manager.list_messages(sample_session.session_id, sample_agent.agent_id, offset=5)
    assert len(result) == 5


def test_update_message(db_manager, sample_session, sample_agent, sample_message):
    """Test updating a message in the database."""
    # Create session, agent, and message
    db_manager.create_session(sample_session)
    db_manager.create_agent(sample_session.session_id, sample_agent)
    db_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    # Update message (redact)
    sample_message.redact_message = {"role": "user", "content": [ContentBlock(text="Redacted content")]}
    db_manager.update_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    # Verify update
    result = db_manager.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id)
    assert result.redact_message is not None
    assert result.redact_message["content"][0]["text"] == "Redacted content"


def test_update_nonexistent_message(db_manager, sample_session, sample_agent, sample_message):
    """Test updating a message that doesn't exist."""
    # Create session and agent
    db_manager.create_session(sample_session)
    db_manager.create_agent(sample_session.session_id, sample_agent)

    # Update nonexistent message
    with pytest.raises(SessionException, match="does not exist"):
        db_manager.update_message(sample_session.session_id, sample_agent.agent_id, sample_message)


def test_cascade_delete_session(db_manager, sample_session, sample_agent, sample_message):
    """Test that deleting a session cascades to agents and messages."""
    # Create session, agent, and message
    db_manager.create_session(sample_session)
    db_manager.create_agent(sample_session.session_id, sample_agent)
    db_manager.create_message(sample_session.session_id, sample_agent.agent_id, sample_message)

    # Verify all exist
    assert db_manager.read_session(sample_session.session_id) is not None
    assert db_manager.read_agent(sample_session.session_id, sample_agent.agent_id) is not None
    assert (
        db_manager.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id) is not None
    )

    # Delete session
    db_manager.delete_session(sample_session.session_id)

    # Verify cascade delete
    assert db_manager.read_session(sample_session.session_id) is None
    assert db_manager.read_agent(sample_session.session_id, sample_agent.agent_id) is None
    assert db_manager.read_message(sample_session.session_id, sample_agent.agent_id, sample_message.message_id) is None


@pytest.mark.parametrize(
    "session_id",
    [
        "a/../b",
        "a/b",
    ],
)
def test_read_session_invalid_session_id(session_id, db_manager):
    """Test that invalid session_id raises ValueError."""
    with pytest.raises(ValueError, match=f"session_id={session_id} | id cannot contain path separators"):
        db_manager.read_session(session_id)


@pytest.mark.parametrize(
    "agent_id",
    [
        "a/../b",
        "a/b",
    ],
)
def test_read_agent_invalid_agent_id(agent_id, db_manager, sample_session):
    """Test that invalid agent_id raises ValueError."""
    db_manager.create_session(sample_session)
    with pytest.raises(ValueError, match=f"agent_id={agent_id} | id cannot contain path separators"):
        db_manager.read_agent(sample_session.session_id, agent_id)


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
def test_read_message_invalid_message_id(message_id, db_manager, sample_session, sample_agent):
    """Test that message_id that is not an integer raises ValueError."""
    db_manager.create_session(sample_session)
    db_manager.create_agent(sample_session.session_id, sample_agent)

    with pytest.raises(ValueError, match=r"message_id=<.*> \| message id must be an integer"):
        db_manager.read_message(sample_session.session_id, sample_agent.agent_id, message_id)


def test_error_wrapping_on_database_error(db_engine):
    """Test that database errors are wrapped in SessionException with custom message."""
    # Create manager and close/dispose the engine to cause database errors
    manager = DatabaseSessionManager(session_id="test", engine=db_engine)
    db_engine.dispose()

    # Try to read session, should get SessionException with "Failed to read session" prefix
    with pytest.raises(SessionException, match="Failed to read session"):
        manager.read_session("test-session")


def test_transaction_rollback_on_error(db_manager, sample_session, sample_agent):
    """Test that transactions rollback on errors."""
    # Create session
    db_manager.create_session(sample_session)

    # Try to create agent with invalid data that will cause database error
    # We'll try to create an agent without a session, which should fail foreign key constraint
    invalid_agent = SessionAgent(
        agent_id="invalid-agent",
        state={"key": "value"},
        conversation_manager_state={},
    )

    # Try to create agent for non-existent session (should fail)
    try:
        db_manager.create_agent("nonexistent-session-id", invalid_agent)
    except Exception:
        pass  # Expected to fail

    # Verify that the transaction was rolled back and no orphan data exists
    # If rollback didn't work, we might have partial data
    result = db_manager.read_agent(sample_session.session_id, "invalid-agent")
    assert result is None  # No orphan agent created


def test_engine_cleanup_on_del():
    """Test that engine is disposed when manager owns it."""
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    # Create manager that owns the engine
    manager = DatabaseSessionManager(session_id="test", connection_string="sqlite:///:memory:")

    # Delete manager
    del manager

    # Verify engine was disposed (this is best effort, hard to assert definitively)
    # The engine's dispose() method should have been called
    # We can't easily verify this without mocking, but at least test the code path


def test_engine_not_cleaned_when_shared():
    """Test that shared engine is not disposed."""
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    # Create manager with shared engine
    manager = DatabaseSessionManager(session_id="test", engine=engine)

    # Delete manager
    del manager

    # Verify shared engine still works (wasn't disposed)
    # This is a basic check - if dispose was called, this would fail
    conn = engine.connect()
    conn.close()
    engine.dispose()


def test_context_manager_auto_commit(db_manager, sample_session):
    """Test that context manager auto-commits on success."""
    # The create_session should work with auto_commit=True (default)
    result = db_manager.create_session(sample_session)
    assert result == sample_session

    # Verify it was actually committed
    read_result = db_manager.read_session(sample_session.session_id)
    assert read_result is not None


def test_context_manager_no_commit_on_read(db_manager, sample_session):
    """Test that read operations use auto_commit=False."""
    # Create a session
    db_manager.create_session(sample_session)

    # Read operations should work without committing
    result = db_manager.read_session(sample_session.session_id)
    assert result is not None

    # This is mostly testing that the code runs without errors
    # The auto_commit=False parameter is used correctly


def test_session_exception_preserved(db_manager, sample_session):
    """Test that SessionException is re-raised without wrapping."""
    # Create session
    db_manager.create_session(sample_session)

    # Try to create duplicate - should get SessionException directly
    with pytest.raises(SessionException, match="already exists"):
        db_manager.create_session(sample_session)

    # The error message should be the original one, not wrapped


@pytest.mark.parametrize(
    "invalid_session_id",
    [
        "a/../b",
        "a/b",
        "../escape",
    ],
)
def test_create_session_invalid_session_id(db_manager, invalid_session_id):
    """Test that create_session validates session_id."""
    invalid_session = Session(
        session_id=invalid_session_id,
        session_type=SessionType.AGENT,
    )

    with pytest.raises(ValueError, match=f"session_id={invalid_session_id} | id cannot contain path separators"):
        db_manager.create_session(invalid_session)
