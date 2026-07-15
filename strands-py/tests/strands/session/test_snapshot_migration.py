"""Tests for one-time message-log -> snapshot migration via ``migrate_from``.

Migration restores an agent through the message-log manager's proven path (offset +
conversation-manager prepend) and captures a native snapshot, so a migrated session
restores identically to one written by the snapshot manager itself. These tests seed a
message-log session directly (mirroring test_repository_session_manager.py) to exercise
the compaction cases a single-turn round trip would miss.
"""

import asyncio
import copy
import tempfile
import warnings

import pytest

from strands.agent.agent import Agent
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands.agent.conversation_manager.summarizing_conversation_manager import SummarizingConversationManager
from strands.session.repository_session_manager import RepositorySessionManager
from strands.session.snapshot_session_manager import SnapshotSessionManager, _snapshot_key
from strands.storage import LocalFileStorage
from strands.types.session import Session, SessionAgent, SessionMessage, SessionType
from tests.fixtures.mock_session_repository import MockedSessionRepository
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def storage(temp_dir):
    """A file-backed unified storage for the snapshot manager."""
    return LocalFileStorage(temp_dir)


def _model(*texts):
    """Build a mock model that replies with the given texts in sequence."""
    return MockedModelProvider([{"role": "assistant", "content": [{"text": text}]} for text in texts])


def _texts(messages):
    """Flatten the text content of a message list."""
    return [content["text"] for message in messages for content in message["content"] if "text" in content]


def _seed_legacy(repository, *, session_id, agent_id, messages, conversation_manager_state, state=None):
    """Seed a message-log session directly in a repository (append-only message files)."""
    repository.create_session(Session(session_id=session_id, session_type=SessionType.AGENT))
    repository.create_agent(
        session_id,
        SessionAgent(
            agent_id=agent_id,
            state=state or {},
            conversation_manager_state=conversation_manager_state,
        ),
    )
    for index, message in enumerate(messages):
        repository.create_message(session_id, agent_id, SessionMessage.from_message(message, index))


def _legacy_manager(repository, session_id):
    """Build a deprecated message-log manager over an existing repository session."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return RepositorySessionManager(session_id=session_id, session_repository=repository)


def test_sliding_window_migration_restores_compacted_view(storage):
    """A sliding-window session that trimmed messages migrates to the compacted view, not all N.

    The old offline migrator copied all N message files with no offset, resurrecting removed
    messages. Migrating through the live manager applies removed_message_count, so only the
    surviving messages reach the snapshot.
    """
    repository = MockedSessionRepository()
    conversation_manager = SlidingWindowConversationManager()
    conversation_manager.removed_message_count = 2
    _seed_legacy(
        repository,
        session_id="sess",
        agent_id="a1",
        messages=[
            {"role": "user", "content": [{"text": "removed-0"}]},
            {"role": "assistant", "content": [{"text": "removed-1"}]},
            {"role": "user", "content": [{"text": "kept-2"}]},
            {"role": "assistant", "content": [{"text": "kept-3"}]},
        ],
        conversation_manager_state=conversation_manager.get_state(),
    )

    # Migrate: a fresh agent restores from the message log and writes a snapshot.
    manager = SnapshotSessionManager("sess", storage=storage, migrate_from=_legacy_manager(repository, "sess"))
    Agent(model=_model("x"), session_manager=manager, agent_id="a1")

    # Restore from the snapshot alone (no migrate_from) — the proof the migrated blob is correct.
    manager_2 = SnapshotSessionManager("sess", storage=storage)
    restored = Agent(model=_model("x"), session_manager=manager_2, agent_id="a1")

    tru_texts = _texts(restored.messages)
    assert tru_texts == ["kept-2", "kept-3"]
    assert "removed-0" not in tru_texts
    assert restored.conversation_manager.removed_message_count == 2


def test_summarizing_migration_preserves_summary_drops_originals(storage):
    """A summarized session migrates with the summary present and summarized originals gone.

    The old offline migrator lost the summary (it lived only in conversation-manager state, not
    a message file) and re-included the summarized originals. Migrating through the live manager
    prepends the summary and offsets the originals.
    """
    repository = MockedSessionRepository()
    conversation_manager = SummarizingConversationManager()
    conversation_manager.removed_message_count = 1
    conversation_manager._summary_message = {"role": "assistant", "content": [{"text": "SUMMARY"}]}
    _seed_legacy(
        repository,
        session_id="sess",
        agent_id="a1",
        messages=[
            {"role": "user", "content": [{"text": "summarized-away"}]},
            {"role": "user", "content": [{"text": "kept-recent"}]},
        ],
        conversation_manager_state=conversation_manager.get_state(),
    )

    manager = SnapshotSessionManager("sess", storage=storage, migrate_from=_legacy_manager(repository, "sess"))
    Agent(
        model=_model("x"),
        session_manager=manager,
        agent_id="a1",
        conversation_manager=SummarizingConversationManager(),
    )

    manager_2 = SnapshotSessionManager("sess", storage=storage)
    restored = Agent(
        model=_model("x"),
        session_manager=manager_2,
        agent_id="a1",
        conversation_manager=SummarizingConversationManager(),
    )

    tru_texts = _texts(restored.messages)
    assert tru_texts == ["SUMMARY", "kept-recent"]
    assert "summarized-away" not in tru_texts


def test_system_prompt_preserved_through_migration(storage):
    """The live agent's system prompt is captured on migration though the old format never stored it."""
    repository = MockedSessionRepository()
    _seed_legacy(
        repository,
        session_id="sess",
        agent_id="a1",
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
        conversation_manager_state=SlidingWindowConversationManager().get_state(),
    )

    manager = SnapshotSessionManager("sess", storage=storage, migrate_from=_legacy_manager(repository, "sess"))
    Agent(
        model=_model("x"),
        session_manager=manager,
        agent_id="a1",
        system_prompt="You are a helpful assistant.",
    )

    manager_2 = SnapshotSessionManager("sess", storage=storage)
    restored = Agent(model=_model("x"), session_manager=manager_2, agent_id="a1")

    assert restored.system_prompt == "You are a helpful assistant."


def test_single_turn_round_trip(storage):
    """A never-compacted single-turn session migrates and restores intact (baseline)."""
    repository = MockedSessionRepository()
    _seed_legacy(
        repository,
        session_id="sess",
        agent_id="a1",
        messages=[
            {"role": "user", "content": [{"text": "what is the answer?"}]},
            {"role": "assistant", "content": [{"text": "42"}]},
        ],
        conversation_manager_state=SlidingWindowConversationManager().get_state(),
        state={"favorite": "blue"},
    )

    manager = SnapshotSessionManager("sess", storage=storage, migrate_from=_legacy_manager(repository, "sess"))
    Agent(model=_model("x"), session_manager=manager, agent_id="a1")

    manager_2 = SnapshotSessionManager("sess", storage=storage)
    restored = Agent(model=_model("x"), session_manager=manager_2, agent_id="a1")

    assert _texts(restored.messages) == ["what is the answer?", "42"]
    assert restored.state.get("favorite") == "blue"


def test_migration_writes_snapshot_on_first_run(storage):
    """The first run with migrate_from writes a snapshot_latest for the migrated agent."""
    repository = MockedSessionRepository()
    _seed_legacy(
        repository,
        session_id="sess",
        agent_id="a1",
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
        conversation_manager_state=SlidingWindowConversationManager().get_state(),
    )

    manager = SnapshotSessionManager("sess", storage=storage, migrate_from=_legacy_manager(repository, "sess"))
    Agent(model=_model("x"), session_manager=manager, agent_id="a1")

    assert asyncio.run(storage.read(_snapshot_key("sess", "a1", snapshot_id=None))) is not None


def test_second_run_restores_from_snapshot_not_legacy(storage):
    """Once a snapshot exists, restore reads it and never consults migrate_from again."""
    repository = MockedSessionRepository()
    _seed_legacy(
        repository,
        session_id="sess",
        agent_id="a1",
        messages=[{"role": "user", "content": [{"text": "original"}]}],
        conversation_manager_state=SlidingWindowConversationManager().get_state(),
    )
    manager = SnapshotSessionManager("sess", storage=storage, migrate_from=_legacy_manager(repository, "sess"))
    Agent(model=_model("x"), session_manager=manager, agent_id="a1")

    # A second run points migrate_from at an empty legacy store. If the snapshot did not win,
    # the guard would leave the agent empty; instead it must restore "original" from the snapshot.
    empty_legacy = _legacy_manager(MockedSessionRepository(), "sess")
    manager_2 = SnapshotSessionManager("sess", storage=storage, migrate_from=empty_legacy)
    restored = Agent(model=_model("x"), session_manager=manager_2, agent_id="a1")

    assert _texts(restored.messages) == ["original"]


def test_empty_legacy_session_is_noop(storage):
    """migrate_from over a legacy store with no agent writes no snapshot and never creates records."""
    repository = MockedSessionRepository()
    manager = SnapshotSessionManager("sess", storage=storage, migrate_from=_legacy_manager(repository, "sess"))
    agent = Agent(model=_model("x"), session_manager=manager, agent_id="a1")

    assert agent.messages == []
    assert asyncio.run(storage.read(_snapshot_key("sess", "a1", snapshot_id=None))) is None
    # The read-only guard means no agent record was created in the legacy store.
    assert repository.read_agent("sess", "a1") is None


def test_migration_does_not_write_to_legacy_store(storage):
    """Migration is a pure read of the message-log store; the legacy records are untouched."""
    repository = MockedSessionRepository()
    conversation_manager = SlidingWindowConversationManager()
    conversation_manager.removed_message_count = 2
    _seed_legacy(
        repository,
        session_id="sess",
        agent_id="a1",
        messages=[
            {"role": "user", "content": [{"text": "removed-0"}]},
            {"role": "assistant", "content": [{"text": "removed-1"}]},
            {"role": "user", "content": [{"text": "kept-2"}]},
            {"role": "assistant", "content": [{"text": "kept-3"}]},
        ],
        conversation_manager_state=conversation_manager.get_state(),
    )
    agents_before = copy.deepcopy(repository.agents)
    messages_before = copy.deepcopy(repository.messages)

    manager = SnapshotSessionManager("sess", storage=storage, migrate_from=_legacy_manager(repository, "sess"))
    Agent(model=_model("x"), session_manager=manager, agent_id="a1")

    assert repository.agents == agents_before
    assert repository.messages == messages_before
