"""Tests for SnapshotSessionManager."""

import tempfile
from unittest.mock import AsyncMock

import pytest

from strands.agent.agent import Agent
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands.session.snapshot_session_manager import (
    SnapshotSessionManager,
    _new_snapshot_id,
    _session_prefix,
    _snapshot_key,
)
from strands.storage import LocalFileStorage
from strands.types.content import ContentBlock
from strands.types.exceptions import ContextWindowOverflowException
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def storage(temp_dir):
    """A file-backed unified storage."""
    return LocalFileStorage(temp_dir)


def _model(*texts):
    """Build a mock model that replies with the given texts in sequence."""
    return MockedModelProvider([{"role": "assistant", "content": [{"text": text}]} for text in texts])


def test_new_session_starts_empty(storage):
    """A brand-new session leaves a fresh agent's messages untouched."""
    manager = SnapshotSessionManager("s1", storage=storage)
    agent = Agent(model=_model("hi"), session_manager=manager, agent_id="a1")

    assert agent.messages == []


def test_restore_across_instances(storage):
    """A fresh agent with the same session id rehydrates prior conversation."""
    manager = SnapshotSessionManager("s1", storage=storage)
    agent = Agent(model=_model("The answer is 42."), session_manager=manager, agent_id="a1")
    agent("What is the answer?")

    # Simulate process restart: new manager + new agent over the same storage.
    manager_2 = SnapshotSessionManager("s1", storage=storage)
    agent_2 = Agent(model=_model("Still 42."), session_manager=manager_2, agent_id="a1")

    tru_texts = [content["text"] for message in agent_2.messages for content in message["content"] if "text" in content]
    assert "What is the answer?" in tru_texts
    assert "The answer is 42." in tru_texts


def test_restore_warns_on_overwrite(storage, caplog):
    """Restoring over an agent that already had messages logs a warning."""
    manager = SnapshotSessionManager("s1", storage=storage)
    agent = Agent(model=_model("saved"), session_manager=manager, agent_id="a1")
    agent("first turn")

    manager_2 = SnapshotSessionManager("s1", storage=storage)
    Agent(
        model=_model("x"),
        session_manager=manager_2,
        agent_id="a1",
        messages=[{"role": "user", "content": [{"text": "pre-existing"}]}],
    )

    assert "overwritten by session restore" in caplog.text


def test_state_round_trips(storage):
    """Agent state persists and restores across instances."""
    manager = SnapshotSessionManager("s1", storage=storage)
    agent = Agent(model=_model("ok"), session_manager=manager, agent_id="a1")
    agent.state.set("favorite", "blue")
    agent("remember my favorite")

    manager_2 = SnapshotSessionManager("s1", storage=storage)
    agent_2 = Agent(model=_model("ok"), session_manager=manager_2, agent_id="a1")

    assert agent_2.state.get("favorite") == "blue"


def test_system_prompt_round_trips(storage):
    """The system prompt persists and restores (session preset opt-in)."""
    manager = SnapshotSessionManager("s1", storage=storage)
    agent = Agent(
        model=_model("ok"), session_manager=manager, agent_id="a1", system_prompt="You are a helpful assistant."
    )
    agent("hi")

    manager_2 = SnapshotSessionManager("s1", storage=storage)
    agent_2 = Agent(model=_model("ok"), session_manager=manager_2, agent_id="a1")

    assert agent_2.system_prompt == "You are a helpful assistant."


def test_bytes_content_round_trips(storage):
    """Image bytes in messages survive JSON serialization via base64 encoding."""
    manager = SnapshotSessionManager("s1", storage=storage)
    agent = Agent(model=_model("saw it"), session_manager=manager, agent_id="a1")
    image_block: ContentBlock = {"image": {"format": "png", "source": {"bytes": b"\x89PNG\r\n\x1a\n"}}}
    agent([image_block])

    manager_2 = SnapshotSessionManager("s1", storage=storage)
    agent_2 = Agent(model=_model("x"), session_manager=manager_2, agent_id="a1")

    tru_bytes = agent_2.messages[0]["content"][0]["image"]["source"]["bytes"]
    assert tru_bytes == b"\x89PNG\r\n\x1a\n"


@pytest.mark.asyncio
async def test_save_latest_on_message_writes_each_message(temp_dir):
    """The ``message`` strategy persists after every message added."""
    storage = LocalFileStorage(temp_dir)
    save_keys = []
    original = storage.write

    async def _spy(key, data, **kwargs):
        save_keys.append(key)
        await original(key, data, **kwargs)

    storage.write = _spy  # type: ignore[method-assign]

    manager = SnapshotSessionManager("s1", storage=storage, save_latest_on="message")
    agent = Agent(model=_model("reply"), session_manager=manager, agent_id="a1")
    save_keys.clear()
    await agent.invoke_async("hello")

    # Exactly two messages were added (user + assistant); each triggered one latest save, and
    # the "message" strategy adds no extra invocation-end save.
    assert len(save_keys) == 2
    assert all(key.endswith("snapshot_latest.json") for key in save_keys)


@pytest.mark.asyncio
async def test_invocation_strategy_saves_once_not_per_message(temp_dir):
    """The default ``invocation`` strategy saves once at invocation end, not per message."""
    storage = LocalFileStorage(temp_dir)
    save_keys = []
    original = storage.write

    async def _spy(key, data, **kwargs):
        save_keys.append(key)
        await original(key, data, **kwargs)

    storage.write = _spy  # type: ignore[method-assign]

    # Default save_latest_on="invocation": MessageAddedEvent must not be registered.
    manager = SnapshotSessionManager("s1", storage=storage)
    agent = Agent(model=_model("reply"), session_manager=manager, agent_id="a1")
    save_keys.clear()
    await agent.invoke_async("hello")

    # One save at invocation end, despite two messages being added during the turn.
    assert len(save_keys) == 1
    assert save_keys[0].endswith("snapshot_latest.json")


def test_snapshot_trigger_creates_immutable(storage):
    """When the trigger fires, an immutable snapshot is appended."""
    manager = SnapshotSessionManager("s1", storage=storage, snapshot_trigger=lambda *, agent, **_: True)
    agent = Agent(model=_model("turn one"), session_manager=manager, agent_id="a1")
    agent("go")

    import asyncio

    ids = asyncio.run(manager.list_snapshot_ids(agent))
    assert len(ids) == 1


def test_time_travel_restore(storage):
    """restore_snapshot rewinds an agent to an earlier immutable checkpoint."""
    manager = SnapshotSessionManager("s1", storage=storage, snapshot_trigger=lambda *, agent, **_: True)
    agent = Agent(model=_model("first", "second"), session_manager=manager, agent_id="a1")
    agent("turn 1")
    agent("turn 2")

    import asyncio

    ids = asyncio.run(manager.list_snapshot_ids(agent))
    assert len(ids) == 2

    restored = asyncio.run(manager.restore_snapshot(agent, snapshot_id=ids[0]))
    assert restored is True

    tru_texts = [content["text"] for message in agent.messages for content in message["content"] if "text" in content]
    assert "turn 1" in tru_texts
    assert "turn 2" not in tru_texts


def _stateful_model(*texts):
    """Build a mock model that reports itself as stateful (server-managed history)."""
    model = _model(*texts)
    # Stateful models manage conversation history server-side; the constructor swaps in a
    # NullConversationManager for them, so both the saved and restored agents match.
    object.__setattr__(model, "_force_stateful", True)
    type(model).stateful = property(lambda self: getattr(self, "_force_stateful", False))
    return model


def test_stateful_model_discards_restored_messages(storage):
    """Restore keeps model_state but drops messages for a stateful model."""
    original = _stateful_model("hi")
    try:
        manager = SnapshotSessionManager("s1", storage=storage)
        agent = Agent(model=original, session_manager=manager, agent_id="a1")
        # Persist a snapshot that contains messages (a stateful model normally clears
        # local history mid-turn, so set them explicitly to exercise the discard branch).
        agent.messages = [{"role": "user", "content": [{"text": "hello"}]}]
        manager.sync_agent(agent)

        manager_2 = SnapshotSessionManager("s1", storage=storage)
        agent_2 = Agent(model=_stateful_model("x"), session_manager=manager_2, agent_id="a1")
        assert agent_2.messages == []
    finally:
        del type(original).stateful


def test_redaction_flush_persists_redacted_content(temp_dir):
    """A guardrail redaction is flushed to the latest snapshot immediately."""
    storage = LocalFileStorage(temp_dir)
    manager = SnapshotSessionManager("s1", storage=storage)
    redaction_model = MockedModelProvider(
        [{"redactedUserContent": "REDACTED", "redactedAssistantContent": "I can't help with that."}]
    )
    agent = Agent(model=redaction_model, session_manager=manager, agent_id="a1")
    agent("sensitive prompt")

    manager_2 = SnapshotSessionManager("s1", storage=storage)
    agent_2 = Agent(model=_model("x"), session_manager=manager_2, agent_id="a1")

    tru_texts = [content["text"] for message in agent_2.messages for content in message["content"] if "text" in content]
    assert "sensitive prompt" not in tru_texts
    assert "REDACTED" in tru_texts


@pytest.mark.asyncio
async def test_delete_session_removes_snapshots(storage):
    """delete_session clears persisted snapshots."""
    manager = SnapshotSessionManager("s1", storage=storage, snapshot_trigger=lambda *, agent, **_: True)
    agent = Agent(model=_model("hi"), session_manager=manager, agent_id="a1")
    agent("go")

    await manager.delete_session()

    assert await storage.read(_snapshot_key("s1", "a1", snapshot_id=None)) is None
    assert await storage.list(_session_prefix("s1")) == []


def test_restore_by_id_missing_returns_false(storage):
    """Restoring a non-existent immutable snapshot returns False."""
    manager = SnapshotSessionManager("s1", storage=storage)
    agent = Agent(model=_model("hi"), session_manager=manager, agent_id="a1")

    import asyncio

    assert asyncio.run(manager.restore_snapshot(agent, snapshot_id=_new_snapshot_id())) is False


def test_trigger_strategy_skips_latest_without_trigger(temp_dir):
    """Under ``trigger`` strategy with no trigger, nothing is persisted on invocation."""
    storage = LocalFileStorage(temp_dir)
    storage.write = AsyncMock()  # type: ignore[method-assign]

    manager = SnapshotSessionManager("s1", storage=storage, save_latest_on="trigger")
    agent = Agent(model=_model("hi"), session_manager=manager, agent_id="a1")
    storage.write.reset_mock()
    agent("go")

    storage.write.assert_not_called()


def test_no_warning_when_restoring_into_empty_agent(storage, caplog):
    """Restoring into a fresh agent with no messages does not log the overwrite warning."""
    manager = SnapshotSessionManager("s1", storage=storage)
    agent = Agent(model=_model("saved"), session_manager=manager, agent_id="a1")
    agent("first turn")

    # The second agent starts empty, so restore should populate it silently.
    manager_2 = SnapshotSessionManager("s1", storage=storage)
    agent_2 = Agent(model=_model("x"), session_manager=manager_2, agent_id="a1")

    assert len(agent_2.messages) > 0
    assert "overwritten by session restore" not in caplog.text


class _OverflowThenAnswerModel(MockedModelProvider):
    """A model that raises a context-overflow on its first stream, then answers normally.

    This drives the Agent's reactive overflow-recovery path (reduce_context), which in turn
    makes the direct ``session_manager.sync_agent`` call at the overflow catch site.
    """

    def __init__(self, *texts):
        super().__init__([{"role": "assistant", "content": [{"text": text}]} for text in texts])
        self._overflowed = False

    async def stream(self, *args, **kwargs):
        if not self._overflowed:
            self._overflowed = True
            raise ContextWindowOverflowException("Input is too long for requested model")
        async for event in super().stream(*args, **kwargs):
            yield event


def test_context_overflow_syncs_reduced_conversation(storage):
    """A context-window overflow persists the reduced conversation via the direct sync_agent call.

    On ContextWindowOverflowException the Agent calls ``session_manager.sync_agent(agent)``
    directly (outside the hook system) after trimming context. To prove this specific path —
    and not the invocation-end save — the manager runs under ``save_latest_on="trigger"`` with
    no trigger, so ``_on_after_invocation`` writes nothing and ``MessageAddedEvent`` is not
    registered. The only thing that can persist ``snapshot_latest`` is the overflow-time
    ``sync_agent``. Deleting the direct call at the overflow catch site would fail this test.
    """
    # window_size=2 forces the reactive trim to actually drop the seeded backlog.
    conversation_manager = SlidingWindowConversationManager(window_size=2)
    manager = SnapshotSessionManager("s1", storage=storage, save_latest_on="trigger")

    sync_calls = []
    original_sync = manager.sync_agent

    def _spy_sync(agent, **kwargs):
        sync_calls.append(len(agent.messages))
        return original_sync(agent, **kwargs)

    manager.sync_agent = _spy_sync  # type: ignore[method-assign]

    seeded = [
        {"role": "user", "content": [{"text": "one"}]},
        {"role": "assistant", "content": [{"text": "1"}]},
        {"role": "user", "content": [{"text": "two"}]},
        {"role": "assistant", "content": [{"text": "2"}]},
    ]
    agent = Agent(
        model=_OverflowThenAnswerModel("recovered."),
        session_manager=manager,
        agent_id="a1",
        conversation_manager=conversation_manager,
        messages=seeded,
    )
    agent("three")

    # The overflow catch site invoked sync_agent exactly once, on the trimmed conversation.
    assert len(sync_calls) == 1

    # A fresh instance restores only what the overflow-path sync persisted: the reduced
    # window, not the full seeded backlog (proving the reduced conversation was the thing saved).
    manager_2 = SnapshotSessionManager("s1", storage=storage, save_latest_on="trigger")
    agent_2 = Agent(
        model=_model("x"),
        session_manager=manager_2,
        agent_id="a1",
        conversation_manager=SlidingWindowConversationManager(window_size=2),
    )

    tru_texts = [content["text"] for message in agent_2.messages for content in message["content"] if "text" in content]
    assert "one" not in tru_texts  # the oldest seeded messages were trimmed before the sync
    assert tru_texts  # but the reduced conversation was persisted (not empty)


def test_redaction_flushes_even_under_trigger_strategy(temp_dir):
    """A guardrail redaction is flushed immediately even under the ``trigger`` strategy.

    This is a deliberate divergence from the TypeScript SDK. TS gates its redaction flush on an
    AfterModelCall hook that it does not register under ``saveLatestOn: 'trigger'``, so TS does not
    flush redactions under that strategy. Python has no redaction signal on AfterModelCallEvent;
    redaction arrives through the Agent's direct ``redact_latest_message`` call, which always
    persists so pre-redaction content never sits at rest. We assert the safer always-flush here.
    """
    storage = LocalFileStorage(temp_dir)
    manager = SnapshotSessionManager("s1", storage=storage, save_latest_on="trigger")
    redaction_model = MockedModelProvider(
        [{"redactedUserContent": "REDACTED", "redactedAssistantContent": "I can't help with that."}]
    )
    agent = Agent(model=redaction_model, session_manager=manager, agent_id="a1")
    agent("sensitive prompt")

    manager_2 = SnapshotSessionManager("s1", storage=storage, save_latest_on="trigger")
    agent_2 = Agent(model=_model("x"), session_manager=manager_2, agent_id="a1")

    tru_texts = [content["text"] for message in agent_2.messages for content in message["content"] if "text" in content]
    assert "sensitive prompt" not in tru_texts
    assert "REDACTED" in tru_texts


def test_snapshot_trigger_returning_false_appends_nothing(storage):
    """A present trigger that returns False creates no immutable snapshot and receives the agent."""
    seen_agents = []

    def trigger(*, agent, **kwargs):
        seen_agents.append(agent)
        return False

    manager = SnapshotSessionManager("s1", storage=storage, snapshot_trigger=trigger)
    agent = Agent(model=_model("hi"), session_manager=manager, agent_id="a1")
    agent("go")

    import asyncio

    assert asyncio.run(manager.list_snapshot_ids(agent)) == []
    # The trigger was invoked with the agent as a keyword argument.
    assert seen_agents and seen_agents[0] is agent


@pytest.mark.asyncio
async def test_list_snapshot_ids_pagination(storage):
    """limit and start_after page the immutable id list; invalid start_after raises."""
    manager = SnapshotSessionManager("s1", storage=storage, snapshot_trigger=lambda *, agent, **_: True)
    agent = Agent(model=_model("a", "b", "c"), session_manager=manager, agent_id="a1")
    agent("one")
    agent("two")
    agent("three")

    all_ids = await manager.list_snapshot_ids(agent)
    assert len(all_ids) == 3
    assert all_ids == sorted(all_ids)

    assert await manager.list_snapshot_ids(agent, limit=2) == all_ids[:2]
    assert await manager.list_snapshot_ids(agent, limit=0) == []
    assert await manager.list_snapshot_ids(agent, start_after=all_ids[0]) == all_ids[1:]

    with pytest.raises(ValueError, match="not a valid snapshot id"):
        await manager.list_snapshot_ids(agent, start_after="not-an-id")


@pytest.mark.asyncio
async def test_restore_snapshot_rejects_malformed_id(storage):
    """restore_snapshot with a malformed id raises rather than silently missing."""
    manager = SnapshotSessionManager("s1", storage=storage)
    agent = Agent(model=_model("hi"), session_manager=manager, agent_id="a1")

    with pytest.raises(ValueError, match="not a valid snapshot id"):
        await manager.restore_snapshot(agent, snapshot_id="../escape")


@pytest.mark.asyncio
async def test_delete_session_is_scoped_to_its_own_session(temp_dir):
    """delete_session removes only its session, leaving other sessions and keys intact."""
    storage = LocalFileStorage(temp_dir)

    manager_a = SnapshotSessionManager("sess-a", storage=storage)
    Agent(model=_model("a"), session_manager=manager_a, agent_id="a1")("hi")
    manager_b = SnapshotSessionManager("sess-b", storage=storage)
    Agent(model=_model("b"), session_manager=manager_b, agent_id="a1")("hi")
    # An unrelated key from another subsystem sharing the same storage.
    await storage.write("memory/note.json", b"keep me")

    await manager_a.delete_session()

    assert await storage.read(_snapshot_key("sess-a", "a1", snapshot_id=None)) is None
    assert await storage.read(_snapshot_key("sess-b", "a1", snapshot_id=None)) is not None
    assert await storage.read("memory/note.json") == b"keep me"


def test_stateful_model_restore_keeps_model_state(storage):
    """Restoring a stateful-model session drops messages but preserves model_state."""
    original = _stateful_model("hi")
    try:
        manager = SnapshotSessionManager("s1", storage=storage)
        agent = Agent(model=original, session_manager=manager, agent_id="a1")
        agent.messages = [{"role": "user", "content": [{"text": "hello"}]}]
        agent._model_state = {"response_id": "resp-123"}
        manager.sync_agent(agent)

        manager_2 = SnapshotSessionManager("s1", storage=storage)
        agent_2 = Agent(model=_stateful_model("x"), session_manager=manager_2, agent_id="a1")

        assert agent_2.messages == []  # messages dropped for the stateful model
        assert agent_2._model_state == {"response_id": "resp-123"}  # but model_state survives
    finally:
        del type(original).stateful
