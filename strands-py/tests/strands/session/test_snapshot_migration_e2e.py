"""Exhaustive end-to-end verification of message-log -> snapshot migration.

Unlike test_snapshot_migration.py (which seeds a repository directly), these tests drive the
real Agent lifecycle end to end: a live agent runs against a real message-log manager on real
storage, then a second agent migrates it via ``migrate_from`` on a real snapshot backend.

The central proof is a *lossless round trip*: for each scenario we capture the legacy agent's
own post-run live state and assert the migrated snapshot restores to a byte-identical agent —
same messages (including persisted tracking_ids and metadata), state, system prompt,
conversation-manager removal count, interrupt state, and model state. If migration drops or
alters any of that, these tests fail.
"""

import tempfile
import warnings

import boto3
import pytest
from moto import mock_aws

from strands import Agent
from strands.agent.conversation_manager.sliding_window_conversation_manager import (
    SlidingWindowConversationManager,
)
from strands.agent.conversation_manager.summarizing_conversation_manager import (
    SummarizingConversationManager,
)
from strands.session import SnapshotSessionManager
from strands.session.file_session_manager import FileSessionManager
from strands.session.s3_session_manager import S3SessionManager
from strands.storage import InMemoryStorage, LocalFileStorage, S3Storage
from strands.storage.storage import Storage
from strands.types.content import ContentBlock, Message
from strands.types.exceptions import ContextWindowOverflowException
from tests.fixtures.mocked_model_provider import MockedModelProvider

# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------


def _model(*responses: Message) -> MockedModelProvider:
    """Mock model that replies with the given assistant messages in sequence."""
    return MockedModelProvider(list(responses))


class _OverflowOnceModel(MockedModelProvider):
    """Replays canned responses but raises a context overflow on the Nth ``stream`` call.

    Overflow drives the summarizing conversation manager's ``reduce_context`` (its only path
    to compaction), which then calls ``model.stream()`` again to generate the summary — served
    from the same canned sequence.
    """

    def __init__(self, responses: list[Message], *, overflow_on_call: int):
        super().__init__(responses)
        self._overflow_on_call = overflow_on_call
        self._call = 0

    async def stream(self, *args, **kwargs):
        self._call += 1
        if self._call == self._overflow_on_call:
            raise ContextWindowOverflowException("Input is too long for requested model")
        async for event in super().stream(*args, **kwargs):
            yield event


def _text(text: str) -> Message:
    return {"role": "assistant", "content": [{"text": text}]}


def _texts(messages: list[Message]) -> list[str]:
    return [content["text"] for message in messages for content in message["content"] if "text" in content]


def _file_legacy(storage_dir: str, session_id: str) -> FileSessionManager:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return FileSessionManager(session_id=session_id, storage_dir=storage_dir)


def _state_signature(agent: Agent) -> dict:
    """A comparable snapshot of everything migration must preserve."""
    return {
        "messages": agent.messages,
        "state": agent.state.get(),
        "system_prompt": agent.system_prompt,
        "removed_message_count": agent.conversation_manager.removed_message_count,
        "interrupt_state": agent._interrupt_state.to_dict(),
        "model_state": agent._model_state,
    }


@pytest.fixture
def temp_dirs():
    """Two isolated temp dirs: one for the legacy store, one for the snapshot store."""
    with tempfile.TemporaryDirectory() as legacy_dir, tempfile.TemporaryDirectory() as snap_dir:
        yield legacy_dir, snap_dir


# ----------------------------------------------------------------------------------------
# Core lossless round trip: migrated state == the legacy agent's own restored state
# ----------------------------------------------------------------------------------------


def _run_scenario_and_compare(
    *,
    legacy_dir: str,
    snap_storage: Storage,
    conversation_manager_factory,
    system_prompt: str | None,
    responses: list[Message],
    prompts: list[str],
    initial_state: dict | None = None,
    model_factory=None,
):
    """Run a scenario against the legacy manager, migrate, and assert lossless reproduction.

    The gold-standard proof: migration must reproduce *the exact agent being migrated*. We
    capture the legacy agent's own post-run live state (its compacted messages carry the
    tracking_ids and metadata persisted in the message log) and assert the migrated snapshot
    restores to a byte-identical agent. Comparing to a separate native run would not work —
    per-message tracking_ids are unique per run — but a lossless round trip of the *same* run
    is the stronger correctness statement.

    ``model_factory`` overrides how the legacy agent's model is built (e.g. to raise a
    context overflow that triggers a summarizing manager's compaction); it defaults to a plain
    replay model over ``responses``. The two post-migration agents always use empty replay
    models since they only restore.

    Returns (source_signature, migrated_signature) for further per-scenario assertions.
    """
    session_id = "sess"
    agent_id = "agent"

    # Run against the legacy message-log manager, then snapshot its own live state as the
    # source of truth migration must reproduce.
    legacy = _file_legacy(legacy_dir, session_id)
    legacy_agent = Agent(
        model=model_factory() if model_factory is not None else _model(*responses),
        session_manager=legacy,
        agent_id=agent_id,
        conversation_manager=conversation_manager_factory(),
        system_prompt=system_prompt,
    )
    if initial_state:
        for key, value in initial_state.items():
            legacy_agent.state.set(key, value)
    for prompt in prompts:
        legacy_agent(prompt)
    source_sig = _state_signature(legacy_agent)

    # Migrate on first run, then restore from the snapshot alone.
    migrating = SnapshotSessionManager(
        session_id, storage=snap_storage, migrate_from=_file_legacy(legacy_dir, session_id)
    )
    Agent(
        model=_model(),
        session_manager=migrating,
        agent_id=agent_id,
        conversation_manager=conversation_manager_factory(),
        system_prompt=system_prompt,
    )
    migrated_restored = Agent(
        model=_model(),
        session_manager=SnapshotSessionManager(session_id, storage=snap_storage),
        agent_id=agent_id,
        conversation_manager=conversation_manager_factory(),
        system_prompt=system_prompt,
    )
    migrated_sig = _state_signature(migrated_restored)

    return source_sig, migrated_sig


def test_no_compaction_round_trips_losslessly(temp_dirs):
    """A short session round-trips losslessly through migration."""
    legacy_dir, snap_dir = temp_dirs
    source, migrated = _run_scenario_and_compare(
        legacy_dir=legacy_dir,
        snap_storage=LocalFileStorage(snap_dir),
        conversation_manager_factory=lambda: SlidingWindowConversationManager(),
        system_prompt="You are helpful.",
        responses=[_text("a1"), _text("a2")],
        prompts=["hello", "again"],
        initial_state={"favorite": "blue"},
    )
    assert migrated == source
    assert _texts(migrated["messages"]) == ["hello", "a1", "again", "a2"]
    assert migrated["state"] == {"favorite": "blue"}
    assert migrated["system_prompt"] == "You are helpful."


def test_sliding_window_compaction_round_trips_losslessly(temp_dirs):
    """A sliding-window session that physically trimmed round-trips losslessly through migration."""
    legacy_dir, snap_dir = temp_dirs
    source, migrated = _run_scenario_and_compare(
        legacy_dir=legacy_dir,
        snap_storage=LocalFileStorage(snap_dir),
        conversation_manager_factory=lambda: SlidingWindowConversationManager(window_size=2),
        system_prompt="Trim me.",
        responses=[_text("r1"), _text("r2"), _text("r3")],
        prompts=["one", "two", "three"],
    )
    assert migrated == source
    # window_size=2 keeps only the last user+assistant pair.
    assert _texts(migrated["messages"]) == ["three", "r3"]
    assert migrated["removed_message_count"] == 4


def test_summarizing_compaction_round_trips_losslessly(temp_dirs):
    """A summarizing session (summary lives only in CM state) round-trips losslessly through migration."""
    legacy_dir, snap_dir = temp_dirs

    def cm_factory():
        # Aggressively summarize: keep few recent, summarize the rest on overflow.
        return SummarizingConversationManager(summary_ratio=0.5, preserve_recent_messages=2)

    # Overflow on the 4th turn's model call drives reduce_context → summarization; the manager
    # then calls the model again to generate the summary, and the turn is retried — all served
    # from this replay sequence.
    def model_factory():
        return _OverflowOnceModel(
            [_text("r1"), _text("r2"), _text("r3"), _text("summary of earlier turns"), _text("r4")],
            overflow_on_call=4,
        )

    source, migrated = _run_scenario_and_compare(
        legacy_dir=legacy_dir,
        snap_storage=LocalFileStorage(snap_dir),
        conversation_manager_factory=cm_factory,
        system_prompt=None,
        responses=[],
        prompts=["one", "two", "three", "four"],
        model_factory=model_factory,
    )
    # Guard that summarization actually fired — otherwise `source` carries no summary and the
    # equality below would pass vacuously without exercising summary preservation.
    assert source["removed_message_count"] > 0
    # The strongest assertion: the migrated messages equal the native snapshot's messages,
    # so if the summary were dropped or originals resurrected this fails.
    assert migrated == source


# ----------------------------------------------------------------------------------------
# Content-shape edge cases
# ----------------------------------------------------------------------------------------


def test_tool_use_messages_survive_migration(temp_dirs):
    """A conversation containing a tool-use / tool-result pair migrates intact."""
    legacy_dir, snap_dir = temp_dirs
    session_id, agent_id = "sess", "agent"

    tool_use_response: Message = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "calc", "input": {"x": 2}}}],
    }
    final_response = _text("done")

    from strands import tool

    @tool
    def calc(x: int) -> int:
        """Double a number."""
        return x * 2

    legacy = _file_legacy(legacy_dir, session_id)
    legacy_agent = Agent(
        model=_model(tool_use_response, final_response),
        session_manager=legacy,
        agent_id=agent_id,
        tools=[calc],
    )
    legacy_agent("use the tool")
    legacy_texts = _texts(legacy_agent.messages)

    migrating = SnapshotSessionManager(
        session_id, storage=LocalFileStorage(snap_dir), migrate_from=_file_legacy(legacy_dir, session_id)
    )
    Agent(model=_model(), session_manager=migrating, agent_id=agent_id, tools=[calc])

    restored = Agent(
        model=_model(),
        session_manager=SnapshotSessionManager(session_id, storage=LocalFileStorage(snap_dir)),
        agent_id=agent_id,
        tools=[calc],
    )
    # Tool-use and tool-result content blocks are preserved (a valid, loadable history).
    tru_has_tool_use = any("toolUse" in content for message in restored.messages for content in message["content"])
    tru_has_tool_result = any(
        "toolResult" in content for message in restored.messages for content in message["content"]
    )
    assert tru_has_tool_use
    assert tru_has_tool_result
    assert _texts(restored.messages) == legacy_texts


def test_bytes_content_survives_migration(temp_dirs):
    """Binary content (image bytes) survives migration via base64 framing."""
    legacy_dir, snap_dir = temp_dirs
    session_id, agent_id = "sess", "agent"

    legacy = _file_legacy(legacy_dir, session_id)
    agent = Agent(model=_model(_text("saw it")), session_manager=legacy, agent_id=agent_id)
    image_block: ContentBlock = {"image": {"format": "png", "source": {"bytes": b"\x89PNG\r\n\x1a\n"}}}
    agent([image_block])

    migrating = SnapshotSessionManager(
        session_id, storage=LocalFileStorage(snap_dir), migrate_from=_file_legacy(legacy_dir, session_id)
    )
    Agent(model=_model(), session_manager=migrating, agent_id=agent_id)

    restored = Agent(
        model=_model(),
        session_manager=SnapshotSessionManager(session_id, storage=LocalFileStorage(snap_dir)),
        agent_id=agent_id,
    )
    tru_bytes = restored.messages[0]["content"][0]["image"]["source"]["bytes"]
    assert tru_bytes == b"\x89PNG\r\n\x1a\n"


# ----------------------------------------------------------------------------------------
# Lifecycle / idempotency / safety
# ----------------------------------------------------------------------------------------


def test_second_run_ignores_legacy_and_continues(temp_dirs):
    """After migration, a new turn appends to the snapshot; the legacy store is irrelevant."""
    legacy_dir, snap_dir = temp_dirs
    session_id, agent_id = "sess", "agent"

    legacy = _file_legacy(legacy_dir, session_id)
    Agent(model=_model(_text("r1")), session_manager=legacy, agent_id=agent_id)(  # noqa: E501
        "first"
    )

    # Run 1: migrate + a new turn.
    migrating = SnapshotSessionManager(
        session_id, storage=LocalFileStorage(snap_dir), migrate_from=_file_legacy(legacy_dir, session_id)
    )
    agent = Agent(model=_model(_text("r2")), session_manager=migrating, agent_id=agent_id)
    agent("second")

    # Run 2: no migrate_from at all — must restore the full continued conversation.
    restored = Agent(
        model=_model(),
        session_manager=SnapshotSessionManager(session_id, storage=LocalFileStorage(snap_dir)),
        agent_id=agent_id,
    )
    assert _texts(restored.messages) == ["first", "r1", "second", "r2"]


def test_migration_is_idempotent(temp_dirs):
    """Constructing the migrating manager twice does not corrupt or duplicate state."""
    legacy_dir, snap_dir = temp_dirs
    session_id, agent_id = "sess", "agent"

    legacy = _file_legacy(legacy_dir, session_id)
    Agent(model=_model(_text("r1")), session_manager=legacy, agent_id=agent_id)("only turn")

    for _ in range(3):
        migrating = SnapshotSessionManager(
            session_id, storage=LocalFileStorage(snap_dir), migrate_from=_file_legacy(legacy_dir, session_id)
        )
        agent = Agent(model=_model(), session_manager=migrating, agent_id=agent_id)

    assert _texts(agent.messages) == ["only turn", "r1"]


def test_empty_legacy_is_noop_and_writes_nothing(temp_dirs):
    """migrate_from over a legacy store with no agent leaves a fresh agent empty and writes no snapshot."""
    import asyncio

    from strands.session.snapshot_session_manager import _snapshot_key

    legacy_dir, snap_dir = temp_dirs
    snap = LocalFileStorage(snap_dir)
    migrating = SnapshotSessionManager("sess", storage=snap, migrate_from=_file_legacy(legacy_dir, "sess"))
    agent = Agent(model=_model(), session_manager=migrating, agent_id="agent")

    assert agent.messages == []
    assert asyncio.run(snap.read(_snapshot_key("sess", "agent", snapshot_id=None))) is None


def test_migration_leaves_legacy_store_on_disk_untouched(temp_dirs):
    """Migration is read-only: the on-disk legacy message files are unchanged afterward."""
    import os

    legacy_dir, snap_dir = temp_dirs
    session_id, agent_id = "sess", "agent"

    legacy = _file_legacy(legacy_dir, session_id)
    agent = Agent(
        model=_model(_text("r1"), _text("r2"), _text("r3")),
        session_manager=legacy,
        agent_id=agent_id,
        conversation_manager=SlidingWindowConversationManager(window_size=2),
    )
    agent("one")
    agent("two")
    agent("three")

    msg_dir = os.path.join(legacy_dir, f"session_{session_id}", "agents", f"agent_{agent_id}", "messages")

    def _fingerprint():
        return {name: os.path.getsize(os.path.join(msg_dir, name)) for name in sorted(os.listdir(msg_dir))}

    before = _fingerprint()
    migrating = SnapshotSessionManager(
        session_id, storage=LocalFileStorage(snap_dir), migrate_from=_file_legacy(legacy_dir, session_id)
    )
    Agent(
        model=_model(),
        session_manager=migrating,
        agent_id=agent_id,
        conversation_manager=SlidingWindowConversationManager(window_size=2),
    )
    after = _fingerprint()

    assert before == after
    # All message files persist (append-only log is never truncated on migration).
    assert len(after) == 6


# ----------------------------------------------------------------------------------------
# Cross-backend
# ----------------------------------------------------------------------------------------


def test_migration_into_in_memory_storage(temp_dirs):
    """Migration works into an InMemoryStorage backend (no filesystem for the snapshot)."""
    legacy_dir, _ = temp_dirs
    session_id, agent_id = "sess", "agent"

    legacy = _file_legacy(legacy_dir, session_id)
    Agent(model=_model(_text("r1")), session_manager=legacy, agent_id=agent_id)("hi")

    snap = InMemoryStorage()
    migrating = SnapshotSessionManager(session_id, storage=snap, migrate_from=_file_legacy(legacy_dir, session_id))
    Agent(model=_model(), session_manager=migrating, agent_id=agent_id)

    restored = Agent(
        model=_model(), session_manager=SnapshotSessionManager(session_id, storage=snap), agent_id=agent_id
    )
    assert _texts(restored.messages) == ["hi", "r1"]


@mock_aws
def test_migration_s3_legacy_to_s3_storage():
    """A real S3 message-log session (moto) migrates into an S3Storage snapshot backend."""
    session_id, agent_id = "sess", "agent"
    bucket = "migration-bucket"
    boto3.client("s3", region_name="us-west-2").create_bucket(
        Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": "us-west-2"}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = S3SessionManager(session_id=session_id, bucket=bucket, prefix="legacy", region_name="us-west-2")
    agent = Agent(
        model=_model(_text("r1"), _text("r2")),
        session_manager=legacy,
        agent_id=agent_id,
        conversation_manager=SlidingWindowConversationManager(window_size=2),
    )
    agent("one")
    agent("two")
    expected = _texts(agent.messages)

    def _legacy_ro():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return S3SessionManager(session_id=session_id, bucket=bucket, prefix="legacy", region_name="us-west-2")

    snap = S3Storage(bucket=bucket, prefix="snapshots", region_name="us-west-2")
    migrating = SnapshotSessionManager(session_id, storage=snap, migrate_from=_legacy_ro())
    Agent(
        model=_model(),
        session_manager=migrating,
        agent_id=agent_id,
        conversation_manager=SlidingWindowConversationManager(window_size=2),
    )

    restored = Agent(
        model=_model(),
        session_manager=SnapshotSessionManager(
            session_id, storage=S3Storage(bucket=bucket, prefix="snapshots", region_name="us-west-2")
        ),
        agent_id=agent_id,
        conversation_manager=SlidingWindowConversationManager(window_size=2),
    )
    assert _texts(restored.messages) == expected
