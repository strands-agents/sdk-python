"""Snapshot-based session manager.

Persists an agent as a single versioned :class:`~strands.types._snapshot.Snapshot`
blob on each lifecycle event, mirroring the TypeScript SDK's ``SessionManager``:

- A mutable ``snapshot_latest`` is overwritten on each save, for crash/restart resume.
- Append-only immutable snapshots (time-ordered keys) are written when a ``snapshot_trigger``
  fires, enabling checkpointing — restore to any prior state, not just the latest.

The manager persists snapshots through the unified :class:`~strands.storage.storage.Storage`
primitive (``write``/``read``/``delete``/``list`` over byte blobs). It owns the key layout,
snapshot-id scheme, and serialization; the storage backend only moves bytes, so the same
``Storage`` instance can back sessions, memory, and other subsystems.

This is distinct from the older message-log session managers
(:class:`~strands.session.repository_session_manager.RepositorySessionManager` and its
subclasses), which persist each message individually. Snapshots capture the whole agent
in one atomic blob and are the recommended path for new agents.
"""

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args, runtime_checkable

from .._async import run_async
from .._identifier import Identifier
from .._identifier import validate as validate_identifier
from ..hooks.events import (
    AfterInvocationEvent,
    AfterMultiAgentInvocationEvent,
    AfterNodeCallEvent,
    AgentInitializedEvent,
    MessageAddedEvent,
    MultiAgentInitializedEvent,
)
from ..hooks.registry import HookRegistry
from ..storage.storage import _NAMESPACED, Storage, _NamespacedStorage
from ..types._snapshot import Snapshot
from ..types.content import Message
from ..types.exceptions import SnapshotException
from ..types.session import decode_bytes_values, encode_bytes_values
from ._snapshot_id import new_snapshot_id as _new_snapshot_id
from ._snapshot_id import validate_snapshot_id as _validate_snapshot_id
from .repository_session_manager import RepositorySessionManager
from .session_manager import SessionManager

if TYPE_CHECKING:
    from ..agent.agent import Agent
    from ..multiagent.base import MultiAgentBase

logger = logging.getLogger(__name__)

SaveLatestStrategy = Literal["message", "invocation", "trigger"]
"""Controls how often ``snapshot_latest`` is saved automatically.

- ``"invocation"``: after every agent invocation completes (default; balances durability and I/O).
- ``"message"``: after every message added (most durable, highest I/O).
- ``"trigger"``: only when ``snapshot_trigger`` fires (or manually via ``sync_agent``).

Guardrail redactions are flushed immediately under every strategy, including ``"trigger"``,
so pre-redaction content never sits at rest. This diverges from the TypeScript SDK, which
does not flush redactions under ``"trigger"``; see :meth:`SnapshotSessionManager.redact_latest_message`.
"""

# Derived from the Literal above so the accepted runtime values cannot drift from the type.
_SAVE_LATEST_STRATEGIES = get_args(SaveLatestStrategy)

# Top-level storage namespace for all session data. Byte-identical to the TypeScript SDK,
# which namespaces its unified storage under "session" (singular) before the session id, so
# the on-disk key layout is shared across SDKs. The manager applies this namespace once (unless
# the caller passed an already-namespaced view) and builds keys relative to it, so a caller who
# pre-namespaces under "session" does not get a doubled "session/session/..." prefix.
_SESSIONS_NAMESPACE = "session"

_SNAPSHOT_LATEST = "snapshot_latest.json"
_IMMUTABLE_HISTORY = "immutable_history"
_SNAPSHOT_REGEX = re.compile(r"snapshot_([\w-]+)\.json\Z")

# Cap concurrent deletes so a session with many immutable checkpoints does not spawn thousands
# of simultaneous blocking storage calls (e.g. S3's per-object delete via asyncio.to_thread).
_DELETE_CONCURRENCY = 100

# -- Key layout, relative to the "session" storage namespace applied in __init__ --
#
#   <session_id>/scopes/agent/<agent_id>/snapshots/
#     snapshot_latest.json
#     immutable_history/snapshot_<id>.json
#
# The namespaced storage view prepends "session/", so the full on-disk key is
# session/<session_id>/... — byte-identical to the TypeScript SDK. These are module-level so the
# migration utility builds the same keys the manager reads.


def _resolve_storage(storage: Storage) -> Storage:
    """Namespace raw storage under ``"session"``; pass an already-namespaced view through.

    Mirrors the TypeScript SDK: a view the caller already scoped (marked with ``_NAMESPACED``)
    is used as-is so its prefix is not doubled, otherwise raw storage is wrapped under the
    ``"session"`` namespace. Manager keys are built relative to the result.
    """
    if getattr(storage, "_namespaced", None) is _NAMESPACED:
        return storage
    return _NamespacedStorage(storage, _SESSIONS_NAMESPACE)


def _session_prefix(session_id: str) -> str:
    """Return the namespace-relative key prefix covering an entire session."""
    session_id = validate_identifier(session_id, Identifier.SESSION)
    return f"{session_id}/"


def _snapshots_prefix(session_id: str, agent_id: str) -> str:
    """Return the namespace-relative key prefix for an agent's snapshots directory."""
    agent_id = validate_identifier(agent_id, Identifier.AGENT)
    return f"{_session_prefix(session_id)}scopes/agent/{agent_id}/snapshots/"


def _snapshot_key(session_id: str, agent_id: str, *, snapshot_id: str | None) -> str:
    """Return the storage key for a snapshot; ``None`` targets ``snapshot_latest``."""
    prefix = _snapshots_prefix(session_id, agent_id)
    if snapshot_id is None:
        return f"{prefix}{_SNAPSHOT_LATEST}"
    _validate_snapshot_id(snapshot_id)
    return f"{prefix}{_IMMUTABLE_HISTORY}/snapshot_{snapshot_id}.json"


def _multi_agent_key(session_id: str, orchestrator_id: str) -> str:
    """Return the storage key for a multi-agent orchestrator's latest snapshot.

    Under the ``multiAgent`` scope, byte-identical to the TypeScript SDK. Orchestrators persist
    latest-only (no immutable-history / time-travel), matching TS.
    """
    session_id = validate_identifier(session_id, Identifier.SESSION)
    orchestrator_id = validate_identifier(orchestrator_id, Identifier.AGENT)
    return f"{session_id}/scopes/multiAgent/{orchestrator_id}/snapshots/{_SNAPSHOT_LATEST}"


def _encode_json(obj: dict[str, Any]) -> bytes:
    """JSON-encode a dict to UTF-8 bytes, base64-encoding any bytes values."""
    return json.dumps(encode_bytes_values(obj), ensure_ascii=False).encode("utf-8")


def _decode_json_object(data: bytes, *, description: str) -> dict[str, Any]:
    """Decode JSON bytes into a dict, base64-decoding any bytes values.

    Args:
        data: The stored bytes.
        description: What the blob is, interpolated into error messages (e.g. ``"snapshot"``).

    Raises:
        SnapshotException: If the bytes are not a JSON object (malformed JSON, wrong encoding, or a
            non-object top-level value). Translating here keeps a corrupt or truncated stored blob
            from surfacing a raw decode error out of the agent constructor's restore-on-init path.
    """
    try:
        decoded = decode_bytes_values(json.loads(data))
    except (ValueError, UnicodeDecodeError) as error:
        raise SnapshotException(f"Failed to deserialize {description}: {error}") from error
    if not isinstance(decoded, dict):
        raise SnapshotException(f"{description} is not an object: got {type(decoded).__name__}")
    return decoded


def _serialize_snapshot(snapshot: Snapshot) -> bytes:
    """Serialize a snapshot to JSON bytes, base64-encoding any bytes content."""
    return _encode_json(snapshot.to_dict())


def _deserialize_snapshot(data: bytes) -> Snapshot:
    """Deserialize JSON bytes into a snapshot, decoding any base64 bytes content.

    Raises:
        SnapshotException: If the bytes are not a valid, current-schema snapshot (malformed
            JSON, wrong encoding, wrong shape, or an unsupported schema/scope).
    """
    decoded = _decode_json_object(data, description="snapshot")
    try:
        return Snapshot.from_dict(decoded)
    except (KeyError, TypeError, AttributeError) as error:
        raise SnapshotException(f"Failed to deserialize snapshot: {error}") from error


def _serialize_multi_agent(orchestrator_id: str, state: dict[str, Any]) -> bytes:
    """Serialize an orchestrator's ``serialize_state()`` dict to JSON bytes, base64-encoding bytes.

    The orchestrator id is stamped alongside the state (mirroring the TypeScript SDK) so restore
    can reject a snapshot that belongs to a different orchestrator; see
    :func:`_deserialize_multi_agent`. Stamping at this boundary — rather than trusting an id inside
    the subclass's ``serialize_state()`` — guards every ``MultiAgentBase`` subclass uniformly.
    """
    return _encode_json({"orchestrator_id": orchestrator_id, "state": state})


def _deserialize_multi_agent(data: bytes, *, expected_orchestrator_id: str) -> dict[str, Any]:
    """Deserialize orchestrator-state JSON bytes back into a state dict, decoding base64 bytes.

    Args:
        data: The stored snapshot bytes.
        expected_orchestrator_id: The id of the orchestrator being restored into. A snapshot whose
            stamped id differs is rejected so state is never loaded into the wrong orchestrator.

    Raises:
        SnapshotException: If the bytes are not valid orchestrator-state JSON (malformed, wrong
            encoding, or wrong shape), or the stamped orchestrator id does not match
            ``expected_orchestrator_id``.
    """
    decoded = _decode_json_object(data, description="multi-agent snapshot")
    stored_orchestrator_id = decoded.get("orchestrator_id")
    if stored_orchestrator_id != expected_orchestrator_id:
        raise SnapshotException(
            "Multi-agent snapshot orchestrator id mismatch: "
            f"expected {expected_orchestrator_id!r}, got {stored_orchestrator_id!r}"
        )
    state = decoded.get("state")
    if not isinstance(state, dict):
        raise SnapshotException(f"Multi-agent snapshot state is not an object: got {type(state).__name__}")
    return state


@runtime_checkable
class SnapshotTrigger(Protocol):
    """Decides whether to write an immutable checkpoint after an invocation."""

    def __call__(self, *, agent: "Agent", **kwargs: Any) -> bool:
        """Return True to append an immutable snapshot for the given agent.

        Args:
            agent: The agent that just completed an invocation.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            True to create an immutable checkpoint, False otherwise.
        """
        ...


class SnapshotSessionManager(SessionManager):
    """Persists agent snapshots to a :class:`~strands.storage.storage.Storage` across invocations.

    On agent initialization the latest snapshot is restored automatically. On each
    qualifying lifecycle event the agent is re-captured and ``snapshot_latest`` is
    overwritten. When ``snapshot_trigger`` returns True after an invocation, an
    additional immutable snapshot is appended for time-travel restore.

    Example:
        ```python
        from strands import Agent
        from strands.session import SnapshotSessionManager
        from strands.storage import LocalFileStorage

        session = SnapshotSessionManager("my-session", storage=LocalFileStorage())
        agent = Agent(session_manager=session)
        ```
    """

    def __init__(
        self,
        session_id: str = "default-session",
        *,
        storage: Storage,
        save_latest_on: SaveLatestStrategy = "invocation",
        snapshot_trigger: SnapshotTrigger | None = None,
        migrate_from: RepositorySessionManager | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the snapshot session manager.

        Args:
            session_id: Unique session identifier. Must not contain path separators.
            storage: Unified storage backend that persists snapshot blobs.
            save_latest_on: When to overwrite ``snapshot_latest``. See :data:`SaveLatestStrategy`.
            snapshot_trigger: Optional callback invoked after each invocation; when it
                returns True an immutable snapshot is appended for checkpointing.
            migrate_from: Optional message-log session manager (``FileSessionManager``,
                ``S3SessionManager``) to migrate from once. When set and no snapshot
                exists yet, the agent is restored from the message log and an equivalent
                snapshot is written; every later run restores from the snapshot and the
                message-log store is never read or written again. The agent must be
                constructed with the same conversation manager and system prompt it ran
                with, since restore replays state into those live components.
            **kwargs: Additional keyword arguments for future extensibility.

        Raises:
            ValueError: If ``session_id`` is empty, is a relative-path segment (``.`` or ``..``),
                normalizes to empty, or contains a path separator; or if ``save_latest_on`` is
                not a recognized strategy.
        """
        self.session_id = validate_identifier(session_id, Identifier.SESSION)
        # validate_identifier permits "."/".."/whitespace, which either collapse the session
        # prefix (silently writing to a shared/wrong key on which delete_session then no-ops)
        # or explode deep in the storage layer. Reject anything that isn't a usable id here.
        if not self.session_id.strip() or self.session_id in (".", ".."):
            raise ValueError(f"session_id is not a valid session identifier: {session_id!r}")
        if save_latest_on not in _SAVE_LATEST_STRATEGIES:
            # Silently accepting an unknown value would register no save hooks — the session
            # would persist nothing with no error.
            raise ValueError(f"save_latest_on must be one of {_SAVE_LATEST_STRATEGIES}, got {save_latest_on!r}")
        self._storage = _resolve_storage(storage)
        self._save_latest_on: SaveLatestStrategy = save_latest_on
        self._snapshot_trigger = snapshot_trigger
        self._migrate_from = migrate_from

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register lifecycle callbacks for snapshot persistence.

        Overrides the base wiring: the message-log callbacks are replaced with
        snapshot save/restore handlers.
        """
        # Restore must be synchronous — AgentInitializedEvent forbids async callbacks.
        registry.add_callback(AgentInitializedEvent, lambda event: self.initialize(event.agent))

        # The save paths run under invoke_callbacks_async, so register them as native
        # async handlers and avoid the sync bridge.
        if self._save_latest_on == "message":
            registry.add_callback(MessageAddedEvent, self._on_message_added)
        registry.add_callback(AfterInvocationEvent, self._on_after_invocation)

        # Multi-agent (Graph/Swarm): restore the orchestrator once on init, persist its state
        # after each node and after the invocation. Latest-only (no immutable checkpoints),
        # matching the TypeScript SDK's multi-agent path.
        registry.add_callback(MultiAgentInitializedEvent, self._on_multi_agent_initialized)
        registry.add_callback(AfterNodeCallEvent, self._on_after_node_call)
        registry.add_callback(AfterMultiAgentInvocationEvent, self._on_after_multi_agent_invocation)

    # -- ABC methods (invoked synchronously by the Agent; bridge to async storage) --

    def initialize(self, agent: "Agent", **kwargs: Any) -> None:
        """Restore the agent from its latest snapshot, if one exists.

        Args:
            agent: Agent to restore.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        run_async(lambda: self._initialize_async(agent))

    def sync_agent(self, agent: "Agent", **kwargs: Any) -> None:
        """Capture the agent and overwrite ``snapshot_latest``.

        Args:
            agent: Agent to persist.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        run_async(lambda: self._save_latest(agent))

    def redact_latest_message(self, redact_message: Message, agent: "Agent", **kwargs: Any) -> None:
        """Persist immediately after a guardrail redaction, under every strategy.

        The Agent has already applied the redaction to ``agent.messages[-1]`` before
        calling this, so re-capturing the agent flushes pre-redaction content out of
        the persisted latest snapshot. This flush happens regardless of ``save_latest_on``
        (including ``"trigger"``) because the Agent invokes this method directly, not through
        a hook the manager could decline to register — so pre-redaction content never sits at
        rest. This diverges from the TypeScript SDK, which gates redaction persistence behind
        an ``AfterModelCall`` hook it skips under ``"trigger"`` and therefore does not flush there.

        Args:
            redact_message: The redacted replacement message (already applied by the Agent).
            agent: Agent whose latest message was redacted.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        run_async(lambda: self._save_latest(agent))

    def append_message(self, message: Message, agent: "Agent", **kwargs: Any) -> None:
        """No-op — snapshots capture the whole agent.

        Per-message persistence under the ``"message"`` strategy is handled by the
        ``MessageAddedEvent`` hook, not by this method.

        Args:
            message: The message that was appended (unused).
            agent: The agent the message was appended to (unused).
            **kwargs: Additional keyword arguments for future extensibility.
        """

    # -- Multi-agent ABC methods (Graph/Swarm orchestrators; latest-only, no time travel) --

    def initialize_multi_agent(self, source: "MultiAgentBase", **kwargs: Any) -> None:
        """Restore an orchestrator's state from its latest snapshot, if one exists.

        Args:
            source: The Graph/Swarm orchestrator to restore.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        run_async(lambda: self._restore_multi_agent(source))

    def sync_multi_agent(self, source: "MultiAgentBase", **kwargs: Any) -> None:
        """Capture the orchestrator's state and overwrite its latest snapshot.

        Args:
            source: The Graph/Swarm orchestrator to persist.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        run_async(lambda: self._save_multi_agent(source))

    # -- Public time-travel API --

    async def list_snapshot_ids(
        self, agent: "Agent", *, limit: int | None = None, start_after: str | None = None
    ) -> list[str]:
        """List immutable snapshot ids for an agent, oldest first.

        Args:
            agent: Agent whose snapshots to list.
            limit: Optional cap on the number of ids returned.
            start_after: Exclusive cursor; a snapshot id from a prior page.

        Returns:
            Immutable snapshot ids in chronological order.

        Raises:
            ValueError: If ``start_after`` is not a valid snapshot id.
        """
        if limit is not None and limit <= 0:
            return []
        if start_after is not None:
            _validate_snapshot_id(start_after)

        history_prefix = f"{_snapshots_prefix(self.session_id, agent.agent_id)}{_IMMUTABLE_HISTORY}/"
        keys = await self._storage.list(history_prefix)
        ids = sorted(match.group(1) for key in keys if (match := _SNAPSHOT_REGEX.search(key)))
        if start_after is not None:
            ids = [snapshot_id for snapshot_id in ids if snapshot_id > start_after]
        if limit is not None:
            ids = ids[:limit]
        return ids

    async def restore_snapshot(self, agent: "Agent", *, snapshot_id: str) -> bool:
        """Restore an agent from a specific immutable snapshot (time travel).

        Args:
            agent: Agent to restore into.
            snapshot_id: The immutable snapshot id to restore.

        Returns:
            True if the snapshot existed and was restored, False otherwise.

        Raises:
            ValueError: If ``snapshot_id`` is not a valid snapshot id.
        """
        return await self._restore(agent, snapshot_id=snapshot_id)

    async def delete_session(self) -> None:
        """Delete all snapshots for this session."""
        keys = await self._storage.list(_session_prefix(self.session_id))
        semaphore = asyncio.Semaphore(_DELETE_CONCURRENCY)

        async def _delete(key: str) -> None:
            async with semaphore:
                await self._storage.delete(key)

        await asyncio.gather(*(_delete(key) for key in keys))

    # -- Async internals --

    async def _initialize_async(self, agent: "Agent") -> None:
        """Restore latest snapshot on init, warning on overwrite and handling stateful models."""
        had_messages = len(agent.messages) > 0
        restored = await self._restore(agent)

        if not restored and self._migrate_from is not None:
            # No snapshot yet: restore through the message-log manager's proven path (offset +
            # conversation-manager prepend + tool-use repair + stateful handling), then capture
            # a snapshot so every later run restores from the snapshot instead. Guarded so the
            # message-log store is only read, never written (see _restore_from_legacy).
            if self._restore_from_legacy(agent):
                await self._save_latest(agent)
                restored = True

        if restored and had_messages:
            logger.warning(
                "agent_id=<%s>, session_id=<%s> | agent had existing messages that were overwritten by session restore",
                agent.agent_id,
                self.session_id,
            )

        # Stateful models manage history server-side, so restored messages would drift
        # from the server's view. Keep the restored model_state and drop the messages.
        if restored and agent.model.stateful and len(agent.messages) > 0:
            logger.debug(
                "agent_id=<%s>, message_count=<%s> | discarding restored messages for stateful model",
                agent.agent_id,
                len(agent.messages),
            )
            agent.messages = []

    def _restore_from_legacy(self, agent: "Agent") -> bool:
        """Restore the agent from the ``migrate_from`` message-log manager, read-only.

        ``RepositorySessionManager.initialize`` creates new records when the session has no
        agent yet, so migration only delegates to it when the agent record already exists.
        This keeps migration a pure read of the message-log store.

        Returns:
            True if the agent existed in the message-log store and was restored, False otherwise.
        """
        legacy = self._migrate_from
        if legacy is None:
            return False

        if legacy.session_repository.read_agent(legacy.session_id, agent.agent_id) is None:
            logger.debug(
                "agent_id=<%s>, session_id=<%s> | no message-log agent to migrate",
                agent.agent_id,
                legacy.session_id,
            )
            return False

        legacy.initialize(agent)
        logger.info(
            "agent_id=<%s>, session_id=<%s> | migrated message-log session to snapshot",
            agent.agent_id,
            self.session_id,
        )
        return True

    async def _restore(self, agent: "Agent", *, snapshot_id: str | None = None) -> bool:
        """Load a snapshot into the agent. Returns False if none exists."""
        data = await self._storage.read(_snapshot_key(self.session_id, agent.agent_id, snapshot_id=snapshot_id))
        if data is None:
            return False
        agent.load_snapshot(_deserialize_snapshot(data))
        return True

    async def _save_latest(self, agent: "Agent") -> None:
        """Capture the agent and overwrite ``snapshot_latest``."""
        data = _serialize_snapshot(self._capture(agent))
        await self._storage.write(_snapshot_key(self.session_id, agent.agent_id, snapshot_id=None), data)

    async def _save_immutable_and_latest(self, agent: "Agent") -> None:
        """Capture once and write the immutable snapshot, then ``snapshot_latest``.

        Ordered, not concurrent: writing the immutable checkpoint first means a partial failure
        can leave an orphaned immutable snapshot (harmless — the next list simply includes it)
        but never a ``snapshot_latest`` pointing at history that was never written.
        """
        data = _serialize_snapshot(self._capture(agent))
        await self._storage.write(_snapshot_key(self.session_id, agent.agent_id, snapshot_id=_new_snapshot_id()), data)
        await self._storage.write(_snapshot_key(self.session_id, agent.agent_id, snapshot_id=None), data)

    async def _restore_multi_agent(self, source: "MultiAgentBase") -> None:
        """Load an orchestrator's latest snapshot into it, if one exists."""
        data = await self._storage.read(_multi_agent_key(self.session_id, source.id))
        if data is None:
            return
        source.deserialize_state(_deserialize_multi_agent(data, expected_orchestrator_id=source.id))

    async def _save_multi_agent(self, source: "MultiAgentBase") -> None:
        """Capture the orchestrator's state and overwrite its latest snapshot."""
        data = _serialize_multi_agent(source.id, source.serialize_state())
        await self._storage.write(_multi_agent_key(self.session_id, source.id), data)

    async def _on_multi_agent_initialized(self, event: MultiAgentInitializedEvent) -> None:
        """Restore the orchestrator on construction (fires once per orchestrator)."""
        await self._restore_multi_agent(event.source)

    async def _on_after_node_call(self, event: AfterNodeCallEvent) -> None:
        """Persist the orchestrator after each node completes."""
        await self._save_multi_agent(event.source)

    async def _on_after_multi_agent_invocation(self, event: AfterMultiAgentInvocationEvent) -> None:
        """Persist the orchestrator after the invocation completes."""
        await self._save_multi_agent(event.source)

    async def _on_message_added(self, event: MessageAddedEvent) -> None:
        """Save latest after each message under the ``"message"`` strategy."""
        await self._save_latest(event.agent)

    async def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        """Save latest on invocation and fire the immutable-checkpoint trigger.

        When the trigger fires, the immutable+latest write subsumes the invocation save, so the
        agent is captured only once even under ``save_latest_on="invocation"``.

        ``"message"`` also saves here, not just per message: the Agent runs
        ``conversation_manager.apply_management`` (trimming/summarizing) after the last
        ``MessageAddedEvent`` but before this event, so the per-message saves would otherwise
        persist pre-management messages and a stale ``removed_message_count``.
        """
        triggered = False
        if self._snapshot_trigger is not None:
            try:
                triggered = self._snapshot_trigger(agent=event.agent)
            except Exception:
                # A caller's trigger raising must not discard the completed turn's latest save;
                # log and fall through to the normal save below.
                logger.exception(
                    "agent_id=<%s>, session_id=<%s> | snapshot_trigger raised; skipping immutable checkpoint",
                    event.agent.agent_id,
                    self.session_id,
                )
        if triggered:
            await self._save_immutable_and_latest(event.agent)
        elif self._save_latest_on in ("invocation", "message"):
            await self._save_latest(event.agent)

    def _capture(self, agent: "Agent") -> Snapshot:
        """Capture a full session snapshot including the system prompt.

        The shared ``"session"`` preset omits ``system_prompt`` (opt-in for callers like
        the goal plugin); session persistence includes it so a rehydrated agent behaves
        identically to the original, matching the TypeScript SDK's session preset.
        """
        return agent.take_snapshot(preset="session", include=["system_prompt"])
