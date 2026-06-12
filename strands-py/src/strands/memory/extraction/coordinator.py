"""Background coordinator that saves conversation messages to memory stores.

The :class:`ExtractionCoordinator` buffers every message the agent produces and,
when a store's trigger fires, saves that store's unsaved messages in the
background without slowing the agent loop down.

How it works, in three pieces:

1. **The buffer.** Every message the agent produces is copied into one shared
   list (``_pending``). Each gets a number (``seq``) that only ever counts up,
   so we can tell which messages are newer. We keep our own copy here (rather
   than reading the agent's live message list) because the agent can delete old
   messages to stay within its context window; our copy means we never lose one
   before it is saved.

2. **Per-store progress.** Each store can save at its own pace, so we remember,
   per store, the ``seq`` of the last message it has already saved (``_marks``).
   When a store saves, it only looks at messages newer than that number, so the
   same message is never saved twice to the same store.

3. **One save at a time per store.** A store might be asked to save again while a
   previous save is still running. We chain each store's saves one after another
   (``_chains``) so they cannot overlap or run out of order.

If a store fails to save :data:`SAVE_FAILURES_BEFORE_BACKOFF` times in a row it
backs off: instead of trying every turn, it retries only once every
:data:`BACKOFF_PROBE_INTERVAL` attempts (a probe). A successful probe clears the
failure streak and resumes normal saving -- so a transient outage recovers on
its own and the messages buffered during it are saved once the store comes back.
A permanently broken store keeps probing and logs an error each time, surfacing
the misconfiguration.

Scheduling uses ``asyncio.create_task`` for fire-and-forget saves, a per-store
``asyncio.Task`` chain to serialize a single store's saves, and
``asyncio.gather(..., return_exceptions=True)`` to run concurrent writes so one
failure does not cancel the rest.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ...models.model import Model
from ...types.content import ContentBlock, Message
from ...types.exceptions import AggregateMemoryError
from ..types import MemoryStore
from .types import DEFAULT_MEMORY_MESSAGE_FILTER, ExtractorContext, MemoryMessageFilter

logger = logging.getLogger(__name__)

# Number of consecutive save failures after which a store backs off (stops trying
# every turn).
SAVE_FAILURES_BEFORE_BACKOFF = 10

# While backed off, a store retries only once every this many save attempts (a
# probe).
BACKOFF_PROBE_INTERVAL = 3


@dataclass
class _Buffered:
    """A buffered message and its sequence number.

    Attributes:
        seq: The monotonically increasing sequence number assigned when the
            message was recorded.
        message: The buffered conversation message.
    """

    seq: int
    message: Message


class ExtractionCoordinator:
    """Saves conversation messages to memory stores in the background.

    Buffers every recorded message and, per store, tracks a high-water mark of
    the last ``seq`` saved so each message is delivered to a store at most once.
    Saves for a single store are serialized through a per-store task chain;
    saves for different stores run independently. Failures are logged and
    swallowed so saving never breaks the agent loop, with per-store backoff for
    repeatedly failing stores.
    """

    def __init__(self, stores: list[MemoryStore], default_model: Model) -> None:
        """Initialize the coordinator.

        Args:
            stores: The extraction-configured stores this coordinator manages.
            default_model: The agent's model, passed to extractors that do not
                configure their own.
        """
        self._stores = list(stores)
        self._default_model = default_model
        # The shared list of messages waiting to be saved, oldest first.
        self._pending: list[_Buffered] = []
        # The number to give the next message added to the buffer.
        self._next_seq = 0
        # Per store (keyed by ``id(store)``): the ``seq`` of the last message
        # that store has already saved. Starts at -1 (saved none).
        self._marks: dict[int, int] = {id(store): -1 for store in stores}
        # Per store: the currently-running save task, so the next save waits its
        # turn.
        self._chains: dict[int, asyncio.Task] = {}
        # Per store: how many saves have failed in a row. Reset to 0 on success.
        self._consecutive_failures: dict[int, int] = {}
        # Per store: while backed off, counts save requests so we let every Nth
        # through as a probe.
        self._backoff_counters: dict[int, int] = {}
        # Fire-and-forget background tasks, retained so they are not garbage
        # collected mid-flight.
        self._background: set[asyncio.Task] = set()

    def record(self, message: Message) -> None:
        """Add a message to the buffer.

        Args:
            message: The conversation message to buffer for later saving.
        """
        self._pending.append(_Buffered(self._next_seq, message))
        self._next_seq += 1

    def schedule(self, store: MemoryStore) -> None:
        """Save this store's unsaved messages in the background, non-blocking.

        Dispatches :meth:`process` as a tracked background task and returns
        immediately, so a trigger calling this from a hook never blocks the
        agent. Exceptions are swallowed and logged.

        Args:
            store: The store to save for.
        """
        task = asyncio.create_task(self.process(store))
        self._background.add(task)

        def _done(completed: asyncio.Task) -> None:
            self._background.discard(completed)
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                logger.warning("store=<%s>, reason=<%s> | background memory save failed", store.name, error)

        task.add_done_callback(_done)

    async def process(self, store: MemoryStore) -> None:
        """Save this store's unsaved messages, queued behind its previous save.

        Skips the save when the store is backed off and this request is not a
        probe (see :meth:`_should_attempt`).

        Args:
            store: The store to save for.
        """
        if not self._should_attempt(store):
            return
        await self._enqueue(store)

    def _enqueue(self, store: MemoryStore) -> asyncio.Task:
        """Queue a save for the store behind its previous one.

        Creates a task that first awaits the store's previous chain task
        (ignoring its result/exception) and then runs :meth:`_extract`, so saves
        for a single store never overlap or reorder.

        Args:
            store: The store to save for.

        Returns:
            The task representing this queued save.
        """
        previous = self._chains.get(id(store))
        task = asyncio.create_task(self._run_chain(store, previous))
        self._chains[id(store)] = task
        return task

    async def _run_chain(self, store: MemoryStore, previous: asyncio.Task | None) -> None:
        """Await the previous save for ``store`` (if any) then run this one.

        Serializes a single store's saves so they never overlap or reorder. The
        previous save handles its own outcome internally (errors are logged and
        swallowed in :meth:`_extract`), so it always completes normally.

        Args:
            store: The store to save for.
            previous: The previous chain task to wait behind, or ``None``.
        """
        if previous is not None:
            await previous
        await self._extract(store)

    def _should_attempt(self, store: MemoryStore) -> bool:
        """Return whether to attempt a save now.

        A healthy store always attempts. A backed-off store (too many failures in
        a row) attempts only once every :data:`BACKOFF_PROBE_INTERVAL` requests
        -- a probe to see if it has recovered -- and skips the rest.

        Args:
            store: The store to check.

        Returns:
            ``True`` if a save should be attempted now.
        """
        if self._consecutive_failures.get(id(store), 0) < SAVE_FAILURES_BEFORE_BACKOFF:
            return True
        count = self._backoff_counters.get(id(store), 0) + 1
        self._backoff_counters[id(store)] = count
        return count % BACKOFF_PROBE_INTERVAL == 0

    async def flush(self) -> None:
        """Save every store's remaining messages and wait for all to finish.

        Call this at a boundary you control -- typically app shutdown -- to make
        sure nothing in the buffer is lost. It first tells every store to save
        (bypassing backoff, so a recovered store still writes its backlog);
        stores with nothing to save do nothing. Then it waits, re-checking until
        no new save has started, so saves that begin while waiting are also
        covered. Never raises (write errors are swallowed inside
        :meth:`_extract`).
        """
        for store in self._stores:
            self._enqueue(store)
        while True:
            snapshot = list(self._chains.values())
            await asyncio.gather(*snapshot, return_exceptions=True)
            current = list(self._chains.values())
            # If nothing new started while we waited, everything is done.
            if len(current) == len(snapshot) and all(
                current_task is snapshot_task for current_task, snapshot_task in zip(current, snapshot, strict=True)
            ):
                return

    async def _extract(self, store: MemoryStore) -> None:
        """Save the store's messages newer than its high-water mark.

        Reads and advances the per-store mark synchronously before the first
        ``await`` so concurrent saves never pick up the same messages. On failure
        the mark is rolled back so the batch retries next time.

        Args:
            store: The store to save for.
        """
        mark = self._marks.get(id(store), -1)
        fresh = [buffered for buffered in self._pending if buffered.seq > mark]
        if not fresh:
            return

        # Mark these messages as saved before we start saving, so a queued save
        # behind this one will not pick them up again. If the save fails we put
        # the mark back (below) and they retry.
        self._marks[id(store)] = fresh[-1].seq

        extraction = store.extraction
        assert extraction is not None  # noqa: S101 - extraction stores always configure this.
        message_filter = extraction.filter or DEFAULT_MEMORY_MESSAGE_FILTER
        filtered = self._filter_messages([buffered.message for buffered in fresh], message_filter)

        try:
            if filtered:
                await self._write(store, filtered)
                # A successful write clears the failure streak and ends any
                # backoff. Only a real write counts as recovery -- a fully
                # filtered (empty) turn never touched the backend, so it leaves
                # backoff state untouched (it still advances the mark above;
                # those messages had nothing to save).
                self._consecutive_failures[id(store)] = 0
                self._backoff_counters.pop(id(store), None)
        except Exception as error:  # noqa: BLE001 - saving must never break the agent loop.
            self._on_save_failed(store, mark, error)
        finally:
            self._trim()

    async def _write(self, store: MemoryStore, messages: list[Message]) -> None:
        """Save the messages to the store, one of two ways.

        - Store has an extractor: run it to pull out facts, then write each fact
          via ``add``. Fact writes run concurrently; if any fails the whole batch
          is re-raised so the caller retries it next time (so a fact that already
          saved may be written again -- stores should expect duplicates).
        - No extractor: hand the raw messages to ``add_messages`` so the store
          keeps their roles.

        Args:
            store: The store to write to.
            messages: The filtered messages to save.

        Raises:
            AggregateMemoryError: If any concurrent ``add`` write fails.
        """
        extraction = store.extraction
        assert extraction is not None  # noqa: S101 - extraction stores always configure this.

        if extraction.extractor is not None:
            entries = await extraction.extractor.extract(messages, ExtractorContext(default_model=self._default_model))
            results = await asyncio.gather(
                *(store.add(entry.content, entry.metadata) for entry in entries),
                return_exceptions=True,
            )
            failures = [result for result in results if isinstance(result, BaseException)]
            if failures:
                raise AggregateMemoryError(
                    f"failed to write {len(failures)} of {len(entries)} extracted entries",
                    failures,
                )
            return

        await store.add_messages(messages)

    def _filter_messages(self, messages: list[Message], message_filter: MemoryMessageFilter) -> list[Message]:
        """Remove excluded content blocks and drop any emptied message.

        Pure: never mutates the input messages or their blocks; builds new
        message dicts and content lists, preserving ``role`` and carrying
        ``metadata`` when present.

        Args:
            messages: The messages to filter.
            message_filter: The filter whose ``exclude`` kinds are stripped.

        Returns:
            New messages with excluded blocks removed and emptied messages
            dropped.
        """
        exclude = set(message_filter.exclude)
        result: list[Message] = []
        for message in messages:
            content = [block for block in message["content"] if self._block_kind(block) not in exclude]
            if content:
                new_message: Message = {"role": message["role"], "content": content}
                if message.get("metadata") is not None:
                    new_message["metadata"] = message["metadata"]
                result.append(new_message)
        return result

    def _block_kind(self, block: ContentBlock) -> str:
        """Return the kind of a content block (e.g. ``"text"``, ``"toolUse"``).

        A text block is ``{"text": ...}``; every other block is a single-key
        wrapper (``{"toolUse": ...}``, ...).

        Args:
            block: The content block to classify.

        Returns:
            The block's kind, or ``""`` for an empty block.
        """
        if "text" in block:
            return "text"
        return next(iter(block.keys()), "")

    def _on_save_failed(self, store: MemoryStore, mark_before_save: int, error: BaseException) -> None:
        """Handle a failed save.

        Puts the mark back so the messages retry next time. Once a store has
        failed :data:`SAVE_FAILURES_BEFORE_BACKOFF` times in a row it logs an
        error and enters backoff; before that it logs a warning. The messages
        stay buffered either way, so a store that recovers saves them.

        Args:
            store: The store whose save failed.
            mark_before_save: The high-water mark to restore.
            error: The underlying failure.
        """
        failures = self._consecutive_failures.get(id(store), 0) + 1
        self._consecutive_failures[id(store)] = failures
        self._marks[id(store)] = mark_before_save
        reason = str(error)

        if failures >= SAVE_FAILURES_BEFORE_BACKOFF:
            logger.error(
                "store=<%s>, failures=<%s>, reason=<%s> | memory store save failing repeatedly",
                store.name,
                failures,
                reason,
            )
        else:
            logger.warning("store=<%s>, reason=<%s> | memory extraction failed", store.name, reason)

    def _trim(self) -> None:
        """Drop buffered messages every store has already saved.

        A store that has not saved a message yet keeps it buffered, so a store
        stuck failing for good slowly grows the buffer; that surfaces as repeated
        error logs and is bounded by the (non-persisted) session.
        """
        min_mark = min(self._marks.values())
        self._pending = [buffered for buffered in self._pending if buffered.seq > min_mark]
