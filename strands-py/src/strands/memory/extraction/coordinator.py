"""Background coordinator that saves conversation messages to memory stores.

The :class:`ExtractionCoordinator` buffers every message the agent produces and,
when a store's trigger fires, saves that store's unsaved messages in the
background. It keeps a per-store high-water mark so each message is delivered to
a store at most once, serializes a single store's saves through a per-store task
chain, and backs off stores that fail repeatedly.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from ...models.model import Model
from ...types.content import ContentBlock, Message
from ...types.exceptions import AggregateMemoryError
from ..types import MemoryStore
from .resolve_extraction_config import _ResolvedExtractionConfig
from .types import Extractor, ExtractorContext, MemoryMessageFilter

logger = logging.getLogger(__name__)

# Number of consecutive save failures after which a store backs off.
SAVE_FAILURES_BEFORE_BACKOFF = 10

# While backed off, a store retries only once every this many save attempts.
BACKOFF_PROBE_INTERVAL = 3


@dataclass
class _ExtractionBinding:
    """A store paired with its fully-resolved extraction config.

    Attributes:
        store: The memory store to extract into.
        config: The store's fully-resolved extraction config (triggers, extractor,
            filter).
    """

    store: MemoryStore
    config: _ResolvedExtractionConfig


@dataclass
class _Buffered:
    """A buffered message and its monotonically increasing sequence number."""

    seq: int
    message: Message


class ExtractionCoordinator:
    """Saves conversation messages to memory stores in the background.

    Buffers every recorded message and, per store, tracks a high-water mark of
    the last ``seq`` saved so each message is delivered at most once. A single
    store's saves are serialized through a per-store task chain; different stores
    save independently. Failures are logged and swallowed, with per-store backoff
    for repeatedly failing stores.
    """

    def __init__(self, bindings: list[_ExtractionBinding], default_model: Model) -> None:
        """Initialize the coordinator.

        Args:
            bindings: The extraction-configured stores this coordinator manages,
                each paired with its fully-resolved config.
            default_model: The agent's model, passed to extractors that do not
                configure their own.
        """
        self._stores = [binding.store for binding in bindings]
        # Per store: its resolved extraction config (triggers, extractor, filter).
        self._configs: dict[int, _ResolvedExtractionConfig] = {
            id(binding.store): binding.config for binding in bindings
        }
        self._default_model = default_model
        # Messages waiting to be saved, oldest first.
        self._pending: list[_Buffered] = []
        # The ``seq`` to assign the next buffered message.
        self._next_seq = 0
        # Per store: ``seq`` of the last message it has saved (-1 means none).
        self._marks: dict[int, int] = {id(binding.store): -1 for binding in bindings}
        # Per store: the currently-running save task, so the next save waits its turn.
        self._chains: dict[int, asyncio.Task] = {}
        # Per store: consecutive save failures, reset to 0 on success.
        self._consecutive_failures: dict[int, int] = {}
        # Per store: save-request count while backed off, to let every Nth through as a probe.
        self._backoff_counters: dict[int, int] = {}
        # Fire-and-forget background tasks, retained so they aren't GC'd mid-flight.
        self._background: set[asyncio.Task] = set()

    def record(self, message: Message) -> None:
        """Add a message to the buffer."""
        self._pending.append(_Buffered(self._next_seq, message))
        self._next_seq += 1

    def schedule(self, store: MemoryStore) -> None:
        """Save this store's unsaved messages in the background, non-blocking.

        Dispatches the save and returns immediately. A no-op when the store is
        backed off and this request is not a probe.
        """
        task = self.process(store)
        if task is None:
            return
        self._background.add(task)

        def _done(completed: asyncio.Task) -> None:
            self._background.discard(completed)
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                logger.warning("store=<%s>, reason=<%s> | background memory save failed", store.name, error)

        task.add_done_callback(_done)

    def process(self, store: MemoryStore) -> asyncio.Task | None:
        """Queue a save for this store behind its previous save.

        Returns the task running the save, or ``None`` when the store is backed
        off and this request is not a probe.
        """
        if not self._should_attempt(store):
            return None
        return self._enqueue(store)

    def _enqueue(self, store: MemoryStore) -> asyncio.Task:
        """Queue this store's save behind its previous one and return the task."""
        previous = self._chains.get(id(store))
        task = asyncio.create_task(self._run_chain(store, previous))
        self._chains[id(store)] = task
        return task

    async def _run_chain(self, store: MemoryStore, previous: asyncio.Task | None) -> None:
        """Run this store's save after its previous one completes."""
        if previous is not None:
            await previous
        await self._extract(store)

    def _should_attempt(self, store: MemoryStore) -> bool:
        """Return whether to attempt a save now.

        A healthy store always attempts. A backed-off store attempts only once
        every :data:`BACKOFF_PROBE_INTERVAL` requests (a probe) and skips the
        rest.
        """
        if self._consecutive_failures.get(id(store), 0) < SAVE_FAILURES_BEFORE_BACKOFF:
            return True
        count = self._backoff_counters.get(id(store), 0) + 1
        self._backoff_counters[id(store)] = count
        return count % BACKOFF_PROBE_INTERVAL == 0

    async def flush(self) -> None:
        """Save every store's remaining buffered messages and wait for completion.

        Bypasses backoff and also waits out saves that start while waiting.
        Never raises.
        """
        for store in self._stores:
            self._enqueue(store)
        while True:
            snapshot = list(self._chains.values())
            await asyncio.gather(*snapshot, return_exceptions=True)
            current = list(self._chains.values())
            # Done once nothing new started while we waited.
            if len(current) == len(snapshot) and all(
                current_task is snapshot_task for current_task, snapshot_task in zip(current, snapshot, strict=True)
            ):
                return

    async def _extract(self, store: MemoryStore) -> None:
        """Save the store's messages newer than its high-water mark.

        On failure the mark is rolled back so the batch retries next time.
        """
        mark = self._marks.get(id(store), -1)
        fresh = [buffered for buffered in self._pending if buffered.seq > mark]
        if not fresh:
            return

        config = self._configs[id(store)]

        # Mark saved before saving so a queued save won't pick these up again;
        # rolled back below on failure.
        self._marks[id(store)] = fresh[-1].seq

        filtered = self._filter_messages([buffered.message for buffered in fresh], config.filter)

        try:
            if filtered:
                await self._write(store, filtered, config.extractor)
                # Successful write clears the failure streak and ends backoff. A
                # fully filtered (empty) turn never touched the backend, so it
                # leaves backoff state untouched.
                self._consecutive_failures[id(store)] = 0
                self._backoff_counters.pop(id(store), None)
        except Exception as error:  # noqa: BLE001 - saving must never break the agent loop.
            self._on_save_failed(store, mark, error)
        finally:
            self._trim()

    async def _write(self, store: MemoryStore, messages: list[Message], extractor: Extractor | None) -> None:
        """Save the messages to the store, one of two ways.

        - With an extractor: run it, then write each fact via ``add``
          concurrently. If any write fails the whole batch is re-raised and
          retried later, so stores should expect duplicate writes.
        - Without an extractor: hand the raw messages to ``add_messages``.

        Raises:
            AggregateMemoryError: If any concurrent ``add`` write fails.
        """
        if extractor is not None:
            entries = await extractor.extract(messages, ExtractorContext(default_model=self._default_model))
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
        """Remove excluded content blocks, dropping any message left empty.

        Builds new message dicts rather than mutating the inputs.
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
        """Return the content block's kind (its single key), or ``""`` if empty."""
        return next(iter(block.keys()), "")

    def _on_save_failed(self, store: MemoryStore, mark_before_save: int, error: BaseException) -> None:
        """Handle a failed save.

        Rolls the mark back so the messages retry next time. After
        :data:`SAVE_FAILURES_BEFORE_BACKOFF` consecutive failures the store
        enters backoff and logs an error; before that it logs a warning.
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

        A store stuck failing keeps its messages buffered, so the buffer grows
        until it recovers; this is bounded by the (non-persisted) session.
        """
        min_mark = min(self._marks.values())
        self._pending = [buffered for buffered in self._pending if buffered.seq > min_mark]
