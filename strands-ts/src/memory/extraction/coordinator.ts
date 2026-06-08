import type { MemoryStore } from '../types.js'
import type { MessageData, ContentBlockData } from '../../types/messages.js'
import type { Model } from '../../models/model.js'
import { logger } from '../../logging/logger.js'
import { normalizeError } from '../../errors.js'
import { DEFAULT_MEMORY_MESSAGE_FILTER, type MemoryMessageFilter, type MemoryContentBlockType } from './types.js'

/** A message captured into the buffer, tagged with a monotonic sequence number. */
interface BufferedMessage {
  seq: number
  message: MessageData
}

/** The memory-facing kind of a content block, derived from its discriminator key. */
function _blockKind(block: ContentBlockData): MemoryContentBlockType | string {
  // A text block is `{ text: string }`; every other block is a single-key wrapper (`{ toolUse }`, …).
  if ('text' in block) return 'text'
  return Object.keys(block)[0] ?? ''
}

/**
 * Applies a {@link MessageFilter} to a batch: strips excluded content blocks and drops messages left
 * with no content. Returns new objects; inputs are not mutated.
 */
function _filterMessages(messages: MessageData[], filter: MemoryMessageFilter): MessageData[] {
  const exclude = new Set<string>(filter.exclude)
  const result: MessageData[] = []
  for (const message of messages) {
    const content = message.content.filter((block) => !exclude.has(_blockKind(block)))
    if (content.length > 0) {
      result.push({
        role: message.role,
        content,
        ...(message.metadata !== undefined && { metadata: message.metadata }),
      })
    }
  }
  return result
}

/**
 * Coordinates automatic extraction for a {@link MemoryManager}'s extraction-configured stores.
 *
 * Buffers every conversation message (via {@link record}) into one shared buffer, then for each store
 * (via {@link process}) handles only the messages newer than that store's high-water mark — so
 * repeated fires never duplicate writes — applies the store's filter, and routes the result by
 * extraction shape (extractor → per-entry `add`; else `addMessages` batch).
 *
 * The buffer is the source of truth, independent of the agent's `messages` (which the conversation
 * manager may evict), so eviction never drops un-extracted messages. Processing is serialized per
 * store, while each store advances its own cursor independently.
 */
export class ExtractionCoordinator {
  private readonly _stores: MemoryStore[]
  private readonly _defaultModel: Model
  /** Messages added this session, newest last, each tagged with a monotonic sequence number. */
  private _pending: BufferedMessage[] = []
  /** Next sequence number to assign to a captured message. */
  private _nextSeq = 0
  /** Per-store high-water mark: the highest message `seq` already processed. */
  private readonly _marks = new Map<MemoryStore, number>()
  /** Per-store serialized write chain, so a store's runs never overlap or reorder. */
  private readonly _chains = new Map<MemoryStore, Promise<void>>()

  /**
   * @param stores - The extraction-configured stores this coordinator manages
   * @param defaultModel - The agent's model, passed to extractors that don't configure their own
   */
  constructor(stores: MemoryStore[], defaultModel: Model) {
    this._stores = stores
    this._defaultModel = defaultModel
    for (const store of stores) {
      this._marks.set(store, -1)
    }
  }

  /** Records a newly added message into the buffer. */
  record(message: MessageData): void {
    this._pending.push({ seq: this._nextSeq++, message })
  }

  /**
   * Enqueues extraction for one store, serialized per store so concurrent trigger fires don't overlap.
   * Errors are logged and swallowed: extraction is best-effort and must never break the agent loop.
   *
   * Returns the per-store chain promise for internal coordination ({@link flush}); the trigger-facing
   * `fire` callback ignores it so firing never blocks the agent loop.
   */
  process(store: MemoryStore): Promise<void> {
    const previous = this._chains.get(store) ?? Promise.resolve()
    const next = previous
      .then(() => this._extract(store))
      .catch((err) => {
        logger.warn(`store=<${store.name}>, reason=<${normalizeError(err).message}> | memory extraction failed`)
      })
    this._chains.set(store, next)
    return next
  }

  /**
   * Force-completes extraction for every store, then awaits it. For graceful shutdown and
   * deterministic tests — the background path never awaits this.
   *
   * Unlike a trigger fire, this **enqueues a run for every store first**, so a store whose trigger
   * hadn't fired yet (e.g. an `IntervalTrigger` mid-cycle when the session ends) still extracts its
   * buffered tail rather than losing it. Runs that find nothing fresh no-op. Then it loops until the
   * set of chain promises stops changing, so runs enqueued while flushing are also drained.
   */
  async flush(): Promise<void> {
    for (const store of this._stores) {
      void this.process(store)
    }
    for (;;) {
      const snapshot = [...this._chains.values()]
      await Promise.all(snapshot)
      const current = [...this._chains.values()]
      // No chain was replaced by a newer run during the await → fully drained.
      if (current.length === snapshot.length && current.every((p, i) => p === snapshot[i])) {
        return
      }
    }
  }

  private async _extract(store: MemoryStore): Promise<void> {
    const mark = this._marks.get(store) ?? -1
    const fresh = this._pending.filter((buffered) => buffered.seq > mark)
    if (fresh.length === 0) {
      return
    }

    // Advance the mark before any async write so a concurrent fire (already serialized behind this
    // one) doesn't re-take the same messages if the write below yields.
    const highestSeq = fresh[fresh.length - 1]!.seq
    this._marks.set(store, highestSeq)

    const extraction = store.extraction!
    const filter = extraction.filter ?? DEFAULT_MEMORY_MESSAGE_FILTER
    const filtered = _filterMessages(
      fresh.map((buffered) => buffered.message),
      filter
    )

    try {
      if (filtered.length > 0) {
        await this._write(store, filtered)
      }
    } catch (err) {
      // Roll the mark back so the next fire retries these messages rather than losing them.
      this._marks.set(store, mark)
      throw err
    } finally {
      this._trim()
    }
  }

  /**
   * Routes filtered messages to the store by extraction shape (validated at construction):
   * 1. extractor configured → distill to entries, write them via `add` concurrently;
   * 2. no extractor → hand the raw batch to `addMessages` in one call (roles preserved).
   *
   * Entry writes fan out in parallel (parity with the `add_memory` tool / `MemoryManager.add` paths)
   * and an `AggregateError` is thrown if any fails, so the caller rolls the high-water mark back and
   * the whole batch retries. (A retried batch can re-write entries that already succeeded, since
   * `add` isn't assumed idempotent — the same at-least-once tradeoff as the tool path.)
   */
  private async _write(store: MemoryStore, messages: MessageData[]): Promise<void> {
    const extractor = store.extraction!.extractor

    if (extractor) {
      const entries = await extractor.extract(messages, { defaultModel: this._defaultModel })
      const settled = await Promise.allSettled(entries.map((entry) => store.add!(entry.content, entry.metadata)))
      const failures = settled.filter((r): r is PromiseRejectedResult => r.status === 'rejected')
      if (failures.length > 0) {
        throw new AggregateError(
          failures.map((f) => f.reason),
          `failed to write ${failures.length} of ${entries.length} extracted entries`
        )
      }
      return
    }

    await store.addMessages!(messages)
  }

  /** Drops buffered messages every store has already processed, keeping the buffer bounded. */
  private _trim(): void {
    let minMark = Infinity
    for (const store of this._stores) {
      minMark = Math.min(minMark, this._marks.get(store) ?? -1)
    }
    if (minMark === Infinity) {
      return
    }
    this._pending = this._pending.filter((buffered) => buffered.seq > minMark)
  }
}
