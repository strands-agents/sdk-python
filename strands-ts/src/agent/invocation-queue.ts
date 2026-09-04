/**
 * Invocation queueing for agents using `'enqueue'` or `'cancelPrevious'` concurrency.
 *
 * The queue lives at the invocation lock: entries are added when `invoke()`/`stream()`
 * is called while the agent is busy, and the lock is handed to the next entry inside
 * `stream()`'s `finally` — before the busy flag is cleared — so a late arrival either
 * gets the lock or lands in the queue the current owner is about to pop.
 */

import { PendingInvocationCancelledError } from '../errors.js'
import type { InvokeArgs } from '../types/agent.js'

/**
 * Supported values for the `concurrentInvocationMode` parameter.
 */
export const CONCURRENT_INVOCATION_MODES = ['throw', 'enqueue', 'cancelPrevious'] as const

/**
 * Behavior when `invoke()` or `stream()` is called while an invocation is already in
 * progress. Set agent-wide via `concurrentInvocationMode`, or per call via
 * `InvokeOptions.ifBusy` (same values).
 *
 * - `'throw'`: reject the new call with `ConcurrentInvocationError` (default).
 * - `'enqueue'`: queue the new call FIFO; it runs as its own invocation — with its own
 *   result, hook events, and cancellation signal — when the current one finishes.
 * - `'cancelPrevious'`: cancel the running invocation via `agent.cancel()` and run this
 *   call next, ahead of any queued invocations. The cancelled caller receives its own
 *   result with `stopReason: 'cancelled'`; this call runs as a fresh invocation with a
 *   fresh cancellation signal. When the running invocation has already completed its
 *   final model pass and is only awaiting background work (e.g. background-task
 *   settlement), it stops waiting and returns its completed result with
 *   `stopReason: 'endTurn'` instead; undelivered background results are delivered
 *   in a later invocation.
 */
export type ConcurrentInvocationMode = (typeof CONCURRENT_INVOCATION_MODES)[number]

/**
 * A queued invocation, as surfaced by `agent.pendingInvocations`.
 */
export interface PendingInvocation {
  /** Queue-unique identifier, usable with `agent.cancelPending(id)`. */
  readonly id: string
  /** When the call was submitted (entered the queue). */
  readonly submittedAt: Date
  /** Short text preview of the call's input, for introspection and model visibility. */
  readonly preview: string
}

/** A queue entry: the public snapshot fields plus the waiter's continuation. */
interface QueueEntry extends PendingInvocation {
  resolve: () => void
  reject: (error: Error) => void
  /** Detaches the entry's abort listener, if any. Safe to call more than once. */
  cleanup: () => void
}

/** Maximum characters of input text preserved in {@link PendingInvocation.preview}. */
const PREVIEW_MAX_CHARS = 200

/**
 * Collapses whitespace runs (including newlines) to single spaces and truncates to
 * {@link PREVIEW_MAX_CHARS}, marking the cut. Collapsing keeps the preview a single
 * line wherever it is rendered — in particular a queued request cannot inject
 * line-structured text into the model-facing pending-invocations block. Truncation
 * splits on code points, never inside a surrogate pair.
 */
function truncatePreview(text: string): string {
  const collapsed = text.replace(/\s+/g, ' ').trim()
  const codePoints = [...collapsed]
  return codePoints.length <= PREVIEW_MAX_CHARS ? collapsed : `${codePoints.slice(0, PREVIEW_MAX_CHARS).join('')}…`
}

/** Collects `text` fields from a content-block-like or message-like array element. */
function textOf(element: unknown): string[] {
  if (typeof element !== 'object' || element === null) return []
  if ('text' in element && typeof element.text === 'string') return [element.text]
  if ('content' in element && Array.isArray(element.content)) return element.content.flatMap(textOf)
  return []
}

/**
 * Derives a short human/model-readable preview from invocation arguments.
 * Text is extracted from string input, content blocks, and message content;
 * inputs with no extractable text yield a bracketed placeholder.
 */
export function previewInvokeArgs(args: InvokeArgs): string {
  if (typeof args === 'string') return truncatePreview(args)
  if (Array.isArray(args)) {
    const text = args.flatMap(textOf).join(' ').trim()
    if (text.length > 0) return truncatePreview(text)
  }
  return '[structured input]'
}

/**
 * FIFO queue of invocations waiting for the agent's invocation lock.
 *
 * All mutating methods are synchronous (no interior `await`), so on a single JS thread
 * no interleaving can observe a half-applied transition — the correctness argument for
 * the lock handoff in `stream()`'s `finally`.
 *
 * @internal Used by `Agent`; consumers observe it through `agent.pendingInvocations`.
 */
export class InvocationQueue {
  private readonly _entries: QueueEntry[] = []
  private _nextSequence = 1
  private readonly _enqueueListeners = new Set<() => void>()

  /** Number of invocations currently waiting. */
  get size(): number {
    return this._entries.length
  }

  /** Point-in-time snapshot of the queue, in run order. */
  snapshot(): readonly PendingInvocation[] {
    return this._entries.map(({ id, submittedAt, preview }) => Object.freeze({ id, submittedAt, preview }))
  }

  /**
   * Registers a listener invoked whenever an invocation enters the queue.
   * Listeners must not throw.
   *
   * @param listener - Called synchronously on each enqueue
   * @returns A function that detaches the listener
   */
  onEnqueue(listener: () => void): () => void {
    this._enqueueListeners.add(listener)
    return (): void => {
      this._enqueueListeners.delete(listener)
    }
  }

  /**
   * Adds a waiter and returns a promise that resolves when the invocation lock is
   * handed to it (via {@link handoff}), or rejects when the entry is removed first.
   *
   * @param args - The invocation arguments, used to derive the entry's preview
   * @param options - `front` inserts at the front of the queue (for `'cancelPrevious'`);
   *   `cancelSignal` is the caller's signal — aborting while queued removes the entry and
   *   rejects with {@link PendingInvocationCancelledError}
   */
  wait(args: InvokeArgs, options?: { front?: boolean; cancelSignal?: AbortSignal }): Promise<void> {
    const id = `pending-${this._nextSequence++}`
    return new Promise<void>((resolve, reject) => {
      const entry: QueueEntry = {
        id,
        submittedAt: new Date(),
        preview: previewInvokeArgs(args),
        resolve,
        reject,
        cleanup: () => {},
      }

      const signal = options?.cancelSignal
      if (signal) {
        if (signal.aborted) {
          reject(new PendingInvocationCancelledError(id))
          return
        }
        const onAbort = (): void => this._remove(entry)
        signal.addEventListener('abort', onAbort, { once: true })
        entry.cleanup = (): void => signal.removeEventListener('abort', onAbort)
      }

      if (options?.front) {
        this._entries.unshift(entry)
      } else {
        this._entries.push(entry)
      }
      for (const listener of [...this._enqueueListeners]) listener()
    })
  }

  /**
   * Hands the invocation lock to the next waiter, if any.
   *
   * @returns `true` when a waiter took ownership (the busy flag must stay set),
   *   `false` when the queue is empty (the caller releases the lock).
   */
  handoff(): boolean {
    const next = this._entries.shift()
    if (!next) return false
    next.cleanup()
    next.resolve()
    return true
  }

  /**
   * Removes a queued entry by id, rejecting its caller with
   * {@link PendingInvocationCancelledError}.
   *
   * @returns `true` when the entry was found and removed, `false` otherwise
   */
  cancel(id: string): boolean {
    const entry = this._entries.find((e) => e.id === id)
    if (!entry) return false
    this._remove(entry)
    return true
  }

  /** Removes an entry, detaches its abort listener, and rejects its caller. */
  private _remove(entry: QueueEntry): void {
    const index = this._entries.indexOf(entry)
    if (index === -1) return
    this._entries.splice(index, 1)
    entry.cleanup()
    entry.reject(new PendingInvocationCancelledError(entry.id))
  }
}
