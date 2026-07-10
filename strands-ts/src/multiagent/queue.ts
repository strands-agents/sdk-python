import type { Node } from './nodes.js'
import type { MultiAgentStreamEvent } from './events.js'
import type { NodeResult } from './state.js'

/**
 * Data produced by a running node: a streaming event, a completion signal, or an error.
 */
export type QueueData =
  | { type: 'event'; node: Node; event: MultiAgentStreamEvent }
  | { type: 'result'; node: Node; result: NodeResult }
  | { type: 'error'; node: Node; error: Error }

/**
 * Queue data paired with an acknowledgement callback.
 * The consumer must call {@link ack} after fully processing the data
 * to unblock any producer waiting via {@link Queue.send}.
 */
export interface QueueEntry {
  data: QueueData
  ack: () => void
}

/**
 * Async queue with promise-based notification and optional back-pressure.
 *
 * Producers use {@link push} for fire-and-forget or {@link send} to
 * block until the consumer has fully processed the data. The consumer calls
 * {@link shift} to dequeue, then {@link QueueEntry.ack} after
 * processing to unblock the producer.
 */
export class Queue {
  private readonly _entries: QueueEntry[] = []
  /** Resolve function for the pending wait() promise, if any. */
  private _notify?: (() => void) | undefined
  private _disposed = false

  /**
   * Push data to the queue, waking any waiting consumer.
   */
  push(data: QueueData): void {
    this._entries.push({ data, ack: () => {} })
    this._notify?.()
    this._notify = undefined
  }

  /**
   * Push data and wait until the consumer has fully processed it.
   * Provides back-pressure so the producer pauses until the event
   * has been yielded and hook callbacks have been invoked.
   *
   * @param data - The queue data to push
   * @returns Promise that resolves when the consumer calls {@link QueueEntry.ack}
   */
  send(data: QueueData): Promise<void> {
    if (this._disposed) return Promise.resolve()

    return new Promise((resolve) => {
      this._entries.push({ data, ack: resolve })
      this._notify?.()
      this._notify = undefined
    })
  }

  /**
   * Wait until at least one entry is available.
   */
  wait(): Promise<void> {
    if (this._entries.length > 0) return Promise.resolve()
    return new Promise((resolve) => {
      this._notify = resolve
    })
  }

  /**
   * Remove and return the next entry, or undefined if empty.
   */
  shift(): QueueEntry | undefined {
    return this._entries.shift()
  }

  /**
   * Dispose the queue by resolving all pending acks and draining entries.
   * Future {@link send} calls resolve immediately.
   */
  dispose(): void {
    this._disposed = true
    while (this._entries.length > 0) {
      this._entries.shift()!.ack()
    }
  }

  /**
   * Number of entries in the queue.
   */
  get size(): number {
    return this._entries.length
  }
}
