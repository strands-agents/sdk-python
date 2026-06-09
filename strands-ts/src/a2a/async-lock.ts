/**
 * Minimal async mutex for serializing access to a resource.
 *
 * JavaScript has no built-in equivalent of Python's `asyncio.Lock`. This
 * implements the same contract: `acquire()` resolves only once all previously
 * acquired holders have released, so awaiting callers run one at a time in FIFO
 * order. Used by {@link A2AExecutor} to serialize same-context requests (one
 * lock per context in factory mode) and all requests in single-agent mode.
 */
export class AsyncLock {
  /** Resolves when the currently-held lock (if any) is released. */
  private _tail: Promise<void> = Promise.resolve()

  /**
   * Acquires the lock, waiting until all previously-acquired holders release.
   *
   * @returns A release function. Call it exactly once (e.g. in a `finally`) to
   *   let the next waiter proceed.
   */
  async acquire(): Promise<() => void> {
    let release!: () => void
    const next = new Promise<void>((resolve) => {
      release = resolve
    })

    // Wait on the current tail, then install ours so the next acquirer waits on us.
    const previous = this._tail
    this._tail = previous.then(() => next)
    await previous

    return release
  }
}
