/**
 * Bounded-concurrency read fan-out shared by the store's search and consolidation paths. Lives here,
 * neutral of either caller, so search need not import from a consolidation-specific module.
 *
 * @internal
 */

/**
 * Cap on concurrent storage reads when fanning out over the store's keys. An unbounded fan-out can
 * exhaust a backend's connection pool or trip throttling on a large corpus.
 *
 * @internal
 */
export const STORAGE_READ_CONCURRENCY = 8

/**
 * Map `items` through `fn` running at most `limit` calls concurrently, preserving input order.
 * A worker pool pulls from a shared cursor so a slow item never blocks others in its batch.
 *
 * @internal
 */
export async function mapWithConcurrency<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length)
  let cursor = 0
  const worker = async (): Promise<void> => {
    while (cursor < items.length) {
      const index = cursor++
      // index is bounded by items.length — the item is always present
      const item = items[index] as T
      results[index] = await fn(item)
    }
  }
  const workers = Array.from({ length: Math.min(limit, items.length) }, () => worker())
  await Promise.all(workers)
  return results
}
