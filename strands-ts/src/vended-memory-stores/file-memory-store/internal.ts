/**
 * Internals shared between {@link FileMemoryStore} and its `consolidation/` modules.
 *
 * These are the primitives both sides must agree on: the frontmatter format the store writes and
 * the validator enforces, the reserved changelog path every walk of the store excludes, and the
 * path-identity rule validation and execution both apply. They live here rather than on either
 * side so `consolidation/` never has to import from the store, which would form a cycle.
 *
 * @internal
 */

/** @internal */
export const encoder = new TextEncoder()

/** @internal */
export const decoder = new TextDecoder()

/**
 * Path (within the store's namespace) reserved for the consolidation audit log.
 *
 * @internal
 */
export const CONSOLIDATION_CHANGELOG = 'consolidation-changelog.md'

/**
 * Frontmatter opening delimiter. Matches the convention used by {@link FileMemoryStore.add}:
 * files start with `---\n`, followed by YAML fields, then a closing `---\n`.
 *
 * @internal
 */
export const FRONTMATTER_OPEN = '---\n'

/**
 * Frontmatter closing delimiter, including the newline that must precede it.
 *
 * @internal
 */
export const FRONTMATTER_CLOSE = '\n---\n'

/**
 * Cap on concurrent storage reads when fanning out over the store's keys. The Storage contract makes
 * no guarantee about concurrent-read capacity, so an unbounded fan-out (one read per key) can exhaust
 * a backend's connection pool or trip throttling on a large corpus. Reads still run in parallel,
 * just no more than this many at once.
 *
 * @internal
 */
export const STORAGE_READ_CONCURRENCY = 8

/**
 * The only `description` form {@link parseFrontmatter} reads. Shared with consolidation's write
 * validation so a plan cannot write a description this parser would read as empty.
 *
 * @internal
 */
export const FRONTMATTER_DESCRIPTION_PATTERN = /^description:\s*(".*")\s*$/m

/**
 * Extract description from YAML frontmatter and return the remaining body.
 *
 * @internal
 */
export function parseFrontmatter(content: string): { description: string; body: string } {
  const match = content.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/)
  if (!match) return { description: '', body: content }

  const frontmatter = match[1] ?? ''
  const body = match[2] ?? ''

  const descMatch = frontmatter.match(FRONTMATTER_DESCRIPTION_PATTERN)
  if (!descMatch) return { description: '', body }

  const rawDesc = descMatch[1] ?? ''
  let description: string
  try {
    description = JSON.parse(rawDesc) as string
  } catch {
    description = rawDesc.slice(1, -1)
  }
  return { description, body }
}

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

/**
 * Whether a path contains single-dot segments that the OS would collapse — e.g. `./foo.md` resolves
 * to `foo.md`. Shared between the public `add()` path (which uses it to prevent changelog aliasing
 * that `normalizeKey` does not strip) and consolidation's `validatePath` (which already rejects both
 * `.` and `..`). Does NOT check `..` — that is handled by `normalizeKey` for `add()` and by
 * `validatePath` for plans.
 *
 * @internal
 */
export function containsDotSegments(key: string): boolean {
  return key.split('/').some((seg) => seg === '.')
}

/**
 * Whether a key addresses the reserved consolidation changelog. The changelog is an audit artifact
 * rather than knowledge, so every path that walks the store excludes it: {@link FileMemoryStore.search}
 * so it is never recalled as a memory, `readAllFiles` so consolidation never ingests and rewrites its
 * own log, and `validatePath` so no plan can clobber the audit trail.
 *
 * @internal
 */
export function isConsolidationChangelog(key: string): boolean {
  return key.toLowerCase() === CONSOLIDATION_CHANGELOG
}
