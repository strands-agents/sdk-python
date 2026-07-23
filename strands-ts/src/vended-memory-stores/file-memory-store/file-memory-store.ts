/**
 * File-based memory store implementing the {@link MemoryStore} interface.
 *
 * Organizes knowledge as a structured file hierarchy under a `memory/` storage namespace. Provides
 * keyword-based search via `search_memory` (registered by {@link MemoryManager}).
 */

import type { JSONValue } from '../../types/json.js'
import type { MemoryEntry, MemoryStore, SearchOptions } from '../../memory/types.js'
import type { ExtractionConfig } from '../../memory/extraction/types.js'
import type { Storage } from '../../storage/storage.js'
import type { FileMemoryStoreConfig } from './types.js'
import { LocalFileStorage } from '../../storage/local-file-storage.js'
import { NAMESPACED, namespace, normalizeKey } from '../../storage/storage.js'
import { DEFAULT_MAX_SEARCH_RESULTS, tokenize, tokenOverlapScore } from '../../memory/search/keyword.js'

/**
 * Top-level storage namespace shared by every file memory store, isolating them as a group from
 * other subsystems (sessions, context offloading) that may share the same backend. See the storage
 * design doc's key-prefix convention (`team/designs/0014-storage.md`). Each store further scopes
 * under its own `name` within this namespace — see {@link FileMemoryStore._resolveStorage}.
 */
const STORAGE_NAMESPACE = 'memory'

/** Default subdirectory (within the store's namespace) for entries added without an explicit path. */
const FACTS_PREFIX = 'facts/'

/**
 * Cap on concurrent storage reads during search. The Storage contract makes no guarantee
 * about concurrent-read capacity, so an unbounded fan-out (one read per key) can exhaust a
 * backend's connection pool or trip throttling on a large corpus. Reads still run in parallel,
 * just no more than this many at once.
 */
const SEARCH_READ_CONCURRENCY = 8

const encoder = new TextEncoder()
const decoder = new TextDecoder()

/** Extract description from YAML frontmatter and return the remaining body. */
function parseFrontmatter(content: string): { description: string; body: string } {
  const match = content.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/)
  if (!match) return { description: '', body: content }

  const frontmatter = match[1] ?? ''
  const body = match[2] ?? ''

  const descMatch = frontmatter.match(/^description:\s*(".*")\s*$/m)
  if (!descMatch) return { description: '', body }

  let description: string
  try {
    description = JSON.parse(descMatch[1]!) as string
  } catch {
    description = descMatch[1]!.slice(1, -1)
  }
  return { description, body }
}

/** Extract the filename stem (without `.md` extension) from a storage key. */
function basename(key: string): string {
  const filename = key.split('/').pop() ?? key
  return filename.replace(/\.md$/, '')
}

/**
 * Map `items` through `fn` running at most `limit` calls concurrently, preserving input order.
 * A worker pool pulls from a shared cursor so a slow item never blocks others in its batch.
 */
async function mapWithConcurrency<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length)
  let cursor = 0
  const worker = async (): Promise<void> => {
    while (cursor < items.length) {
      const index = cursor++
      results[index] = await fn(items[index]!)
    }
  }
  const workers = Array.from({ length: Math.min(limit, items.length) }, () => worker())
  await Promise.all(workers)
  return results
}

/** Convert text to a URL-safe kebab-case slug, truncated to 50 characters. */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
    .slice(0, 50)
}

/**
 * A file-based memory store backed by the unified {@link Storage} interface.
 *
 * Implements {@link MemoryStore} for use with {@link MemoryManager}. Knowledge is stored as
 * markdown files with YAML frontmatter under a `memory/` storage namespace. Retrieval is via the
 * `search_memory` tool registered by {@link MemoryManager}, which calls {@link search} (keyword-based).
 *
 * The storage backend defaults to {@link LocalFileStorage} when no custom {@link Storage}
 * implementation is provided. Keys are auto-scoped under `memory/<name>/` (so a store named
 * `agent-memory` with the default backend lands knowledge under `./.strands/memory/agent-memory/`),
 * isolating it from other subsystems — and from differently-named file memory stores — that share
 * the same backend. Two stores with the same name on one backend share storage. Pass a storage view
 * that is already namespaced to override this.
 *
 * @example
 * ```typescript
 * import { Agent, MemoryManager } from '@strands-agents/sdk'
 * import { FileMemoryStore } from '@strands-agents/sdk/vended-memory-stores/file-memory-store'
 *
 * const memoryStore = new FileMemoryStore({ name: 'agent-memory' })
 *
 * const agent = new Agent({
 *   model,
 *   memoryManager: new MemoryManager({ stores: [memoryStore], injection: false }),
 * })
 * ```
 */
export class FileMemoryStore implements MemoryStore {
  readonly name: string
  readonly writable: boolean
  readonly description?: string
  readonly maxSearchResults?: number
  readonly extraction?: boolean | ExtractionConfig

  private readonly _storage: Storage

  constructor(config: FileMemoryStoreConfig) {
    this.name = config.name
    this.writable = config.writable ?? true
    if (config.description !== undefined) this.description = config.description
    if (config.maxSearchResults !== undefined) this.maxSearchResults = config.maxSearchResults
    if (config.extraction !== undefined) this.extraction = config.extraction
    this._storage = this._resolveStorage(config.storage ?? new LocalFileStorage())
  }

  /**
   * Auto-scopes keys under `memory/<name>/` so this store never collides with other subsystems
   * (sessions, context offloading) sharing the same backend, nor with a differently-named file
   * memory store on it — distinct {@link name}s yield non-overlapping scopes. Two stores sharing
   * both a name and a backend still share storage. Storage that is already namespaced — e.g. handed
   * down pre-scoped by a future router — is used as-is, so scoping never stacks twice.
   */
  private _resolveStorage(storage: Storage): Storage {
    if (NAMESPACED in storage) return storage
    const prefix = `${STORAGE_NAMESPACE}/${this.name}`
    if (storage.namespace) return storage.namespace(prefix)
    return namespace(storage, prefix)
  }

  /**
   * Search knowledge files by keyword matching against filenames, descriptions, and content.
   *
   * Returns the top matches ranked by distinct token overlap. Each result's `metadata.path`
   * reflects the entry's current storage location and may change after consolidation.
   */
  async search(query: string, options?: SearchOptions): Promise<MemoryEntry[]> {
    const maxResults = options?.maxSearchResults ?? this.maxSearchResults ?? DEFAULT_MAX_SEARCH_RESULTS
    const queryTokens = tokenize(query)
    if (queryTokens.size === 0) return []

    const allKeys = await this._storage.list('')

    const scored = (
      await mapWithConcurrency(allKeys, SEARCH_READ_CONCURRENCY, async (key) => {
        try {
          const bytes = await this._storage.read(key)
          if (!bytes) return null

          const content = decoder.decode(bytes)
          const { description, body } = parseFrontmatter(content)
          const searchable = `${basename(key)} ${description} ${body}`

          const relevanceScore = tokenOverlapScore(queryTokens, searchable)
          if (relevanceScore === 0) return null
          return {
            entry: {
              content: body.trim(),
              metadata: { path: key, description, _relevanceScore: relevanceScore },
            } as MemoryEntry,
            relevanceScore,
          }
        } catch {
          return null
        }
      })
    ).filter((s): s is { entry: MemoryEntry; relevanceScore: number } => s !== null)

    scored.sort((a, b) => b.relevanceScore - a.relevanceScore)
    return scored.slice(0, maxResults).map((s) => s.entry)
  }

  /**
   * Add a knowledge entry to the store.
   *
   * Writes a markdown file with YAML frontmatter. By default writes to `facts/` within the store's
   * namespace. Pass `metadata.path` to write to a custom location within the namespace.
   *
   * @param content - The knowledge content to store
   * @param metadata - Optional metadata: `title`, `description`, and `path` (custom target path)
   * @returns The canonical storage-relative key the entry was written under, normalized to
   *   match what {@link search} and the backend's `list` report (slash runs collapsed, leading
   *   and trailing slashes stripped)
   */
  async add(content: string, metadata?: Record<string, JSONValue>): Promise<string> {
    const customPath = metadata?.['path'] as string | undefined
    const firstSentence = content.split(/[.\n]/)[0]!
    const title = (metadata?.['title'] as string | undefined) ?? firstSentence.slice(0, 60)
    const description = (metadata?.['description'] as string | undefined) ?? firstSentence.slice(0, 120)

    let key: string
    if (customPath) {
      key = customPath
      if (!key.endsWith('.md')) key += '.md'
    } else {
      const slug = slugify(title) || `entry-${Date.now()}`
      key = `${FACTS_PREFIX}${slug}.md`

      // Probe with read() so the backend resolves key identity: a case-insensitive
      // filesystem treats Topic.md and topic.md as the same file, which comparing
      // against list()'s exact key spellings in memory would miss. A miss returns null
      // without transferring a body, so the only full reads are on genuine collisions.
      // Best-effort for a single-writer local store (TOCTOU is acceptable).
      let suffix = 1
      while (await this._storage.read(key)) {
        key = `${FACTS_PREFIX}${slug}-${suffix}.md`
        suffix++
      }
    }

    // Canonicalize with the same helper the shipped backends apply internally, so the
    // returned receipt matches the key search() and the backend's list() report.
    const canonicalKey = normalizeKey(key)
    const fileContent = `---\ndescription: ${JSON.stringify(description)}\n---\n\n${content}\n`
    await this._storage.write(canonicalKey, encoder.encode(fileContent))
    return canonicalKey
  }
}
