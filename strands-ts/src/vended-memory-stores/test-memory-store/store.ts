import { v7 as uuidv7 } from 'uuid'

import type { MemoryEntry, MemoryStore, MemoryStoreConfig, SearchOptions } from '../../memory/types.js'
import type { ExtractionConfig } from '../../memory/extraction/types.js'
import type { JSONValue } from '../../types/json.js'

const DEFAULT_MAX_SEARCH_RESULTS = 10

/**
 * Metadata key holding the token-overlap relevance score on a search result.
 */
const RELEVANCE_SCORE_KEY = '_relevanceScore'

/**
 * A stored memory, as it is persisted on disk.
 */
interface TestMemoryRecord {
  id: string
  content: string
  metadata?: Record<string, JSONValue>
  createdAt: string
}

/**
 * Configuration for {@link TestMemoryStore}.
 *
 * The store persists to disk by default so the memory records persist across restarts. Set
 * {@link persist} to `false` for an ephemeral, single session store (useful for e.g. testing).
 */
export interface TestMemoryStoreConfig extends MemoryStoreConfig {
  /**
   * Whether to persist entries to disk so they survive across sessions.
   * - `true` (default): writes are flushed to {@link path} (or the default location).
   * - `false`: entries live only in memory and are lost when the process exits.
   *
   * @defaultValue true
   */
  persist?: boolean
  /**
   * Full path to the JSON file backing this store. Defaults to
   * `~/.strands/memory/<sanitized-store-name>.json`. Ignored when {@link persist} is `false`.
   */
  path?: string
}

/** Result returned by {@link TestMemoryStore.add}. */
export interface TestMemoryAddResult {
  /** The id of the stored record. */
  id: string
}

/**
 * Sanitizes a store name into a safe single-path-segment filename.
 * Guards the default-path branch against a name that would escape the memory directory.
 * Ensures cross-SDK consistent sanitization.
 */
function sanitizeName(name: string): string {
  return name
    .replace(/\.\./g, '_')
    .replace(/[/\\]/g, '_')
    .replace(/[^\w\-.]/g, '_')
}

/**
 * Lowercases and splits text into a set of word tokens, dropping empties. Splits on any run of
 * characters that are not Unicode letters, numbers, or underscore. Ensures cross-SDK consistent
 * tokenization.
 */
function tokenize(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .split(/[^\p{L}\p{N}_]+/u)
      .filter(Boolean)
  )
}

/**
 * Lexical relevance score for one record: the number of distinct query tokens that appear in the
 * record's content. A higher count means more of the query's words are present. Returns 0 when there
 * is no overlap.
 */
function tokenOverlapScore(queryTokens: Set<string>, content: string): number {
  let score = 0
  for (const token of tokenize(content)) {
    if (queryTokens.has(token)) score++
  }
  return score
}

/**
 * A zero-infrastructure store {@link MemoryStore} that keeps entries in memory and by default
 * persists them to a local JSON file. Use for prototyping and testing.
 *
 * Recall is lexical: results are ranked by how many query tokens overlap an entry's content, with
 * the most recent entry winning ties. This is keyword matching, not the semantic search a managed
 * vector store (e.g. {@link BedrockKnowledgeBaseStore}) provides.
 *
 * Each {@link add} rewrites the whole file, so this fits modest volumes, not fit for high volume
 * production workloads. Use a managed store like {@link BedrockKnowledgeBaseStore} for that.
 *
 * The on-disk format is shared with the Python SDK's `TestMemoryStore`: records use the same
 * camelCase keys (`id`, `content`, `metadata`, `createdAt`) and the same timestamp shape, so a
 * backing file written by either SDK can be read by the other.
 *
 * @example
 * ```typescript
 * import { TestMemoryStore } from '@strands-agents/sdk/vended-memory-stores/test-memory-store'
 *
 * // Persists to ~/.strands/memory/notes.json by default.
 * const store = new TestMemoryStore({ name: 'notes' })
 *
 * const { id } = await store.add('User prefers dark mode')
 * const results = await store.search('what theme does the user like?')
 * ```
 */
export class TestMemoryStore implements MemoryStore {
  readonly name: string
  readonly description?: string
  readonly maxSearchResults?: number
  readonly writable: boolean
  readonly extraction?: boolean | ExtractionConfig

  private readonly _persist: boolean
  /** Explicit `path` override from config, if any; the default path is resolved lazily in {@link _getPath}. */
  private readonly _explicitPath: string | undefined
  private _resolvedPath: string | undefined
  /** The loaded records once {@link _load} resolves; the working in-memory copy thereafter. */
  private _records: TestMemoryRecord[] | undefined
  /**
   * Memoizes the first (async) load so concurrent `search`/`add` callers share a single file read
   * instead of each racing their own — without it, a search interleaved with a first-use add could
   * overwrite the cache with a pre-write snapshot and drop the just-added record.
   */
  private _loadPromise: Promise<TestMemoryRecord[]> | undefined
  /** Serializes writes so concurrent `add`s never interleave the load-modify-flush cycle. */
  private _writeChain: Promise<unknown> = Promise.resolve()

  constructor(options: TestMemoryStoreConfig) {
    const { name, description, maxSearchResults, writable, extraction, persist, path } = options

    if (!name.trim()) {
      throw new Error('TestMemoryStore: name must not be empty.')
    }
    this.name = name
    if (description !== undefined) this.description = description
    if (maxSearchResults !== undefined) {
      if (maxSearchResults < 1) {
        throw new Error('TestMemoryStore: maxSearchResults must be at least 1.')
      }
      this.maxSearchResults = maxSearchResults
    }
    // A local store is writable by default.
    this.writable = writable ?? true
    if (extraction !== undefined) this.extraction = extraction

    if (path !== undefined && !path.trim()) {
      throw new Error('TestMemoryStore: path must not be empty.')
    }
    this._persist = persist ?? true
    this._explicitPath = path
  }

  /**
   * Searches stored entries for those whose content overlaps the query, ranked by token overlap with
   * the most recent entry winning ties.
   *
   * @param query - The search query text
   * @param options - Optional search configuration
   * @returns Matching memory entries ordered by relevance. Each entry's `metadata` includes a
   *   `_relevanceScore` key (the token-overlap count). An empty or token-less query returns
   *   no results.
   */
  async search(query: string, options?: SearchOptions): Promise<MemoryEntry[]> {
    if (options?.maxSearchResults !== undefined && options.maxSearchResults < 1) {
      throw new Error('TestMemoryStore: maxSearchResults must be at least 1.')
    }
    const limit = options?.maxSearchResults || this.maxSearchResults || DEFAULT_MAX_SEARCH_RESULTS

    const queryTokens = tokenize(query)
    if (queryTokens.size === 0) return []

    const records = await this._load()

    const scored: Array<{ record: TestMemoryRecord; score: number }> = []
    for (const record of records) {
      const score = tokenOverlapScore(queryTokens, record.content)
      if (score > 0) scored.push({ record, score })
    }

    scored.sort(
      (left, right) => right.score - left.score || right.record.createdAt.localeCompare(left.record.createdAt)
    )

    return scored.slice(0, limit).map(({ record, score }) => ({
      content: record.content,
      metadata: { ...record.metadata, [RELEVANCE_SCORE_KEY]: score },
    }))
  }

  /**
   * Adds `content` (with optional `metadata`) to the store. Identical content is deduplicated: a
   * repeat write returns the existing record's id without storing a second copy, so the at-least-once
   * retries that extraction may perform never accumulate duplicates.
   *
   * @param content - The text content to store
   * @param metadata - Optional metadata to attach to the entry. The key `_relevanceScore` is
   *   reserved: {@link search} populates it on results, so a value stored under it here is
   *   overwritten in search output.
   * @returns The id of the stored (or already-present) record
   */
  async add(content: string, metadata?: Record<string, JSONValue>): Promise<TestMemoryAddResult> {
    if (!this.writable) {
      throw new Error('TestMemoryStore: store is not writable. Set writable: true in config to enable add().')
    }
    if (!content.trim()) {
      throw new Error('TestMemoryStore: content must not be empty.')
    }

    // Serialize the whole load-modify-flush cycle behind any in-flight write so concurrent `add`s
    // don't each load the same snapshot and clobber one another (last-write-wins).
    const run = this._writeChain.then(async () => {
      const records = await this._load()

      const normalizedContent = content.trim()
      const existing = records.find((record) => record.content.trim() === normalizedContent)
      if (existing) return { id: existing.id }

      const record: TestMemoryRecord = { id: uuidv7(), content, createdAt: new Date().toISOString() }
      if (metadata !== undefined) record.metadata = metadata

      // Flush the candidate list first and only commit it to the in-memory cache once the write
      // succeeds, so a failed flush never leaves a phantom record that later writes resurrect.
      const next = [...records, record]
      await this._flush(next)
      this._records = next
      return { id: record.id }
    })
    // Keep the chain alive even if this write rejects, so a failed write doesn't wedge later ones.
    this._writeChain = run.then(
      () => undefined,
      () => undefined
    )
    return run
  }

  /**
   * Resolves (and caches) the backing-file path: the explicit `path` from config, else
   * `~/.strands/memory/<sanitized-store-name>.json`. Returns `undefined` for ephemeral stores. The
   * `node:os`/`node:path` imports are dynamic so the module stays safe to bundle for the browser.
   */
  private async _getPath(): Promise<string | undefined> {
    if (!this._persist) return undefined
    if (this._resolvedPath !== undefined) return this._resolvedPath
    if (this._explicitPath !== undefined) {
      this._resolvedPath = this._explicitPath
      return this._resolvedPath
    }
    const os = await import('node:os')
    const path = await import('node:path')
    this._resolvedPath = path.join(os.homedir(), '.strands', 'memory', `${sanitizeName(this.name)}.json`)
    return this._resolvedPath
  }

  /**
   * Loads records from disk on first use; ephemeral stores (and a missing file) start empty. The
   * first call's promise is memoized in {@link _loadPromise} so concurrent callers await one shared
   * read rather than each loading independently and racing to assign the cache.
   */
  private async _load(): Promise<TestMemoryRecord[]> {
    if (this._records !== undefined) return this._records
    if (this._loadPromise !== undefined) return this._loadPromise

    this._loadPromise = this._readFromDisk()
    try {
      this._records = await this._loadPromise
    } finally {
      this._loadPromise = undefined
    }
    return this._records
  }

  /** Reads and parses the backing file (or returns an empty list when ephemeral / missing). */
  private async _readFromDisk(): Promise<TestMemoryRecord[]> {
    const filePath = await this._getPath()
    if (filePath === undefined) return []

    const { readFile } = await import('node:fs/promises')
    let rawContent: string
    try {
      rawContent = await readFile(filePath, 'utf8')
    } catch (error: unknown) {
      if ((error as { code?: string }).code === 'ENOENT') return []
      throw new Error(`TestMemoryStore: failed to read ${filePath}`, { cause: error })
    }

    let parsedFile: unknown
    try {
      parsedFile = JSON.parse(rawContent)
    } catch (error: unknown) {
      throw new Error(`TestMemoryStore: invalid JSON in ${filePath}`, { cause: error })
    }
    if (!Array.isArray(parsedFile)) {
      throw new Error(`TestMemoryStore: invalid backing file ${filePath}: expected a JSON array of records`)
    }
    for (const record of parsedFile) {
      if (
        record === null ||
        typeof record !== 'object' ||
        typeof record.id !== 'string' ||
        typeof record.content !== 'string' ||
        typeof record.createdAt !== 'string'
      ) {
        throw new Error(
          `TestMemoryStore: invalid backing file ${filePath}: ` +
            "each record must have string 'id', 'content', and 'createdAt' fields"
        )
      }
    }
    return parsedFile as TestMemoryRecord[]
  }

  /**
   * Persists `records` to disk with an atomic write (write to a `.tmp` file, then rename) so a
   * crash mid-write can never leave a partially written file. A no-op for ephemeral stores. Callers
   * serialize invocations via {@link _writeChain}. Throws with the target path (and the OS error as
   * `cause`) when the path is unreachable or not writable.
   */
  private async _flush(records: TestMemoryRecord[]): Promise<void> {
    const filePath = await this._getPath()
    if (filePath === undefined) return

    const { mkdir, writeFile, rename } = await import('node:fs/promises')
    const { dirname } = await import('node:path')
    try {
      await mkdir(dirname(filePath), { recursive: true })
      const tmpPath = `${filePath}.tmp`
      await writeFile(tmpPath, JSON.stringify(records, null, 2), 'utf8')
      await rename(tmpPath, filePath)
    } catch (error: unknown) {
      throw new Error(`TestMemoryStore: failed to write ${filePath}`, { cause: error })
    }
  }
}
