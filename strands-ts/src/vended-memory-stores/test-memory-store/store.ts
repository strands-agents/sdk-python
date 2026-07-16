import { v7 as uuidv7 } from 'uuid'

import { InMemoryStorage } from '../../storage/in-memory-storage.js'
import { NAMESPACED, namespace } from '../../storage/storage.js'
import type { MemoryEntry, MemoryStore, MemoryStoreConfig, SearchOptions } from '../../memory/types.js'
import type { ExtractionConfig } from '../../memory/extraction/types.js'
import type { JSONValue } from '../../types/json.js'
import type { Storage } from '../../storage/storage.js'

const DEFAULT_MAX_SEARCH_RESULTS = 10

/**
 * Metadata key holding the token-overlap relevance score on a search result.
 */
const RELEVANCE_SCORE_KEY = '_relevanceScore'

/**
 * A stored memory, as it is serialized into the record blob.
 */
interface TestMemoryRecord {
  id: string
  content: string
  metadata?: Record<string, JSONValue>
  createdAt: string
}

/**
 * Configuration for {@link TestMemoryStore}.
 */
export interface TestMemoryStoreConfig extends MemoryStoreConfig {
  /**
   * Storage backend the records are persisted through. Records are held as a single JSON blob
   * under the key `memory/<sanitized-store-name>.json`.
   *
   * Defaults to an ephemeral {@link InMemoryStorage} — entries live only in memory and are lost
   * when the process exits. Pass a `LocalFileStorage` (or any {@link Storage}) to persist across
   * restarts, e.g. `new LocalFileStorage()` to write under `./.strands/`.
   */
  storage?: Storage
}

/** Result returned by {@link TestMemoryStore.add}. */
export interface TestMemoryAddResult {
  /** The id of the stored record. */
  id: string
}

/**
 * Sanitizes a store name into a safe single storage-key segment, guarding against a name that
 * would escape the `memory/` prefix. Ensures cross-SDK consistent sanitization.
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
 * A zero-infrastructure {@link MemoryStore} that persists its records through a {@link Storage}
 * backend. Use for prototyping and testing.
 *
 * Recall is lexical: results are ranked by how many query tokens overlap an entry's content, with
 * the most recent entry winning ties. This is keyword matching, not the semantic search a managed
 * vector store (e.g. {@link BedrockKnowledgeBaseStore}) provides.
 *
 * Each {@link add} rewrites the whole record blob, so this fits modest volumes, not high-volume
 * production workloads. Use a managed store like {@link BedrockKnowledgeBaseStore} for that.
 *
 * The store defaults to an ephemeral {@link InMemoryStorage}: entries are lost when the process
 * exits. Pass a persistent {@link Storage} (e.g. `new LocalFileStorage()`) to keep them across
 * restarts.
 *
 * The serialized record format is shared with the Python SDK's `TestMemoryStore`: records use the
 * same camelCase keys (`id`, `content`, `metadata`, `createdAt`) and the same timestamp shape, so
 * a backing store written by either SDK can be read by the other.
 *
 * @example
 * ```typescript
 * import { TestMemoryStore } from '@strands-agents/sdk/vended-memory-stores/test-memory-store'
 * import { LocalFileStorage } from '@strands-agents/sdk/storage'
 *
 * // Ephemeral by default; pass a LocalFileStorage to persist under ./.strands/memory/notes.json.
 * const store = new TestMemoryStore({ name: 'notes', storage: new LocalFileStorage() })
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

  /** Storage backend the records are persisted through, scoped under the `memory/` namespace. */
  private readonly _storage: Storage
  /** Key within the backend the record blob is stored under: `<sanitized-store-name>.json`. */
  private readonly _key: string
  /** Serializes writes so concurrent `add`s never interleave the read-modify-write cycle. */
  private _writeChain: Promise<unknown> = Promise.resolve()

  constructor(options: TestMemoryStoreConfig) {
    const { name, description, maxSearchResults, writable, extraction, storage } = options

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

    // Ephemeral by default. Scope every backend under `memory/`, unless the caller already namespaced it.
    const backend = storage ?? new InMemoryStorage()
    this._storage = NAMESPACED in backend ? backend : namespace(backend, 'memory')
    this._key = `${sanitizeName(name)}.json`
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
   * @throws An `Error` if `options.maxSearchResults` is less than 1, or if the backing blob is
   *   malformed (invalid JSON, not an array, or a record missing required string fields).
   * @throws {@link StorageError} if the backend read fails.
   */
  async search(query: string, options?: SearchOptions): Promise<MemoryEntry[]> {
    if (options?.maxSearchResults !== undefined && options.maxSearchResults < 1) {
      throw new Error('TestMemoryStore: maxSearchResults must be at least 1.')
    }
    const limit = options?.maxSearchResults || this.maxSearchResults || DEFAULT_MAX_SEARCH_RESULTS

    const queryTokens = tokenize(query)
    if (queryTokens.size === 0) return []

    const records = await this._read()

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
   * @throws An `Error` if the store is not writable, if `content` is empty or whitespace, or if
   *   the existing backing blob is malformed (invalid JSON, not an array, or a record missing
   *   required string fields).
   * @throws {@link StorageError} if the backend read or write fails.
   */
  async add(content: string, metadata?: Record<string, JSONValue>): Promise<TestMemoryAddResult> {
    if (!this.writable) {
      throw new Error('TestMemoryStore: store is not writable. Set writable: true in config to enable add().')
    }
    if (!content.trim()) {
      throw new Error('TestMemoryStore: content must not be empty.')
    }

    // Serialize the whole read-modify-write cycle behind any in-flight write so concurrent `add`s
    // don't each read the same snapshot and clobber one another (last-write-wins). Reading inside
    // the chained callback guarantees add #N sees add #N-1's write.
    const run = this._writeChain.then(async () => {
      const records = await this._read()

      const normalizedContent = content.trim()
      const existing = records.find((record) => record.content.trim() === normalizedContent)
      if (existing) return { id: existing.id }

      const record: TestMemoryRecord = { id: uuidv7(), content, createdAt: new Date().toISOString() }
      if (metadata !== undefined) record.metadata = metadata

      await this._write([...records, record])
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
   * Reads and parses the record blob from storage; a missing key (or empty store) starts empty.
   * Reads fresh on every call — there is no in-memory cache, so a search always reflects the
   * latest write (including from another writer sharing the backend).
   *
   * @throws An `Error` if the stored blob is not valid JSON, is not an array, or holds a record
   *   missing the required string fields. A backend I/O failure surfaces as its own `StorageError`.
   */
  private async _read(): Promise<TestMemoryRecord[]> {
    const bytes = await this._storage.read(this._key)
    if (bytes === null) return []

    const rawContent = new TextDecoder().decode(bytes)
    let parsedBlob: unknown
    try {
      parsedBlob = JSON.parse(rawContent)
    } catch (error: unknown) {
      throw new Error(`TestMemoryStore: invalid JSON in ${this._key}`, { cause: error })
    }
    if (!Array.isArray(parsedBlob)) {
      throw new Error(`TestMemoryStore: invalid backing store ${this._key}: expected a JSON array of records`)
    }
    for (const record of parsedBlob) {
      if (
        record === null ||
        typeof record !== 'object' ||
        typeof record.id !== 'string' ||
        typeof record.content !== 'string' ||
        typeof record.createdAt !== 'string'
      ) {
        throw new Error(
          `TestMemoryStore: invalid backing store ${this._key}: ` +
            "each record must have string 'id', 'content', and 'createdAt' fields"
        )
      }
      if (
        record.metadata !== undefined &&
        (record.metadata === null || typeof record.metadata !== 'object' || Array.isArray(record.metadata))
      ) {
        throw new Error(
          `TestMemoryStore: invalid backing store ${this._key}: ` +
            "a record's 'metadata', when present, must be a JSON object"
        )
      }
    }
    return parsedBlob as TestMemoryRecord[]
  }

  /**
   * Persists `records` as a single JSON blob through the storage backend. Callers serialize
   * invocations via {@link _writeChain}; atomicity is the backend's responsibility. A backend I/O
   * failure surfaces as its own `StorageError`, naming the key.
   */
  private async _write(records: TestMemoryRecord[]): Promise<void> {
    const bytes = new TextEncoder().encode(JSON.stringify(records, null, 2))
    await this._storage.write(this._key, bytes)
  }
}
