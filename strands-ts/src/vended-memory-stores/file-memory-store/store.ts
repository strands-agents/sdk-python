/**
 * File-based memory store implementing the {@link MemoryStore} interface.
 *
 * Stores knowledge as markdown files under a `memory/` storage namespace. Provides keyword-based
 * search via `search_memory` (registered by {@link MemoryManager}).
 */

import type { MemoryEntry, MemoryStore, MemoryStoreConfig, SearchOptions } from '../../memory/types.js'
import type { ExtractionConfig, ExtractionResult, Extractor, ExtractorContext } from '../../memory/extraction/types.js'
import type { JSONValue } from '../../types/json.js'
import type { MessageData } from '../../types/messages.js'
import type { Storage } from '../../storage/storage.js'

/**
 * Configuration for {@link FileMemoryStore}.
 */
export interface FileMemoryStoreConfig extends MemoryStoreConfig {
  /**
   * The unified Storage backend for file operations. Defaults to LocalFileStorage at `./.strands/`.
   * Keys are auto-scoped under `memory/<name>/` unless the provided storage is already namespaced, so
   * stores with distinct names safely share one backend. Two stores with the same name on the same
   * backend share storage — give them different names (or separate storage) to isolate them.
   */
  storage?: Storage
}
import { LocalFileStorage } from '../../storage/local-file-storage.js'
import { ModelExtractor } from '../../memory/extraction/model-extractor.js'
import { normalizeKey, resolveNamespace, tokenize, tokenOverlapScore } from '../../storage/storage.js'
import { logger } from '../../logging/logger.js'

const DEFAULT_EXTRACTION_PROMPT = `You extract durable facts worth remembering across future conversations from a transcript.

Return ONLY a JSON array of objects: {"content": string}.

Group related facts into a single entry. The first line is a markdown heading (e.g. "# User preferences", "# Project setup", "# Team conventions"). Put each fact on its own line below the heading.

If there is nothing worth remembering, return [].`

const DEFAULT_MAX_SEARCH_RESULTS = 10
const STORAGE_NAMESPACE = 'memory'
/** Cap concurrent reads to avoid throttling on remote storage backends (e.g. S3). */
const BATCH_SIZE = 8
const encoder = new TextEncoder()
const decoder = new TextDecoder()

/** Extract the filename stem (without `.md` extension) from a storage key. */
function basename(key: string): string {
  const filename = key.split('/').pop() ?? key
  return filename.replace(/\.md$/, '')
}

/** Convert text to a URL-safe kebab-case slug, truncated to 50 characters. */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
    .slice(0, 50)
    .replace(/-+$/, '')
}

/** Creates an extractor that injects existing topic headings so the model reuses them. */
function createKeyAwareExtractor(storage: Storage): Extractor {
  return {
    async extract(messages: MessageData[], context?: ExtractorContext): Promise<ExtractionResult[]> {
      const existingKeys = await storage.list('')
      const headings = existingKeys.map((key) => basename(key).replace(/-/g, ' '))

      let systemPrompt = DEFAULT_EXTRACTION_PROMPT
      if (headings.length > 0) {
        systemPrompt += `\n\nExisting topics: ${headings.join(', ')}. Reuse an existing topic heading when new facts belong to it.`
      }

      return new ModelExtractor({ systemPrompt }).extract(messages, context)
    },
  }
}

/**
 * A file-based memory store backed by the unified {@link Storage} interface.
 *
 * Implements {@link MemoryStore} for use with {@link MemoryManager}. Knowledge is stored as plain
 * markdown files under a `memory/` storage namespace. Retrieval uses keyword-based token-overlap
 * scoring against filename and body content.
 *
 * The storage backend defaults to {@link LocalFileStorage}. Keys are auto-scoped under
 * `memory/<name>/` (so a store named `agent-memory` with the default backend writes to
 * `./.strands/memory/agent-memory/`).
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
  private readonly _writeLocks = new Map<string, Promise<string>>()

  constructor(config: FileMemoryStoreConfig) {
    this.name = config.name
    this.writable = config.writable ?? true
    if (config.description !== undefined) this.description = config.description
    if (config.maxSearchResults !== undefined) this.maxSearchResults = config.maxSearchResults
    this._storage = this._resolveStorage(config.storage ?? new LocalFileStorage())
    const extraction = this._resolveExtraction(config)
    if (extraction !== undefined) this.extraction = extraction
  }

  private _resolveExtraction(config: FileMemoryStoreConfig): boolean | ExtractionConfig | undefined {
    if (!config.extraction) return config.extraction
    if (config.extraction === true) {
      return { extractor: createKeyAwareExtractor(this._storage) }
    }
    if (!config.extraction.extractor) {
      return { ...config.extraction, extractor: createKeyAwareExtractor(this._storage) }
    }
    return config.extraction
  }

  private _resolveStorage(storage: Storage): Storage {
    return resolveNamespace(storage, `${STORAGE_NAMESPACE}/${this.name}`)
  }

  /**
   * Search knowledge files by delegating to the storage backend's `search()` when available,
   * or falling back to a local keyword token-overlap scan.
   *
   * @param query - Natural-language search query
   * @param options - Optional search configuration (e.g. maxSearchResults)
   * @returns Top matches ranked by relevance.
   */
  async search(query: string, options?: SearchOptions): Promise<MemoryEntry[]> {
    const maxResults = options?.maxSearchResults ?? this.maxSearchResults ?? DEFAULT_MAX_SEARCH_RESULTS

    if (this._storage.search) {
      const results = await this._storage.search(query)
      const entries: MemoryEntry[] = []
      for (const result of results.slice(0, maxResults)) {
        const bytes = await this._storage.read(result.key)
        if (bytes) {
          entries.push({
            content: decoder.decode(bytes).trim(),
            metadata: { path: result.key, score: result.score },
          } as MemoryEntry)
        }
      }
      return entries
    }

    return this._keywordSearch(query, maxResults)
  }

  private async _keywordSearch(query: string, maxResults: number): Promise<MemoryEntry[]> {
    const queryTokens = tokenize(query)
    if (queryTokens.size === 0) return []

    const allKeys = await this._storage.list('')

    const scored: Array<{ entry: MemoryEntry; relevanceScore: number }> = []
    for (let offset = 0; offset < allKeys.length; offset += BATCH_SIZE) {
      const batch = await Promise.all(
        allKeys.slice(offset, offset + BATCH_SIZE).map(async (key) => {
          try {
            const bytes = await this._storage.read(key)
            if (!bytes) return null

            const body = decoder.decode(bytes)
            const searchable = `${basename(key)} ${body}`

            const relevanceScore = tokenOverlapScore(queryTokens, searchable)
            if (relevanceScore === 0) return null
            return {
              entry: { content: body.trim(), metadata: { path: key, score: relevanceScore } } as MemoryEntry,
              relevanceScore,
            }
          } catch (error) {
            logger.debug(`key=<${key}> | skipped unreadable memory entry: ${error}`)
            return null
          }
        })
      )
      for (const result of batch) {
        if (result) scored.push(result)
      }
    }

    scored.sort((a, b) => b.relevanceScore - a.relevanceScore)
    return scored.slice(0, maxResults).map((s) => s.entry)
  }

  /**
   * Add a knowledge entry to the store.
   *
   * The filename is derived from the first line of content (slugified, truncated to 50 chars).
   * If a file with the same slug already exists, new facts (lines after the heading) are appended
   * to it rather than overwriting. This keeps extraction token-cheap (only headings are injected
   * into the prompt) while avoiding data loss on repeated topics.
   *
   * @param content - The knowledge content to store
   * @param _metadata - Unused; accepted for interface compatibility with the ExtractionCoordinator
   * @returns The canonical storage key the entry was written under
   */
  async add(content: string, _metadata?: Record<string, JSONValue>): Promise<string> {
    const lines = content.split(/\n/)
    const firstLine = lines[0]!.replace(/^#+\s*/, '')
    const key = `${slugify(firstLine) || `entry-${Date.now()}`}.md`
    const canonicalKey = normalizeKey(key).toLowerCase()

    // Serialize writes per key to prevent lost-update from concurrent read-modify-write
    const prev = this._writeLocks.get(canonicalKey) ?? Promise.resolve('')
    const current = prev.then(async () => {
      const existing = await this._storage.read(canonicalKey)
      let merged: string
      if (existing) {
        const existingContent = decoder.decode(existing)
        const newFacts = lines.slice(1).join('\n').trim()
        merged = newFacts ? `${existingContent.trimEnd()}\n${newFacts}` : existingContent
      } else {
        merged = content
      }
      await this._storage.write(canonicalKey, encoder.encode(merged))
      return canonicalKey
    })
    this._writeLocks.set(canonicalKey, current)
    try {
      return await current
    } finally {
      if (this._writeLocks.get(canonicalKey) === current) {
        this._writeLocks.delete(canonicalKey)
      }
    }
  }
}
