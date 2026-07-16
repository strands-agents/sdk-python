/**
 * File-based memory store implementing the {@link MemoryStore} interface.
 *
 * Organizes knowledge as a structured file hierarchy under `knowledge/`. Provides
 * keyword-based search via `search_memory` (registered by {@link MemoryManager}).
 */

import type { JSONValue } from '../../types/json.js'
import type { MemoryEntry, MemoryStore, SearchOptions } from '../../memory/types.js'
import type { ExtractionConfig } from '../../memory/extraction/types.js'
import type { Storage } from '../../storage/storage.js'
import type { FileMemoryStoreConfig } from './types.js'
import { LocalFileStorage } from '../../storage/local-file-storage.js'
import { DEFAULT_MAX_SEARCH_RESULTS, tokenize } from '../../memory/search/keyword.js'

const KNOWLEDGE_PREFIX = 'knowledge/'
const FACTS_PREFIX = `${KNOWLEDGE_PREFIX}facts/`

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
 * markdown files with YAML frontmatter under `knowledge/`. Retrieval is via the `search_memory`
 * tool registered by {@link MemoryManager}, which calls {@link search} (keyword-based).
 *
 * The storage backend defaults to {@link LocalFileStorage} (writing to `./.strands/`) when no
 * custom {@link Storage} implementation is provided.
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
    this._storage = config.storage ?? new LocalFileStorage()
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

    const allKeys = await this._storage.list(KNOWLEDGE_PREFIX)

    const scored = (
      await Promise.all(
        allKeys.map(async (key) => {
          try {
            const bytes = await this._storage.read(key)
            if (!bytes) return null

            const content = decoder.decode(bytes)
            const { description, body } = parseFrontmatter(content)
            const searchable = `${basename(key)} ${description} ${body}`

            let relevanceScore = 0
            for (const token of tokenize(searchable)) {
              if (queryTokens.has(token)) relevanceScore++
            }

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
      )
    ).filter((s): s is { entry: MemoryEntry; relevanceScore: number } => s !== null)

    scored.sort((a, b) => b.relevanceScore - a.relevanceScore)
    return scored.slice(0, maxResults).map((s) => s.entry)
  }

  /**
   * Add a knowledge entry to the store.
   *
   * Writes a markdown file with YAML frontmatter. By default writes to `knowledge/facts/`.
   * Pass `metadata.path` to write to a custom location under `knowledge/`.
   *
   * @param content - The knowledge content to store
   * @param metadata - Optional metadata: `title`, `description`, and `path` (custom target path)
   */
  async add(content: string, metadata?: Record<string, JSONValue>): Promise<string> {
    const customPath = metadata?.['path'] as string | undefined
    const firstSentence = content.split(/[.\n]/)[0]!
    const title = (metadata?.['title'] as string | undefined) ?? firstSentence.slice(0, 60)
    const description = (metadata?.['description'] as string | undefined) ?? firstSentence.slice(0, 120)

    let key: string
    if (customPath) {
      key = customPath.startsWith(KNOWLEDGE_PREFIX) ? customPath : `${KNOWLEDGE_PREFIX}${customPath}`
      if (!key.endsWith('.md')) key += '.md'
    } else {
      const slug = slugify(title) || `entry-${Date.now()}`
      key = `${FACTS_PREFIX}${slug}.md`

      // Best-effort collision avoidance for a single-writer local store (TOCTOU is acceptable).
      const existingKeys = new Set(await this._storage.list(FACTS_PREFIX))
      let suffix = 1
      while (existingKeys.has(key)) {
        key = `${FACTS_PREFIX}${slug}-${suffix}.md`
        suffix++
      }
    }

    const fileContent = `---\ndescription: ${JSON.stringify(description)}\n---\n\n${content}\n`
    await this._storage.write(key, encoder.encode(fileContent))
    return key
  }
}
