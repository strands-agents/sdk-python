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

const KNOWLEDGE_PREFIX = 'knowledge/'
const FACTS_PREFIX = `${KNOWLEDGE_PREFIX}facts/`

const encoder = new TextEncoder()
const decoder = new TextDecoder()

function parseFrontmatter(content: string): { description: string; body: string } {
  const match = content.match(/^---\n([\s\S]*?)\n---\n?([\s\S]*)$/)
  if (!match) return { description: '', body: content }

  const frontmatter = match[1] ?? ''
  const body = match[2] ?? ''

  const descMatch = frontmatter.match(/^description:\s*["']?(.+?)["']?\s*$/m)
  return { description: descMatch?.[1] ?? '', body }
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
    .slice(0, 50)
}

/**
 * A zero-infrastructure memory store backed by a file hierarchy.
 *
 * Implements {@link MemoryStore} for use with {@link MemoryManager}. Knowledge is stored as
 * markdown files with YAML frontmatter under `knowledge/`. Retrieval is via the `search_memory`
 * tool registered by {@link MemoryManager}, which calls {@link search} (keyword-based).
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
  readonly description?: string
  readonly writable: boolean
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
   * Returns the top matches ranked by term frequency.
   */
  async search(query: string, options?: SearchOptions): Promise<MemoryEntry[]> {
    const maxResults = options?.maxSearchResults ?? this.maxSearchResults ?? 5
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean)
    if (terms.length === 0) return []

    const allKeys = await this._storage.list(KNOWLEDGE_PREFIX)
    const scored: Array<{ entry: MemoryEntry; score: number }> = []

    for (const key of allKeys) {
      const bytes = await this._storage.read(key)
      if (!bytes) continue

      const content = decoder.decode(bytes)
      const { description, body } = parseFrontmatter(content)
      const searchable = `${key} ${description} ${body}`.toLowerCase()

      let score = 0
      for (const term of terms) {
        const matches = searchable.split(term).length - 1
        score += matches
      }

      if (score > 0) {
        scored.push({
          entry: { content: body.trim(), metadata: { path: key, description } },
          score,
        })
      }
    }

    scored.sort((a, b) => b.score - a.score)
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
  async add(content: string, metadata?: Record<string, JSONValue>): Promise<void> {
    const customPath = metadata?.['path'] as string | undefined
    const title = (metadata?.['title'] as string | undefined) ?? slugify(content.split(/[.\n]/)[0]!.slice(0, 60))
    const description = (metadata?.['description'] as string | undefined) ?? content.split(/[.\n]/)[0]!.slice(0, 120)

    let key: string
    if (customPath) {
      key = customPath.startsWith(KNOWLEDGE_PREFIX) ? customPath : `${KNOWLEDGE_PREFIX}${customPath}`
      if (!key.endsWith('.md')) key += '.md'
    } else {
      const slug = slugify(title) || `entry-${Date.now()}`
      key = `${FACTS_PREFIX}${slug}.md`
    }

    const fileContent = `---\ndescription: "${description.replace(/"/g, '\\"')}"\n---\n\n${content}\n`
    await this._storage.write(key, encoder.encode(fileContent))
  }
}
