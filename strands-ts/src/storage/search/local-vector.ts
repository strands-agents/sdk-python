import type { Storage, StorageSearchResult } from '../storage.js'
import type { Embedder, SearchStrategy } from './types.js'

import { cosineSimilarity } from './cosine.js'

/** Configuration for {@link LocalVectorSearchStrategy}. */
export interface LocalVectorSearchStrategyConfig {
  /** Function that produces embedding vectors from text. */
  embedder: Embedder
  /** Directory to store vector index files. Defaults to `<baseDir>/.vectors/`. */
  vectorDir?: string
  /** Maximum number of results to return. Defaults to 10. */
  maxResults?: number
}

/**
 * Local-filesystem vector search strategy using brute-force cosine similarity.
 *
 * Stores embedding vectors as JSON files under a `.vectors/` directory and computes
 * cosine similarity at query time. Suitable for small-to-medium datasets where an
 * external vector store is not needed.
 *
 * @example
 * ```typescript
 * import { LocalVectorSearchStrategy } from '@strands-agents/sdk/storage/search'
 * import { LocalFileStorage } from '@strands-agents/sdk/storage'
 *
 * const storage = new LocalFileStorage('./.strands/', undefined, {
 *   embeddings: { embedder: async (text) => embed(text) },
 * })
 * ```
 */
export class LocalVectorSearchStrategy implements SearchStrategy {
  private readonly _config: LocalVectorSearchStrategyConfig
  private readonly _vectorDir: string

  constructor(config: LocalVectorSearchStrategyConfig, baseDir: string) {
    this._config = config
    this._vectorDir = config.vectorDir ?? `${baseDir.replace(/\/$/, '')}/.vectors`
  }

  /**
   * Embeds content and stores the vector as a JSON file.
   *
   * @param _storage - The storage the content was written to (unused)
   * @param key - The storage key to associate the vector with
   * @param data - The raw bytes to embed
   */
  async index(_storage: Storage, key: string, data: Uint8Array): Promise<void> {
    const text = new TextDecoder().decode(data)
    const vector = await this._config.embedder(text)
    const vectorPath = this._vectorPath(key)
    const { mkdir, writeFile } = await import('node:fs/promises')
    const { dirname } = await import('node:path')
    await mkdir(dirname(vectorPath), { recursive: true })
    await writeFile(vectorPath, JSON.stringify(vector))
  }

  /**
   * Queries stored vectors for keys similar to the query using cosine similarity.
   *
   * @param _storage - The storage to search over (unused)
   * @param query - Natural-language search query
   * @returns Matched keys with similarity scores, ranked best-first
   */
  async search(_storage: Storage, query: string): Promise<StorageSearchResult[]> {
    const queryVector = await this._config.embedder(query)
    const maxResults = this._config.maxResults ?? 10
    const entries = await this._loadAllVectors()
    const scored: StorageSearchResult[] = []
    for (const [key, vector] of entries) {
      const score = cosineSimilarity(queryVector, vector)
      if (score > 0) scored.push({ key, score })
    }
    scored.sort((a, b) => b.score - a.score)
    return scored.slice(0, maxResults)
  }

  private _vectorPath(key: string): string {
    return `${this._vectorDir}/${key}.vec.json`
  }

  private async _loadAllVectors(): Promise<Array<[string, number[]]>> {
    const { readdir, readFile } = await import('node:fs/promises')
    const entries: Array<[string, number[]]> = []

    const walk = async (dir: string, prefix: string): Promise<void> => {
      let dirEntries
      try {
        dirEntries = await readdir(dir, { withFileTypes: true })
      } catch (error: unknown) {
        if (error !== null && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') return
        throw error
      }
      for (const entry of dirEntries) {
        const childPath = `${dir}/${entry.name}`
        const childPrefix = prefix ? `${prefix}/${entry.name}` : entry.name
        if (entry.isDirectory()) {
          await walk(childPath, childPrefix)
        } else if (entry.name.endsWith('.vec.json')) {
          const content = await readFile(childPath, 'utf-8')
          const vector = JSON.parse(content) as number[]
          const key = childPrefix.slice(0, -'.vec.json'.length)
          entries.push([key, vector])
        }
      }
    }

    await walk(this._vectorDir, '')
    return entries
  }
}
