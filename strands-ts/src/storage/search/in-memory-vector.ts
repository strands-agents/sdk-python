import type { Storage, StorageSearchResult } from '../storage.js'
import type { Embedder, SearchStrategy } from './types.js'

import { cosineSimilarity } from './cosine.js'

/** Configuration for {@link InMemoryVectorSearchStrategy}. */
export interface InMemoryVectorSearchStrategyConfig {
  /** Function that produces embedding vectors from text. */
  embedder: Embedder
  /** Maximum number of results to return. Defaults to 10. */
  maxResults?: number
}

/**
 * In-memory vector search strategy using brute-force cosine similarity.
 *
 * Stores embedding vectors in a `Map` and computes cosine similarity at query
 * time. Suitable for small-to-medium datasets where an external vector store is
 * not needed.
 *
 * @example
 * ```typescript
 * import { InMemoryVectorSearchStrategy } from '@strands-agents/sdk/storage/search'
 * import { InMemoryStorage } from '@strands-agents/sdk/storage'
 *
 * const storage = new InMemoryStorage({
 *   embeddings: { embedder: async (text) => embed(text) },
 * })
 * ```
 */
export class InMemoryVectorSearchStrategy implements SearchStrategy {
  private readonly _config: InMemoryVectorSearchStrategyConfig
  private readonly _vectors = new Map<string, number[]>()

  constructor(config: InMemoryVectorSearchStrategyConfig) {
    this._config = config
  }

  /**
   * Embeds and stores a vector for the given key.
   *
   * @param _storage - The storage the content was written to (unused)
   * @param key - The storage key to associate the vector with
   * @param data - The raw bytes to embed
   */
  async index(_storage: Storage, key: string, data: Uint8Array): Promise<void> {
    const text = new TextDecoder().decode(data)
    const vector = await this._config.embedder(text)
    this._vectors.set(key, vector)
  }

  /**
   * Queries stored vectors for keys similar to the query using cosine similarity.
   *
   * @param _storage - The storage to search over (unused)
   * @param query - Natural-language search query
   * @returns Matched keys with similarity scores, ranked best-first
   */
  async search(_storage: Storage, query: string): Promise<StorageSearchResult[]> {
    if (this._vectors.size === 0) return []
    const queryVector = await this._config.embedder(query)
    const maxResults = this._config.maxResults ?? 10
    const scored: StorageSearchResult[] = []
    for (const [key, vector] of this._vectors) {
      const score = cosineSimilarity(queryVector, vector)
      if (score > 0) scored.push({ key, score })
    }
    scored.sort((a, b) => b.score - a.score)
    return scored.slice(0, maxResults)
  }
}
