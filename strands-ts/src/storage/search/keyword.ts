import type { Storage, StorageSearchResult } from '../storage.js'
import type { SearchStrategy } from './types.js'

/** Split text into a set of lowercased word tokens for keyword matching. */
export function tokenize(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .split(/[^\p{L}\p{N}_]+/u)
      .filter(Boolean)
  )
}

/** Lexical relevance score: distinct query tokens that appear in the content. Returns 0 when there is no overlap. */
export function tokenOverlapScore(queryTokens: Set<string>, content: string): number {
  let score = 0
  for (const token of tokenize(content)) {
    if (queryTokens.has(token)) score++
  }
  return score
}

/**
 * Keyword search strategy using token-overlap scoring.
 *
 * Tokenizes the query and each stored entry (key + content), then scores by the
 * number of distinct query tokens that appear. Works on any storage backend with
 * `list()` and `read()` — no index or embedding model required.
 *
 * This is the default search strategy for all shipped storage backends.
 *
 * @example
 * ```typescript
 * import { KeywordSearchStrategy } from '@strands-agents/sdk/storage/search'
 *
 * const results = await KeywordSearchStrategy.search(storage, 'dark mode toggle')
 * ```
 */
export const KeywordSearchStrategy: SearchStrategy = {
  async search(storage: Storage, query: string): Promise<StorageSearchResult[]> {
    const queryTokens = tokenize(query)
    if (queryTokens.size === 0) return []

    const allKeys = await storage.list('')
    const scored: StorageSearchResult[] = []
    for (const key of allKeys) {
      const bytes = await storage.read(key)
      if (!bytes) continue
      const content = new TextDecoder().decode(bytes)
      const score = tokenOverlapScore(queryTokens, `${key} ${content}`)
      if (score > 0) scored.push({ key, score })
    }

    scored.sort((a, b) => b.score - a.score)
    return scored
  },
}
