import type { Storage, StorageSearchResult } from '../storage.js'
import type { SearchStrategy } from './types.js'

import { tokenize, tokenOverlapScore } from '../storage.js'

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
 * import { KeywordSearchStrategy } from '@strands-agents/sdk/storage'
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
