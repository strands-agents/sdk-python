import type { Storage, StorageSearchResult } from '../storage.js'

/**
 * A pluggable search strategy for storage backends.
 *
 * Strategies encapsulate a single approach to searching stored content —
 * keyword/lexical scan, vector similarity, full-text index, etc.
 * Storage backends delegate their `search()` to a strategy, and consumers
 * (memory stores, context offloaders) can override the default.
 */
export interface SearchStrategy {
  /**
   * Searches content in `storage` matching `query`.
   *
   * @param storage - The storage to search over
   * @param query - Natural-language search query
   * @returns Matched keys with relevance scores, ranked best-first
   */
  search(storage: Storage, query: string): Promise<StorageSearchResult[]>
}
