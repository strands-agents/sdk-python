import type { Storage, StorageSearchResult } from '../storage.js'

/**
 * A pluggable search strategy for storage backends.
 *
 * Strategies encapsulate a single approach to searching stored content —
 * keyword/lexical scan, vector similarity, full-text index, etc.
 * Storage backends delegate their `search()` to a strategy, and consumers
 * (memory stores, context offloaders) can override the default.
 *
 * The `SearchQuery` type parameter controls what the strategy accepts. It defaults
 * to `string` (a natural-language query). Strategies that support richer queries
 * (e.g. a pre-computed embedding vector with metadata filters) can widen this type.
 */
export interface SearchStrategy<SearchQuery = string> {
  /**
   * Searches content in `storage` matching `query`.
   *
   * @param storage - The storage to search over
   * @param query - A string query or strategy-specific query object
   * @returns Matched keys with relevance scores, ranked best-first
   */
  search(storage: Storage, query: SearchQuery): Promise<StorageSearchResult[]>
}
