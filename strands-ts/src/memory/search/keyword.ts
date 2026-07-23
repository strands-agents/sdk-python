/**
 * Shared keyword-search utilities for memory stores that use lexical token-overlap scoring.
 */

/** Default maximum number of results returned by keyword-based memory store searches. */
export const DEFAULT_MAX_SEARCH_RESULTS = 10

/** Split text into a set of lowercased word tokens for keyword matching. */
export function tokenize(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .split(/[^\p{L}\p{N}_]+/u)
      .filter(Boolean)
  )
}

/**
 * Lexical relevance score: the number of distinct query tokens that appear in the content.
 * Returns 0 when there is no overlap.
 */
export function tokenOverlapScore(queryTokens: Set<string>, content: string): number {
  let score = 0
  for (const token of tokenize(content)) {
    if (queryTokens.has(token)) score++
  }
  return score
}
