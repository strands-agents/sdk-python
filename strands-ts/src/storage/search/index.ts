/**
 * Pluggable search strategies for storage backends.
 *
 * Each strategy encapsulates a single approach to searching stored content.
 * Storage backends use {@link KeywordSearchStrategy} by default; consumers
 * (memory stores, context offloaders) can override with a different strategy.
 *
 * @packageDocumentation
 */

export type { SearchStrategy, Embedder, EmbeddingsConfig } from './types.js'
export type { StorageSearchResult } from '../storage.js'
export { KeywordSearchStrategy } from './keyword.js'
export { InMemoryVectorSearchStrategy } from './in-memory-vector.js'
export type { InMemoryVectorSearchStrategyConfig } from './in-memory-vector.js'
export { LocalVectorSearchStrategy } from './local-vector.js'
export type { LocalVectorSearchStrategyConfig } from './local-vector.js'
export { S3VectorSearchStrategy } from './s3-vector.js'
export type { S3VectorSearchStrategyConfig } from './s3-vector.js'
