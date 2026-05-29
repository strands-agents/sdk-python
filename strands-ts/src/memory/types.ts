import type { JSONValue } from '../types/json.js'

/**
 * A single entry retrieved from or stored to a memory store.
 */
export interface MemoryEntry {
  /** The textual content of this memory entry. */
  content: string
  /** Optional metadata (e.g., score, source, id, timestamp). */
  metadata?: Record<string, JSONValue>
}

/**
 * Options passed to {@link MemoryStore.search}.
 * Store implementations may extend this with additional fields in their own signatures.
 */
export interface SearchOptions {
  /** Maximum number of results to return. */
  limit?: number
}

/**
 * Interface for a memory store backend.
 *
 * Only `search` is required. Stores that support mutation may additionally implement `add`.
 */
export interface MemoryStore {
  /** Identifier for this store, used to target specific stores in search/store tools. */
  readonly name: string
  /** Human-readable description of what this store contains. Included in tool descriptions. */
  readonly description?: string
  /** Default max results per query for this store. Defaults to 3. */
  readonly limit?: number
  /** Search the store for entries matching the query, ordered by relevance. */
  search(query: string, options?: SearchOptions): Promise<MemoryEntry[]>
  /** Add content to the store. Optional — only present on mutable stores. */
  add?(content: string, metadata?: Record<string, JSONValue>): Promise<void>
}

/**
 * Options for {@link MemoryManager.search}.
 */
export interface MemorySearchOptions {
  /** Maximum number of results per store. */
  limit?: number
  /** Filter to specific stores by name. Omit to search all. */
  stores?: string[]
}

/**
 * Options for {@link MemoryManager.store}.
 */
export interface MemoryStoreOptions {
  /** Metadata to associate with the stored entry. */
  metadata?: Record<string, JSONValue>
  /** Filter to specific writable stores by name. Omit to write to all. */
  stores?: string[]
}

/**
 * Configuration for customizing a memory tool's name, description, or store scoping.
 */
export interface MemoryToolConfig {
  /** Custom tool name. */
  name?: string
  /** Custom tool description. */
  description?: string
  /** Scopes which stores this tool targets by name. Defaults to all applicable stores. */
  stores?: string[]
}

/**
 * Configuration for the {@link MemoryManager}.
 */
export interface MemoryManagerConfig {
  /** One or more memory stores to manage. */
  stores: MemoryStore[]
  /** Search tool configuration. Defaults to `true` (auto-created targeting all stores). */
  searchToolConfig?: MemoryToolConfig | boolean
  /** Store tool configuration. Defaults to `false` (opt-in). */
  storeToolConfig?: MemoryToolConfig | boolean
}
