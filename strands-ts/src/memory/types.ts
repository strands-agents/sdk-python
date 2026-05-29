import type { JSONValue } from '../types/json.js'
import type { Tool } from '../tools/tool.js'

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
 * Common configuration shared by all built-in memory stores.
 *
 * Store implementations should extend this so identity, result limits, and writability are
 * consistent across every store. Concrete stores add their own backend-specific fields.
 */
export interface MemoryStoreConfig {
  /** Identifier for this store, used to target specific stores in search/store tools. */
  name: string
  /** Human-readable description of what this store contains. Included in tool descriptions. */
  description?: string
  /**
   * Default maximum number of results this store returns per search, used when a caller does not
   * pass a per-call `limit`. Defaults to 3.
   */
  maxSearchResults?: number
  /**
   * Whether the caller wants this store instance to accept writes. This is per-instance intent,
   * not a capability of the store type: a backend may support writing, yet you can pin a given
   * instance to read-only by leaving this unset.
   *
   * @defaultValue false
   */
  writable?: boolean
}

/**
 * Interface for a memory store backend.
 *
 * Only `search` is required. Stores the caller wants to write to additionally implement `add` and
 * report `writable: true`.
 */
export interface MemoryStore {
  /** Identifier for this store, used to target specific stores in search/store tools. */
  readonly name: string
  /** Human-readable description of what this store contains. Included in tool descriptions. */
  readonly description?: string
  /** Default maximum number of results this store returns per search. Defaults to 3. */
  readonly maxSearchResults?: number
  /**
   * Whether this instance accepts writes, reflecting the caller's per-instance intent. A store the
   * caller made writable exposes `add` and reports `writable: true`; a read-only instance omits
   * `add`. When omitted, writability is inferred from the presence of `add` (backwards-compatible
   * default for ad-hoc stores).
   */
  readonly writable?: boolean
  /** Search the store for entries matching the query, ordered by relevance. */
  search(query: string, options?: SearchOptions): Promise<MemoryEntry[]>
  /** Add content to the store. Optional — only present on stores the caller made writable. */
  add?(content: string, metadata?: Record<string, JSONValue>): Promise<void>
  /**
   * Returns store-specific tools to register with the agent. Optional — implement to expose
   * backend-specific capabilities (e.g. management or query tools) beyond the manager's
   * `search`/`store` tools.
   */
  getTools?(): Tool[]
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
