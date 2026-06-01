import type { JSONValue } from '../types/json.js'
import type { Tool } from '../tools/tool.js'
import type { MessageData } from '../types/messages.js'

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
  maxSearchResults?: number
}

/**
 * Common configuration shared by all built-in memory stores.
 *
 * Store implementations should extend this so identity, result limits, and writability are
 * consistent across every store. Concrete stores add their own backend-specific fields.
 */
export interface MemoryStoreConfig {
  /** Identifier for this store, used to target specific stores in search/add tools. */
  name: string
  /** Human-readable description of what this store contains. Included in tool descriptions. */
  description?: string
  /**
   * Default maximum number of results this store returns per search, used when a caller does not
   * pass a per-call `maxSearchResults`. Defaults to 3.
   */
  maxSearchResults?: number
  /**
   * Whether this store instance accepts writes. Concrete stores resolve this to a definite
   * `writable` boolean on the {@link MemoryStore} (defaulting to `false` when omitted).
   *
   * @defaultValue false
   */
  writable?: boolean
}

/**
 * Interface for a memory store backend.
 *
 * Every store is searchable. The `writable` flag declares whether the store also accepts writes,
 * which is how the {@link MemoryManager} decides where to route them: `search_memory` queries all
 * stores, while `add_memory` only writes to `writable` stores.
 */
export interface MemoryStore {
  /** Identifier for this store, used to target specific stores in search/add tools. */
  readonly name: string
  /**
   * Whether this store accepts writes.
   * - `false`: searchable only; never written to.
   * - `true`: searchable and writable. Requires `add` to be implemented.
   */
  readonly writable: boolean
  /** Human-readable description of what this store contains. Included in tool descriptions. */
  readonly description?: string
  /** Default max results per query for this store. Defaults to 3. */
  readonly maxSearchResults?: number
  /** Search the store for entries matching the query, ordered by relevance. */
  search(query: string, options?: SearchOptions): Promise<MemoryEntry[]>
  /**
   * Add content to the store. Required when `writable` is `true`; ignored otherwise.
   * A store may implement `add` while declaring `writable: false`, in which case it is never invoked.
   */
  add?(content: string, metadata?: Record<string, JSONValue>): Promise<void>
  /**
   * Returns store-specific tools to register with the agent, alongside the manager's own
   * `search_memory` / `add_memory` tools. Implement to expose backend-specific capabilities
   * (e.g. a store-native query tool). Optional — mirrors {@link Plugin.getTools}.
   *
   * @returns Array of tools provided by this store
   */
  getTools?(): Tool[]
}

/**
 * Options for {@link MemoryManager.search}.
 */
export interface SearchMemoryOptions {
  /** Maximum number of results per store. */
  maxSearchResults?: number
  /** Filter to specific stores by name. Omit to search all. */
  stores?: string[]
}

/**
 * Options for {@link MemoryManager.add}.
 */
export interface AddMemoryOptions {
  /** Metadata to associate with the added entry. */
  metadata?: Record<string, JSONValue>
  /** Filter to specific writable stores by name. Omit to write to all. */
  stores?: string[]
}

/**
 * Configuration for customizing a memory tool's name or description.
 *
 * Store targeting is derived from each store's `writable` flag (see {@link MemoryStore}), not
 * configured here: `search_memory` targets all stores, `add_memory` targets `writable` stores.
 */
export interface MemoryToolConfig {
  /** Custom tool name. */
  name?: string
  /** Custom tool description. */
  description?: string
}

/**
 * Configuration for passive context injection.
 *
 * When enabled, the {@link MemoryManager} searches memory before each model call and injects the
 * top results as a `user` message placed immediately before the user's latest message, so relevant
 * knowledge is always present without the model choosing to search. Injection only runs when the
 * latest message is a user ask (not a tool result), keeping the user's ask the final message the
 * model sees.
 */
export interface InjectionConfig {
  /** Maximum number of entries to retrieve and inject per model call. Defaults to 1. */
  maxResults?: number
  /**
   * Derives the search query from the current conversation. Return `undefined` or an empty string
   * to skip injection for this call. Defaults to the text of the most recent assistant message.
   */
  query?: (messages: MessageData[]) => string | undefined
  /** Renders retrieved entries into the injected message text. Defaults to an XML block. */
  format?: (entries: MemoryEntry[]) => string
}

/**
 * Configuration for the {@link MemoryManager}.
 */
export interface MemoryManagerConfig {
  /** One or more memory stores to manage. */
  stores: MemoryStore[]
  /** Search tool configuration. Defaults to `true` (auto-created targeting all stores). */
  searchToolConfig?: MemoryToolConfig | boolean
  /** Add tool configuration. Defaults to `false` (opt-in). */
  addToolConfig?: MemoryToolConfig | boolean
  /** Passive context injection. Defaults to `false` (opt-in). `true` uses default injection settings. */
  injection?: boolean | InjectionConfig
}
