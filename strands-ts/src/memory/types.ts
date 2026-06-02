import type { JSONValue } from '../types/json.js'
import type { Tool } from '../tools/tool.js'

/**
 * A single memory entry retrieved from or stored to a memory store.
 */
export interface MemoryEntry {
  /** The textual content of this memory entry. */
  content: string
  /**
   * Name of the store this entry came from. Populated by {@link MemoryManager.search} so callers
   * (and the model, via `search_memory`) can tell which store produced each result and refine
   * targeting. Stores need not set this themselves.
   */
  storeName?: string
  /** Optional metadata (e.g., score, source, id, timestamp). */
  metadata?: Record<string, JSONValue>
}

/**
 * Options passed to {@link MemoryStore.search}.
 *
 * Store implementations may extend this with backend-specific fields (e.g. a metadata filter or
 * search-type override) in their own `search` signature. Note that {@link MemoryManager.search}
 * only forwards the base fields here across its (potentially heterogeneous) stores — to use a
 * store's extended options, call that store's `search` directly, or set them as per-instance
 * defaults on the store.
 */
export interface SearchOptions {
  /** Maximum number of results to return from this store. */
  maxSearchResults?: number
}

/**
 * Declarative properties shared by every memory store and its config.
 *
 * This is the single source of truth for a store's identity and behavior knobs. Both the runtime
 * {@link MemoryStore} interface and concrete store configs extend it, so these fields are declared
 * once. Concrete stores add their own backend-specific config fields on top.
 */
export interface MemoryStoreConfig {
  /** Identifier for this store, used to target specific stores in search/add tools. Must be unique. */
  readonly name: string
  /** Human-readable description of what this store contains. Included in tool descriptions. */
  readonly description?: string
  /**
   * Default maximum number of results this store returns per search, used when a caller does not
   * pass a per-call `maxSearchResults`.
   */
  readonly maxSearchResults?: number
  /**
   * Whether this store accepts writes. Optional at config time (caller intent, defaults to `false`);
   * concrete stores resolve it to a definite boolean on the {@link MemoryStore} interface.
   *
   * @defaultValue false
   */
  readonly writable?: boolean
}

/**
 * Interface for a memory store backend.
 *
 * Extends {@link MemoryStoreConfig} with the runtime methods a store provides. Every store is
 * searchable; the resolved `writable` flag declares whether it also accepts writes, which is how
 * the {@link MemoryManager} decides where to route them. `search_memory` can query all stores, while
 * `add_memory` can only write to `writable` stores.
 */
export interface MemoryStore extends MemoryStoreConfig {
  /**
   * Whether this store accepts writes.
   * - `false`: searchable only; never written to.
   * - `true`: searchable and writable. Requires `add` to be implemented.
   */
  readonly writable: boolean
  /** Search the store for entries matching the query, ordered by relevance. */
  search(query: string, options?: SearchOptions): Promise<MemoryEntry[]>
  /**
   * Add content to the store. Required when `writable` is `true`; ignored otherwise.
   * A store may implement `add` while declaring `writable: false`, in which case it is never invoked.
   */
  add?(content: string, metadata?: Record<string, JSONValue>): Promise<void>
  /**
   * Returns store-specific tools to register with the agent, through a {@link MemoryManager}. Registers
   * tools alongside `search_memory` / `add_memory` tools if enabled on the {@link MemoryManager}.
   * Implement to expose backend-specific capabilities (e.g. a store-native query tool).
   *  Optional, mirrors {@link Plugin.getTools}.
   *
   * @returns Array of tools provided by this store
   */
  getTools?(): Tool[]
}

/**
 * Options for {@link MemoryManager.search}.
 *
 * Extends the store primitive {@link SearchOptions} with manager-level store routing.
 */
export interface MemorySearchOptions extends SearchOptions {
  /** Filter to specific stores by name. Omit to search all. */
  stores?: string[]
}

/**
 * Options for {@link MemoryManager.add}.
 */
export interface MemoryAddOptions {
  /** Metadata to associate with the added entry. */
  metadata?: Record<string, JSONValue>
  /** Filter to specific writable stores by name. Omit to write to all writable stores. */
  stores?: string[]
}

/**
 * Configuration for customizing a memory tool's name or description.
 */
export interface MemoryToolConfig {
  /** Custom tool name. */
  name?: string
  /** Custom tool description. */
  description?: string
}

/**
 * Configuration for the `add_memory` tool. Extends {@link MemoryToolConfig} with an explicit
 * allowlist of stores the tool may write to.
 */
export interface MemoryAddToolConfig extends MemoryToolConfig {
  /**
   * The writable stores the `add_memory` tool may write to, given as store names or the
   * {@link MemoryStore} instances themselves. Each must be a configured, `writable` store.
   *  Omit (or set `addToolConfig: true`) to allow all writable stores.
   */
  stores?: (string | MemoryStore)[]
  /**
   * Whether the tool waits for store writes before returning to the model. Defaults to `true`.
   * - `true` (default): waits for writes — the tool returns `{ stored }` on success, or surfaces a
   *   failure to the model if any store write fails.
   * - `false`: fire-and-forget — the tool returns `{ accepted }` once writes are dispatched (so a
   *   slow backend never blocks the agent loop); per-store failures are logged.
   */
  waitForWrites?: boolean
}

/**
 * Configuration for the {@link MemoryManager}.
 */
export interface MemoryManagerConfig {
  /** One or more memory stores to manage. */
  stores: MemoryStore[]
  /** Search tool configuration. Defaults to `true`. */
  searchToolConfig?: MemoryToolConfig | boolean
  /**
   * Add tool configuration. Defaults to `false` (opt-in). `true` lets the tool write to all
   * writable stores; pass a {@link MemoryAddToolConfig} with `stores` to restrict it to specific ones.
   */
  addToolConfig?: MemoryAddToolConfig | boolean
}
