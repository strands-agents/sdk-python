/**
 * Configuration types for the ContextManager.
 */

/**
 * Configuration for the L1 stash (durable message store).
 */
export interface StashConfig {
  /**
   * Whether to write messages to L1 on arrival. Defaults to true.
   * When false, offload operations are destructive (originals are lost).
   */
  enabled?: boolean
}

/**
 * Full configuration for a ContextManager instance.
 */
export interface ContextManagerConfig {
  /**
   * Storage backend for L1 writes. Falls back to InMemoryStorage if not provided.
   */
  storage?: import('../storage/storage.js').Storage

  /**
   * L1 stash configuration. Set to `false` to disable writes entirely,
   * or pass a StashConfig object for fine-grained control.
   */
  stash?: StashConfig | boolean

  /**
   * Strategy definitions (reserved for future PRs).
   */
  strategies?: unknown[]
}
