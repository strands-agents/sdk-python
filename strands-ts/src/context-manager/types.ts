/**
 * Configuration types for the ContextManager.
 */

/**
 * Semantic categories for message content.
 *
 * A message matches a category if it contains at least one block of that type.
 * `'media'` is a convenience shorthand that matches any of `'image'`, `'video'`, or `'document'`.
 * `'user'` and `'assistant'` are role-level shorthands that match by message role.
 */
export type MessageCategory =
  | 'text'
  | 'toolUse'
  | 'toolResult'
  | 'toolError'
  | 'reasoning'
  | 'image'
  | 'video'
  | 'document'
  | 'citations'
  | 'json'
  | 'cachePoint'
  | 'guardContent'
  | 'media'
  | 'user'
  | 'assistant'

/**
 * Configuration for the L1 stash (durable message store).
 */
export interface StashConfig {
  /**
   * Whether to write messages to L1 on arrival. Defaults to true.
   * When false, offload operations are destructive (originals are lost).
   */
  enabled?: boolean

  /**
   * Only stash messages matching these categories. Mutually exclusive with `exclude`.
   * A message is written if it contains at least one block matching any listed category.
   */
  include?: MessageCategory | MessageCategory[]

  /**
   * Skip stashing messages matching these categories. Mutually exclusive with `include`.
   * A message is skipped if it contains at least one block matching any listed category.
   */
  exclude?: MessageCategory | MessageCategory[]
}

/**
 * A context pass that can reduce or transform context when triggered.
 *
 * Passes are applied in order during `apply()`. Each decides whether
 * to act based on the current context state (utilization, message count, etc.).
 */
export interface ContextPass {
  /** Stable identifier for logging and observability. */
  readonly name: string

  /**
   * Called once when the ContextManager is attached to an agent.
   * Passes can use this to register hooks (e.g., eager offloading on message arrival).
   */
  init?(context: PassInitContext): void

  /**
   * Attempt to reduce context. Returns true if it made changes, false if it
   * decided not to act (e.g., conditions not met, nothing to offload).
   */
  apply(context: PassContext): Promise<boolean>
}

/**
 * Context passed to context passes during initialization.
 */
export interface PassInitContext {
  /** The agent instance. */
  agent: import('../types/agent.js').LocalAgent

  /** Storage backend for offloading content. */
  storage: import('../storage/storage.js').Storage
}

/**
 * State passed to context passes during apply().
 */
export interface PassContext {
  /** The agent's current message array (L0). Passes mutate this in place. */
  messages: import('../types/messages.js').Message[]

  /** The agent instance. */
  agent: import('../types/agent.js').LocalAgent

  /** Current context utilization ratio (0-1+). Above 1.0 means overflow. */
  utilization: number

  /** Storage backend for offloading content. */
  storage: import('../storage/storage.js').Storage
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
   * Context pass pipeline for context reduction. Applied in order during `apply()`.
   * When omitted, uses the default pipeline: offload tool results → summarize oldest.
   */
  passes?: ContextPass[]
}
