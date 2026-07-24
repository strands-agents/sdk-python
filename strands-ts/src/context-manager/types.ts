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
 * A context management strategy that can reduce context when triggered.
 *
 * Strategies are applied in order during `apply()`. Each decides whether
 * to act based on the current context state (utilization, message count, etc.).
 */
export interface ContextStrategy {
  /** Stable identifier for logging and observability. */
  readonly name: string

  /**
   * Attempt to reduce context. Returns true if it made changes, false if it
   * decided not to act (e.g., conditions not met, nothing to offload).
   */
  apply(context: StrategyContext): Promise<boolean>
}

/**
 * State passed to strategies during apply().
 */
export interface StrategyContext {
  /** The agent's current message array (L0). Strategies mutate this in place. */
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
   * Strategy pipeline for context reduction. Applied in order during `apply()`.
   * When omitted, uses the default pipeline: offload tool results → summarize oldest.
   */
  strategies?: ContextStrategy[]
}
