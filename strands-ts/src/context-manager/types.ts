/**
 * Configuration types for the ContextManager.
 */

/**
 * A context reduction strategy that can offload, summarize, or otherwise
 * transform the message array to reduce token usage.
 *
 * Strategies are applied in order during `apply()`. Each decides whether
 * to act based on the current context state (utilization, message count, etc.).
 */
export interface ContextStrategy {
  /** Stable identifier for logging and observability. */
  readonly name: string

  /**
   * Called once when the ContextManager is attached to an agent.
   * Strategies can use this to register hooks (e.g., eager offloading on message arrival).
   */
  init?(agent: import('../types/agent.js').LocalAgent): void

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
}

/**
 * Full configuration for a ContextManager instance.
 */
export interface ContextManagerConfig {
  /**
   * Strategies for context reduction. Applied in order during `apply()`.
   * When omitted, uses the default pipeline: offload tool results → summarize oldest.
   */
  strategies?: ContextStrategy[]
}
