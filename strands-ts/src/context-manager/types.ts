/**
 * Configuration types for the ContextManager.
 */

import type { LocalAgent } from '../types/agent.js'
import type { Message } from '../types/messages.js'

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
  init?(agent: LocalAgent): void

  /**
   * Attempt to reduce context. Returns true if it made changes, false if it
   * decided not to act (e.g., conditions not met, nothing to offload).
   */
  apply(context: ContextState): Promise<boolean>
}

/**
 * State passed to strategies during apply().
 */
export interface ContextState {
  /** The agent's current message array (the context window). Strategies mutate this in place. */
  messages: Message[]

  /** The agent instance. */
  agent: LocalAgent

  /** Current context utilization ratio (0-1+). Above 1.0 means overflow. */
  utilization: number
}

/**
 * Full configuration for a ContextManager instance.
 */
export interface ContextManagerConfig {
  /**
   * Strategies for context reduction. Applied as an ordered pipeline: each strategy
   * sees the output of the previous. Order determines priority — if two strategies
   * target the same content, the first one to shrink it below the next strategy's
   * threshold wins. When omitted, uses the default pipeline.
   */
  strategies?: ContextStrategy[]
}
