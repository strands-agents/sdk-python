/**
 * Configuration types for the ContextManager.
 */

import type { Storage } from '../storage/storage.js'
import type { Stash } from './stash.js'

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
  init?(agent: import('../types/agent.js').LocalAgent, stash?: Stash): void

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
  /** The agent's current message array (L0). Strategies mutate this in place. */
  messages: import('../types/messages.js').Message[]

  /** The agent instance. */
  agent: import('../types/agent.js').LocalAgent

  /** Current context utilization ratio (0-1+). Above 1.0 means overflow. */
  utilization: number

  /** L1 stash for persisting offloaded content. Present when storage is configured. */
  stash?: Stash
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

  /**
   * Storage backend for persisting offloaded content (L1 stash).
   *
   * When provided, offload strategies persist the original content before replacing
   * it in L0. The agent can retrieve stashed content on demand via the
   * `retrieve_context` tool (registered automatically when storage is set).
   */
  storage?: Storage
}
