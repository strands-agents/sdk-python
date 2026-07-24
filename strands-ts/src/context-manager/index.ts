/**
 * Context management for Strands agents.
 *
 * Provides the ContextManager class — a first-class agent component that manages
 * the L1 stash and strategy-driven context reduction.
 *
 * @example
 * ```typescript
 * import { Agent, ContextManager } from '@strands-agents/sdk'
 *
 * const agent = new Agent({
 *   model,
 *   contextManager: new ContextManager({ storage }),
 * })
 * ```
 */

export { ContextManager } from './context-manager.js'
export { OffloadStrategy } from './strategies/offload-strategy.js'
export { SummarizeStrategy } from './strategies/summarize-strategy.js'
export type { ContextManagerConfig, ContextStrategy, StashConfig, StrategyContext } from './types.js'
export type { OffloadStrategyConfig } from './strategies/offload-strategy.js'
export type { SummarizeStrategyConfig } from './strategies/summarize-strategy.js'
