/**
 * Context management for Strands agents.
 *
 * Provides the ContextManager class — a first-class agent component that manages
 * the L1 stash and strategy-driven context reduction.
 *
 * @example
 * ```typescript
 * import { Agent, ContextManager, Offload } from '@strands-agents/sdk'
 *
 * const agent = new Agent({
 *   model,
 *   contextManager: new ContextManager({
 *     strategies: [
 *       Offload.truncate("toolResults", { previewTokens: 750 })
 *         .when({ threshold: 1500, skipRecent: 3 }),
 *       Offload.summarize({ ratio: 0.3 })
 *         .when({ utilization: 0.85 }),
 *     ],
 *   }),
 * })
 * ```
 */

export { ContextManager } from './context-manager.js'
export { Offload } from './strategies/offload.js'
export { TruncateMethod } from './strategies/methods/truncate-method.js'
export { SummarizeMethod } from './strategies/methods/summarize-method.js'
export type {
  ContextManagerConfig,
  ContextStrategy,
  MessageCategory,
  StashConfig,
  StrategyContext,
  StrategyInitContext,
} from './types.js'
export type { TruncateMethodConfig } from './strategies/methods/truncate-method.js'
export type { SummarizeMethodConfig } from './strategies/methods/summarize-method.js'
export type { OffloadTarget, OffloadWhenConditions, OffloadStrategyBuilder, OffloadNamespace } from './strategies/offload.js'

// Legacy re-exports
export { OffloadStrategy } from './strategies/offload-strategy.js'
export { SummarizeStrategy } from './strategies/summarize-strategy.js'
export type { OffloadStrategyConfig } from './strategies/offload-strategy.js'
export type { SummarizeStrategyConfig } from './strategies/summarize-strategy.js'
