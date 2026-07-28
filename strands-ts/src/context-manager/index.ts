/**
 * Context management for Strands agents.
 *
 * Provides the ContextManager class — a first-class agent component that manages
 * strategy-driven context reduction.
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
 *       Offload("toolResultErrors"),
 *     ],
 *   }),
 * })
 * ```
 */

export { ContextManager } from './context-manager.js'
export { Offload } from './strategies/offload.js'
export type {
  ContextManagerConfig,
  ContextStrategy,
  StrategyContext,
  StrategyInitContext,
} from './types.js'
export type { OffloadTarget, WhenConditions, StrategyBuilder, OffloadNamespace } from './strategies/offload.js'
export type { TruncateConfig } from './strategies/methods/truncate.js'
export type { SummarizeConfig } from './strategies/methods/summarize.js'
