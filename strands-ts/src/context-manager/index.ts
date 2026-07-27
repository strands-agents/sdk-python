/**
 * Context management for Strands agents.
 *
 * Provides the ContextManager class — a first-class agent component that manages
 * the L1 stash and pass-driven context reduction.
 *
 * @example
 * ```typescript
 * import { Agent, ContextManager } from '@strands-agents/sdk'
 *
 * const agent = new Agent({
 *   model,
 *   contextManager: new ContextManager(),
 * })
 * ```
 */

export { ContextManager } from './context-manager.js'
export { OffloadPass } from './strategies/offload-strategy.js'
export { SummarizePass } from './strategies/summarize-strategy.js'
export type { ContextManagerConfig, ContextPass, MessageCategory, PassContext, PassInitContext, StashConfig } from './types.js'
export type { OffloadPassConfig } from './strategies/offload-strategy.js'
export type { SummarizePassConfig } from './strategies/summarize-strategy.js'
