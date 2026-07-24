/**
 * Context management for Strands agents.
 *
 * Provides the ContextManager class — a first-class agent component that manages
 * the L1 durable transcript and (in later PRs) strategy-driven context offloading.
 *
 * @example
 * ```typescript
 * import { Agent } from 'strands-agents'
 * import { ContextManager } from 'strands-agents/context-manager'
 * import { LocalFileStorage } from 'strands-agents/storage'
 *
 * const agent = new Agent({
 *   model,
 *   plugins: [new ContextManager({ storage: new LocalFileStorage('./.strands/') })],
 * })
 * ```
 */

export { ContextManager } from './context-manager.js'
export type { ContextManagerConfig, StashConfig } from './types.js'
