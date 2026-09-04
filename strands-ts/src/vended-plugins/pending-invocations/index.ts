/**
 * Pending-invocations visibility plugin for Strands Agents.
 *
 * Provides the {@link PendingInvocations} plugin, which renders the agent's invocation
 * queue into the model input of the running invocation — ephemerally, never into
 * durable history — so the model can wrap up early when a queued request supersedes
 * its current work.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { PendingInvocations } from '@strands-agents/sdk/vended-plugins/pending-invocations'
 *
 * // Registered automatically when concurrentInvocationMode is 'enqueue' or
 * // 'cancelPrevious'. Attach manually on a 'throw'-mode agent whose callers
 * // use per-call `ifBusy`:
 * const agent = new Agent({
 *   model,
 *   plugins: [new PendingInvocations({ name: 'my-queue-view' })],
 * })
 * ```
 */

export { PendingInvocations } from './plugin.js'
export type { PendingInvocationsConfig } from './plugin.js'
