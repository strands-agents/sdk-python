/**
 * Handoff plugin for Strands Agents — enforces handoff semantics for
 * agent-as-tool routing scenarios where a sub-agent's response should
 * reach the user verbatim without post-processing.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 *
 * const specialist = new Agent({ name: 'Specialist' })
 * const orchestrator = new Agent({
 *   tools: [specialist.asTool({ handoff: true })],
 *   // HandoffPlugin is auto-registered — no manual setup needed
 * })
 *
 * const result = await orchestrator.invoke('Help me with billing')
 * console.log(result.stopReason) // 'handoff'
 * ```
 */

export { HandoffPlugin } from './plugin.js'
