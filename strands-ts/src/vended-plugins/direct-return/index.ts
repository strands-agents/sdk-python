/**
 * Direct-return plugin for Strands Agents — enforces direct-return semantics for
 * tool routing scenarios where a tool's response should reach the user verbatim
 * without post-processing.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 *
 * const specialist = new Agent({ name: 'Specialist' })
 * const orchestrator = new Agent({
 *   tools: [specialist.asTool({ handoff: true })],
 *   // DirectReturnPlugin is auto-registered — no manual setup needed
 * })
 *
 * const result = await orchestrator.invoke('Help me with billing')
 * console.log(result.stopReason) // 'handoff'
 * ```
 */

export { DirectReturnPlugin } from './plugin.js'
