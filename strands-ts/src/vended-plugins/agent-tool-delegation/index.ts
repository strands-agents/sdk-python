/**
 * Agent tool-delegation plugin for Strands Agents — enforces delegation semantics
 * for tool routing scenarios where a tool's response should reach the caller verbatim
 * without additional model processing.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 *
 * const specialist = new Agent({ name: 'Specialist' })
 * const orchestrator = new Agent({
 *   tools: [specialist.asTool({ delegate: true })],
 *   // AgentToolDelegation is auto-registered — no manual setup needed
 * })
 *
 * const result = await orchestrator.invoke('Help me with billing')
 * console.log(result.stopReason) // 'delegated'
 * ```
 */
export { AgentToolDelegation } from './plugin.js'
