/**
 * Context-injection plugin for Strands Agents.
 *
 * Provides the {@link ContextInjector} plugin, which folds just-in-time text into the model input
 * before each call without touching durable history. `escapeXml` is offered for providers that render
 * XML and want to escape untrusted content themselves — it is never applied automatically.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { ContextInjector, escapeXml } from '@strands-agents/sdk/vended-plugins/context-injector'
 *
 * const agent = new Agent({
 *   model,
 *   plugins: [
 *     new ContextInjector({
 *       renderContent: async ({ messages }) => `<context>${escapeXml(derive(messages))}</context>`,
 *     }),
 *   ],
 * })
 * ```
 */

export { ContextInjector } from './plugin.js'
export type { ContextInjectorConfig } from './plugin.js'
export type { InjectionTrigger, InjectionContext } from '../../injection/types.js'
export { escapeXml } from '../../injection/xml.js'
