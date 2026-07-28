/**
 * Legacy re-export for backwards compatibility during transition.
 * Use `Offload.summarize()` builder instead.
 *
 * @deprecated Use `Offload.summarize(config)` from `./offload.js`
 * @internal
 */

export { SummarizeMethod as SummarizeStrategy } from './methods/summarize-method.js'
export type { SummarizeMethodConfig as SummarizeStrategyConfig } from './methods/summarize-method.js'
