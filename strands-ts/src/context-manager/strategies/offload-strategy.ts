/**
 * Legacy re-export for backwards compatibility during transition.
 * Use `Offload.truncate()` builder instead.
 *
 * @deprecated Use `Offload.truncate(target, config)` from `./offload.js`
 * @internal
 */

export { TruncateMethod as OffloadStrategy } from './methods/truncate-method.js'
export type { TruncateMethodConfig as OffloadStrategyConfig } from './methods/truncate-method.js'
