/**
 * Strategy-based model routing.
 *
 * `ModelRouter` asks its strategy before the first model call and after unclaimed failures. The API is
 * provisional and may change before it is finalized.
 */
export { FallbackStrategy } from './fallback-strategy.js'
export { ModelRouter, RoutingCandidate } from './router.js'
export type { ModelRouterOptions, RoutingCandidateOptions } from './router.js'
export type { CandidateInput, RoutingAttempt, RoutingContext, RoutingStrategy } from './strategy.js'
