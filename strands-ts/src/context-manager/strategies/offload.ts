/**
 * Builder API for offload strategies.
 *
 * Offload strategies move content out of L0 (the context window) and into storage.
 * The builder composes a target, a reduction method (truncate or summarize), and
 * optional conditions into a strategy that implements `ContextStrategy`.
 *
 * @example
 * ```typescript
 * import { ContextManager, Offload } from '@strands-agents/sdk'
 *
 * const cm = new ContextManager({
 *   strategies: [
 *     Offload.truncate("toolResults", { previewTokens: 750 })
 *       .when({ threshold: 1500, skipRecent: 3 }),
 *     Offload.truncate(["bash", "list_files"], { previewTokens: 200 })
 *       .when({ threshold: 500 }),
 *     Offload.summarize({ ratio: 0.3 })
 *       .when({ utilization: 0.85 }),
 *   ],
 * })
 * ```
 */

import type { ContextStrategy } from '../types.js'
import { TruncateMethod, type TruncateMethodConfig } from './methods/truncate-method.js'
import { SummarizeMethod, type SummarizeMethodConfig } from './methods/summarize-method.js'

/**
 * Target for offload operations.
 *
 * - `"toolResults"` — all successful tool results
 * - `"toolResultErrors"` — all error tool results
 * - `string[]` — specific tool names to include; prefix with `!` to exclude
 */
export type OffloadTarget = 'toolResults' | 'toolResultErrors' | string[]

/**
 * Conditions that determine when an offload strategy fires.
 */
export interface OffloadWhenConditions {
  /** Token threshold above which individual results are offloaded. */
  threshold?: number

  /** Context utilization ratio (0-1) above which the strategy fires. */
  utilization?: number

  /** Number of recent messages to skip during retroactive apply(). */
  skipRecent?: number
}

/**
 * Intermediate builder result that allows chaining `.when()` conditions.
 * Also implements `ContextStrategy` directly so it can be used without `.when()`.
 */
export interface OffloadStrategyBuilder extends ContextStrategy {
  /** Add conditions that determine when this strategy fires. */
  when(conditions: OffloadWhenConditions): ContextStrategy
}

function createTruncateBuilder(
  target: OffloadTarget,
  config?: TruncateMethodConfig
): OffloadStrategyBuilder {
  const method = new TruncateMethod(target, config)
  return {
    get name(): string {
      return method.name
    },
    init: method.init.bind(method),
    apply: method.apply.bind(method),
    when(conditions: OffloadWhenConditions): ContextStrategy {
      return new TruncateMethod(target, config, conditions)
    },
  }
}

function createSummarizeBuilder(config?: SummarizeMethodConfig): OffloadStrategyBuilder {
  const method = new SummarizeMethod(config)
  return {
    get name(): string {
      return method.name
    },
    apply: method.apply.bind(method),
    when(conditions: OffloadWhenConditions): ContextStrategy {
      return new SummarizeMethod(config, conditions)
    },
  }
}

/**
 * Offload strategy builder.
 *
 * Use as a namespace with static methods for constructing offload strategies:
 * - `Offload.truncate(target, config)` — truncate oversized content with a preview
 * - `Offload.summarize(config)` — summarize oldest messages
 *
 * Or call directly as a function for simple target-only offloading:
 * - `Offload("toolResults")` — offload all tool results with default settings
 */
export interface OffloadNamespace {
  /** Shorthand: offload with default truncate method for the given target. */
  (target: OffloadTarget): OffloadStrategyBuilder

  /** Create a strategy that truncates oversized content into a head-tail preview. */
  truncate(target: OffloadTarget, config?: TruncateMethodConfig): OffloadStrategyBuilder

  /** Create a strategy that summarizes the oldest messages to free token space. */
  summarize(config?: SummarizeMethodConfig): OffloadStrategyBuilder
}

function offloadFn(target: OffloadTarget): OffloadStrategyBuilder {
  return createTruncateBuilder(target)
}

offloadFn.truncate = function truncate(target: OffloadTarget, config?: TruncateMethodConfig): OffloadStrategyBuilder {
  return createTruncateBuilder(target, config)
}

offloadFn.summarize = function summarize(config?: SummarizeMethodConfig): OffloadStrategyBuilder {
  return createSummarizeBuilder(config)
}

/**
 * Builder for offload strategies.
 *
 * @example
 * ```typescript
 * Offload.truncate("toolResults", { previewTokens: 750 }).when({ threshold: 1500 })
 * Offload.summarize({ ratio: 0.3 }).when({ utilization: 0.85 })
 * Offload("toolResultErrors")
 * ```
 */
export const Offload: OffloadNamespace = offloadFn as OffloadNamespace
