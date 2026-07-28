/**
 * Builder API for inject strategies.
 *
 * Inject strategies pull content into L0 (the context window) from storage.
 * The builder composes a source, a reduction method (truncate or summarize), and
 * optional conditions into a strategy that implements `ContextStrategy`.
 *
 * @example
 * ```typescript
 * import { ContextManager, Inject } from '@strands-agents/sdk'
 *
 * const cm = new ContextManager({
 *   strategies: [
 *     Inject.truncate("stash", { previewTokens: 500 })
 *       .when({ utilization: 0.5 }),
 *     Inject.summarize("stash")
 *       .when({ utilization: 0.7 }),
 *   ],
 * })
 * ```
 */

import { logger } from '../../logging/logger.js'
import type { Model } from '../../models/model.js'
import { TextBlock } from '../../types/messages.js'
import { Message } from '../../types/messages.js'
import type { ContextStrategy, StrategyContext } from '../types.js'
import { type TruncateConfig } from './methods/truncate.js'
import { summarizeText, type SummarizeConfig } from './methods/summarize.js'
import type { WhenConditions, StrategyBuilder } from './offload.js'

/**
 * Source for inject operations.
 *
 * - `"stash"` — inject from the L1 stash
 */
export type InjectSource = 'stash'

// --- Inject + Truncate strategy ---

class InjectTruncateStrategy implements ContextStrategy {
  readonly name = 'inject:truncate'

  private readonly _source: InjectSource
  private readonly _truncateConfig: TruncateConfig
  private readonly _utilization: number | undefined

  constructor(source: InjectSource, config?: TruncateConfig, conditions?: WhenConditions) {
    this._source = source
    this._truncateConfig = config ?? {}
    this._utilization = conditions?.utilization
  }

  async apply(context: StrategyContext): Promise<boolean> {
    if (this._utilization !== undefined && context.utilization > this._utilization) {
      logger.debug(
        `utilization=<${context.utilization}>, threshold=<${this._utilization}> | skipping inject, above threshold`
      )
      return false
    }

    const { messages, storage } = context
    const previewTokens = this._truncateConfig.previewTokens ?? 1000
    const previewChars = previewTokens * 4

    const keys = await storage.list(`${this._source}/`)
    if (keys.length === 0) return false

    let injected = false
    for (const key of keys) {
      const data = await storage.read(key)
      if (!data) continue

      const fullText = new TextDecoder().decode(data)
      let content: string
      if (fullText.length <= previewChars) {
        content = fullText
      } else {
        const headChars = Math.floor(previewChars * 0.6)
        const tailChars = previewChars - headChars
        content = `${fullText.slice(0, headChars)}\n\n[... truncated ...]\n\n${fullText.slice(-tailChars)}`
      }

      messages.push(new Message({ role: 'user', content: [new TextBlock(`[Injected from ${key}]\n${content}`)] }))
      injected = true
    }

    return injected
  }
}

// --- Inject + Summarize strategy ---

class InjectSummarizeStrategy implements ContextStrategy {
  readonly name = 'inject:summarize'

  private readonly _source: InjectSource
  private readonly _summarizeConfig: SummarizeConfig
  private readonly _utilization: number | undefined

  constructor(source: InjectSource, config?: SummarizeConfig, conditions?: WhenConditions) {
    this._source = source
    this._summarizeConfig = config ?? {}
    this._utilization = conditions?.utilization
  }

  async apply(context: StrategyContext): Promise<boolean> {
    if (this._utilization !== undefined && context.utilization > this._utilization) {
      logger.debug(
        `utilization=<${context.utilization}>, threshold=<${this._utilization}> | skipping inject summarize, above threshold`
      )
      return false
    }

    const { messages, storage, agent } = context
    const model = this._summarizeConfig.model ?? (agent as unknown as Record<string, unknown>)['model'] as Model | undefined

    if (!model) {
      logger.warn('no model available for inject summarization')
      return false
    }

    const keys = await storage.list(`${this._source}/`)
    if (keys.length === 0) return false

    const parts: string[] = []
    for (const key of keys) {
      const data = await storage.read(key)
      if (!data) continue
      parts.push(new TextDecoder().decode(data))
    }

    if (parts.length === 0) return false

    const combined = parts.join('\n\n---\n\n')
    const summary = await summarizeText(combined, model, this._summarizeConfig)
    if (!summary) return false

    messages.push(new Message({ role: 'user', content: [new TextBlock(`[Summarized from ${this._source}]\n${summary}`)] }))
    return true
  }
}

// --- Builder ---

function createInjectTruncateBuilder(source: InjectSource, config?: TruncateConfig): StrategyBuilder {
  const strategy = new InjectTruncateStrategy(source, config)
  return {
    get name(): string {
      return strategy.name
    },
    apply: strategy.apply.bind(strategy),
    when(conditions: WhenConditions): ContextStrategy {
      return new InjectTruncateStrategy(source, config, conditions)
    },
  }
}

function createInjectSummarizeBuilder(source: InjectSource, config?: SummarizeConfig): StrategyBuilder {
  const strategy = new InjectSummarizeStrategy(source, config)
  return {
    get name(): string {
      return strategy.name
    },
    apply: strategy.apply.bind(strategy),
    when(conditions: WhenConditions): ContextStrategy {
      return new InjectSummarizeStrategy(source, config, conditions)
    },
  }
}

/**
 * Inject strategy builder namespace.
 *
 * Use as a namespace with static methods for constructing inject strategies:
 * - `Inject.truncate(source, config)` — inject truncated content from storage
 * - `Inject.summarize(source, config)` — inject summarized content from storage
 *
 * Or call directly as a function for simple source-only injection:
 * - `Inject("stash")` — inject from stash with default truncate method
 */
export interface InjectNamespace {
  /** Shorthand: inject from source with default truncate method. */
  (source: InjectSource): StrategyBuilder

  /** Create a strategy that injects truncated content from storage. */
  truncate(source: InjectSource, config?: TruncateConfig): StrategyBuilder

  /** Create a strategy that injects summarized content from storage. */
  summarize(source: InjectSource, config?: SummarizeConfig): StrategyBuilder
}

function injectFn(source: InjectSource): StrategyBuilder {
  return createInjectTruncateBuilder(source)
}

injectFn.truncate = function truncate(source: InjectSource, config?: TruncateConfig): StrategyBuilder {
  return createInjectTruncateBuilder(source, config)
}

injectFn.summarize = function summarize(source: InjectSource, config?: SummarizeConfig): StrategyBuilder {
  return createInjectSummarizeBuilder(source, config)
}

/**
 * Builder for inject strategies — pulls content from storage into L0.
 *
 * @example
 * ```typescript
 * Inject.truncate("stash", { previewTokens: 500 }).when({ utilization: 0.5 })
 * Inject.summarize("stash").when({ utilization: 0.7 })
 * Inject("stash")
 * ```
 */
export const Inject: InjectNamespace = injectFn as InjectNamespace
