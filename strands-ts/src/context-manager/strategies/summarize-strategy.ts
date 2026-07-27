/**
 * Summarize strategy: compresses the oldest messages into a summary,
 * preserving key context while freeing token space.
 *
 * @internal
 */

import { logger } from '../../logging/logger.js'
import type { Model } from '../../models/model.js'
import {
  adjustSplitPointForToolPairs,
  generateSummary,
} from '../../conversation-manager/compression/context-compression.js'
import type { ContextStrategy, StrategyContext } from '../types.js'

const DEFAULT_SUMMARY_RATIO = 0.3
const DEFAULT_PRESERVE_RECENT = 10

/**
 * Configuration for the summarize strategy.
 */
export interface SummarizeStrategyConfig {
  /** Ratio of messages to summarize (0.1 - 0.8). Defaults to 0.3. */
  summaryRatio?: number

  /** Number of recent messages to always preserve. Defaults to 10. */
  preserveRecent?: number

  /** Only fire when context utilization exceeds this ratio (0-1). Defaults to undefined (always fire). */
  utilization?: number

  /** Model to use for summarization. When omitted, uses the agent's model. */
  model?: Model

  /** Custom system prompt for the summarization model. When omitted, uses the default summarization prompt. */
  systemPrompt?: string
}

/**
 * Summarizes the oldest messages into a single summary message, preserving
 * key information while reducing token count.
 *
 * Included in the default pipeline when no custom strategies are configured.
 */
export class SummarizeStrategy implements ContextStrategy {
  readonly name = 'summarize'

  private readonly _summaryRatio: number
  private readonly _preserveRecent: number
  private readonly _utilization: number | undefined
  private readonly _model: Model | undefined
  private readonly _systemPrompt: string | undefined

  constructor(config?: SummarizeStrategyConfig) {
    this._summaryRatio = Math.max(0.1, Math.min(0.8, config?.summaryRatio ?? DEFAULT_SUMMARY_RATIO))
    this._preserveRecent = config?.preserveRecent ?? DEFAULT_PRESERVE_RECENT
    this._utilization = config?.utilization
    this._model = config?.model
    this._systemPrompt = config?.systemPrompt
  }

  async apply(context: StrategyContext): Promise<boolean> {
    if (this._utilization !== undefined && context.utilization < this._utilization) {
      logger.debug(
        `utilization=<${context.utilization}>, threshold=<${this._utilization}> | skipping summarization, below threshold`
      )
      return false
    }

    const { messages, agent } = context
    const model = this._model ?? (agent as unknown as Record<string, unknown>)['model'] as Model | undefined

    if (!model) {
      logger.warn('no model available for summarization')
      return false
    }

    let messagesToSummarize = Math.max(1, Math.floor(messages.length * this._summaryRatio))
    messagesToSummarize = Math.min(messagesToSummarize, messages.length - this._preserveRecent)

    if (messagesToSummarize <= 0) {
      logger.debug(
        `preserveRecent=<${this._preserveRecent}>, messages=<${messages.length}> | insufficient messages for summarization`
      )
      return false
    }

    try {
      messagesToSummarize = adjustSplitPointForToolPairs(messages, messagesToSummarize)
    } catch {
      logger.warn('unable to find valid split point for summarization')
      return false
    }

    const toSummarize = messages.slice(0, messagesToSummarize)

    if (toSummarize.length === 0) return false

    try {
      const summaryMessage = await generateSummary(toSummarize, model, this._systemPrompt)
      messages.splice(0, messagesToSummarize, summaryMessage)

      logger.debug(
        `summarized=<${messagesToSummarize}>, remaining=<${messages.length}> | summarized oldest messages`
      )
      return true
    } catch (error) {
      logger.warn(`error=<${error}> | summarization failed`)
      return false
    }
  }
}
