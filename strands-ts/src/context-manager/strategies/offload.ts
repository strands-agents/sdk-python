/**
 * Builder API for offload strategies.
 *
 * Offload strategies reduce content in L0 (the context window).
 * The builder composes a target, a reduction method (truncate, drop, or summarize),
 * and optional conditions into a strategy that implements `ContextStrategy`.
 *
 * @example
 * ```typescript
 * import { ContextManager, Offload } from '@strands-agents/sdk'
 *
 * const cm = new ContextManager({
 *   strategies: [
 *     Offload.truncate("toolResults", { previewTokens: 750 })
 *       .when({ threshold: 1500 }),
 *     Offload.truncate(["bash", "list_files"], { previewTokens: 200 })
 *       .when({ threshold: 500 }),
 *     Offload.summarize("toolResults", { ratio: 0.3 })
 *       .when({ threshold: 2000, utilization: 0.85 }),
 *     Offload("toolResultErrors"),  // drop from L0 entirely
 *   ],
 * })
 * ```
 */

import { logger } from '../../logging/logger.js'
import { MessageAddedEvent } from '../../hooks/events.js'
import { TextBlock, ToolResultBlock } from '../../types/messages.js'
import type { Message } from '../../types/messages.js'
import type { Model } from '../../models/model.js'
import type { ContextStrategy, StrategyContext, StrategyInitContext } from '../types.js'
import {
  estimateBlockTokens,
  estimateTextBlockTokens,
  extractBlockText,
  isAlreadyProcessed,
  truncateToolResultBlock,
  truncateTextBlock,
  type TruncateConfig,
} from './methods/truncate.js'
import { summarizeText, type SummarizeConfig } from './methods/summarize.js'

const DEFAULT_THRESHOLD = 2500

/**
 * Target for offload operations.
 *
 * - `"toolResults"` — all successful tool result blocks
 * - `"toolResultErrors"` — all failed tool result blocks
 * - `"assistantMessages"` — text blocks in assistant messages
 * - `"userMessages"` — text blocks in user messages (excluding tool results)
 * - `"images"` — image blocks (planned, not yet implemented)
 * - `"documents"` — document blocks (planned, not yet implemented)
 * - `string[]` — tool results from specific tools; prefix with `!` to exclude
 */
export type OffloadTarget =
  | 'toolResults'
  | 'toolResultErrors'
  | 'assistantMessages'
  | 'userMessages'
  | 'images'
  | 'documents'
  | string[]

/**
 * Conditions that determine when a strategy fires.
 */
export interface WhenConditions {
  /** Token threshold above which individual blocks are offloaded (truncate/summarize). */
  threshold?: number

  /** Context utilization ratio (0-1) above which the strategy fires. */
  utilization?: number
}

/**
 * Intermediate builder result that allows chaining `.when()` conditions.
 * Also implements `ContextStrategy` directly so it can be used without `.when()`.
 */
export interface StrategyBuilder extends ContextStrategy {
  /** Add conditions that determine when this strategy fires. */
  when(conditions: WhenConditions): ContextStrategy
}


// --- Offload bare: drop from L0 entirely ---

class OffloadDropStrategy implements ContextStrategy {
  readonly name = 'offload:drop'

  private readonly _target: OffloadTarget
  private readonly _threshold: number
  private readonly _toolFilter: Set<string> | undefined
  private readonly _excludeFilter: Set<string> | undefined

  constructor(target: OffloadTarget, conditions?: WhenConditions) {
    this._target = target
    this._threshold = conditions?.threshold ?? 0

    const resolved = resolveToolFilter(target)
    this._toolFilter = resolved.include
    this._excludeFilter = resolved.exclude
  }

  init(context: StrategyInitContext): void {
    const { agent } = context
    agent.addHook(MessageAddedEvent, async (event) => {
      const message = event.message
      this._processMessage(message)
    })
  }

  async apply(context: StrategyContext): Promise<boolean> {
    const { messages } = context
    let dropped = false

    for (const message of messages) {
      if (this._processMessage(message)) {
        dropped = true
      }
    }

    return dropped
  }

  private _processMessage(message: Message): boolean {
    let dropped = false

    if (this._target === 'assistantMessages') {
      if (message.role !== 'assistant') return false
      for (let blockIndex = message.content.length - 1; blockIndex >= 0; blockIndex--) {
        const block = message.content[blockIndex]!
        if (!(block instanceof TextBlock)) continue
        if (isAlreadyProcessed(block)) continue
        if (this._threshold > 0 && estimateTextBlockTokens(block) <= this._threshold) continue
        ;(message.content as unknown[])[blockIndex] = new TextBlock('[Dropped]')
        dropped = true
        logger.debug(`trackingId=<${message.trackingId}> | dropped assistant text block from L0`)
      }
      return dropped
    }

    if (this._target === 'userMessages') {
      if (message.role !== 'user') return false
      for (let blockIndex = message.content.length - 1; blockIndex >= 0; blockIndex--) {
        const block = message.content[blockIndex]!
        if (!(block instanceof TextBlock)) continue
        if (isAlreadyProcessed(block)) continue
        if (this._threshold > 0 && estimateTextBlockTokens(block) <= this._threshold) continue
        ;(message.content as unknown[])[blockIndex] = new TextBlock('[Dropped]')
        dropped = true
        logger.debug(`trackingId=<${message.trackingId}> | dropped user text block from L0`)
      }
      return dropped
    }

    // Tool result targets
    if (message.role !== 'user') return false
    for (let blockIndex = message.content.length - 1; blockIndex >= 0; blockIndex--) {
      const block = message.content[blockIndex]!
      if (!(block instanceof ToolResultBlock)) continue
      if (!this._matchesToolTarget(block, message)) continue
      if (isAlreadyProcessed(block)) continue

      if (this._threshold > 0) {
        const tokens = estimateBlockTokens(block)
        if (tokens <= this._threshold) continue
      }

      const replacement = new ToolResultBlock({
        toolUseId: block.toolUseId,
        status: block.status,
        content: [new TextBlock('[Dropped]')],
      })
      ;(message.content as unknown[])[blockIndex] = replacement
      dropped = true
      logger.debug(`toolUseId=<${block.toolUseId}> | dropped tool result from L0`)
    }

    return dropped
  }

  private _matchesToolTarget(block: ToolResultBlock, message: Message): boolean {
    if (this._target === 'toolResults') return block.status !== 'error'
    if (this._target === 'toolResultErrors') return block.status === 'error'

    const toolName = resolveToolName(block, message)
    if (!toolName) return this._toolFilter === undefined && this._excludeFilter === undefined

    if (this._excludeFilter) return !this._excludeFilter.has(toolName)
    if (this._toolFilter) return this._toolFilter.has(toolName)

    return true
  }
}

// --- Offload + Truncate strategy ---

class OffloadTruncateStrategy implements ContextStrategy {
  readonly name = 'offload:truncate'

  private readonly _threshold: number
  private readonly _truncateConfig: TruncateConfig
  private readonly _target: OffloadTarget
  private readonly _toolFilter: Set<string> | undefined
  private readonly _excludeFilter: Set<string> | undefined

  constructor(target: OffloadTarget, config?: TruncateConfig, conditions?: WhenConditions) {
    this._threshold = conditions?.threshold ?? DEFAULT_THRESHOLD
    this._truncateConfig = config ?? {}
    this._target = target

    const resolved = resolveToolFilter(target)
    this._toolFilter = resolved.include
    this._excludeFilter = resolved.exclude
  }

  init(context: StrategyInitContext): void {
    const { agent } = context
    agent.addHook(MessageAddedEvent, async (event) => {
      const message = event.message
      this._processMessage(message)
    })
  }

  async apply(context: StrategyContext): Promise<boolean> {
    const { messages } = context
    let truncated = false

    for (const message of messages) {
      if (this._processMessage(message)) {
        truncated = true
      }
    }

    return truncated
  }

  private _processMessage(message: Message): boolean {
    let truncated = false

    if (this._target === 'assistantMessages') {
      if (message.role !== 'assistant') return false
      for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
        const block = message.content[blockIndex]!
        if (!(block instanceof TextBlock)) continue
        if (isAlreadyProcessed(block)) continue
        const tokens = estimateTextBlockTokens(block)
        if (tokens <= this._threshold) continue
        ;(message.content as unknown[])[blockIndex] = truncateTextBlock(block, this._truncateConfig)
        truncated = true
        logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | truncated assistant text block`)
      }
      return truncated
    }

    if (this._target === 'userMessages') {
      if (message.role !== 'user') return false
      for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
        const block = message.content[blockIndex]!
        if (!(block instanceof TextBlock)) continue
        if (isAlreadyProcessed(block)) continue
        const tokens = estimateTextBlockTokens(block)
        if (tokens <= this._threshold) continue
        ;(message.content as unknown[])[blockIndex] = truncateTextBlock(block, this._truncateConfig)
        truncated = true
        logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | truncated user text block`)
      }
      return truncated
    }

    // Tool result targets
    if (message.role !== 'user') return false
    for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
      const block = message.content[blockIndex]!
      if (!(block instanceof ToolResultBlock)) continue
      if (!this._matchesToolTarget(block, message)) continue
      if (isAlreadyProcessed(block)) continue

      const estimatedTokens = estimateBlockTokens(block)
      if (estimatedTokens <= this._threshold) continue

      const replacement = truncateToolResultBlock(block, this._truncateConfig)
      ;(message.content as unknown[])[blockIndex] = replacement
      truncated = true
      logger.debug(
        `toolUseId=<${block.toolUseId}>, tokens=<${estimatedTokens}> | truncated tool result`
      )
    }

    return truncated
  }

  private _matchesToolTarget(block: ToolResultBlock, message: Message): boolean {
    if (this._target === 'toolResults') return block.status !== 'error'
    if (this._target === 'toolResultErrors') return block.status === 'error'

    const toolName = resolveToolName(block, message)
    if (!toolName) return this._toolFilter === undefined && this._excludeFilter === undefined

    if (this._excludeFilter) return !this._excludeFilter.has(toolName)
    if (this._toolFilter) return this._toolFilter.has(toolName)

    return true
  }
}

// --- Offload + Summarize strategy (block-level) ---

class OffloadSummarizeStrategy implements ContextStrategy {
  readonly name = 'offload:summarize'

  private readonly _config: SummarizeConfig
  private readonly _target: OffloadTarget
  private readonly _threshold: number
  private readonly _utilization: number | undefined
  private readonly _toolFilter: Set<string> | undefined
  private readonly _excludeFilter: Set<string> | undefined

  constructor(target: OffloadTarget, config?: SummarizeConfig, conditions?: WhenConditions) {
    this._config = config ?? {}
    this._target = target
    this._threshold = conditions?.threshold ?? DEFAULT_THRESHOLD
    this._utilization = conditions?.utilization

    const resolved = resolveToolFilter(target)
    this._toolFilter = resolved.include
    this._excludeFilter = resolved.exclude
  }

  init(context: StrategyInitContext): void {
    const { agent } = context
    agent.addHook(MessageAddedEvent, async (event) => {
      const message = event.message
      const model = this._config.model ?? (agent as unknown as Record<string, unknown>)['model'] as Model | undefined
      if (!model) return
      await this._processMessage(message, model)
    })
  }

  async apply(context: StrategyContext): Promise<boolean> {
    if (this._utilization !== undefined && context.utilization < this._utilization) {
      logger.debug(
        `utilization=<${context.utilization}>, threshold=<${this._utilization}> | skipping summarization, below threshold`
      )
      return false
    }

    const { messages, agent } = context
    const model = this._config.model ?? (agent as unknown as Record<string, unknown>)['model'] as Model | undefined

    if (!model) {
      logger.warn('no model available for summarization')
      return false
    }

    let summarized = false

    for (const message of messages) {
      if (await this._processMessage(message, model)) {
        summarized = true
      }
    }

    return summarized
  }

  private async _processMessage(message: Message, model: Model): Promise<boolean> {
    let summarized = false

    if (this._target === 'assistantMessages') {
      if (message.role !== 'assistant') return false
      for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
        const block = message.content[blockIndex]!
        if (!(block instanceof TextBlock)) continue
        if (isAlreadyProcessed(block)) continue
        const tokens = estimateTextBlockTokens(block)
        if (tokens <= this._threshold) continue

        const summary = await summarizeText(block.text, model, this._config)
        if (summary) {
          ;(message.content as unknown[])[blockIndex] = new TextBlock(`[Summarized: ~${tokens.toLocaleString()} tokens]\n\n${summary}`)
          summarized = true
          logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | summarized assistant text block`)
        }
      }
      return summarized
    }

    if (this._target === 'userMessages') {
      if (message.role !== 'user') return false
      for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
        const block = message.content[blockIndex]!
        if (!(block instanceof TextBlock)) continue
        if (isAlreadyProcessed(block)) continue
        const tokens = estimateTextBlockTokens(block)
        if (tokens <= this._threshold) continue

        const summary = await summarizeText(block.text, model, this._config)
        if (summary) {
          ;(message.content as unknown[])[blockIndex] = new TextBlock(`[Summarized: ~${tokens.toLocaleString()} tokens]\n\n${summary}`)
          summarized = true
          logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | summarized user text block`)
        }
      }
      return summarized
    }

    // Tool result targets
    if (message.role !== 'user') return false
    for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
      const block = message.content[blockIndex]!
      if (!(block instanceof ToolResultBlock)) continue
      if (!this._matchesToolTarget(block, message)) continue
      if (isAlreadyProcessed(block)) continue

      const tokens = estimateBlockTokens(block)
      if (tokens <= this._threshold) continue

      const fullText = extractBlockText(block)
      const summary = await summarizeText(fullText, model, this._config)
      if (summary) {
        const replacement = new ToolResultBlock({
          toolUseId: block.toolUseId,
          status: block.status,
          content: [new TextBlock(`[Summarized: ~${tokens.toLocaleString()} tokens]\n\n${summary}`)],
        })
        ;(message.content as unknown[])[blockIndex] = replacement
        summarized = true
        logger.debug(`toolUseId=<${block.toolUseId}>, tokens=<${tokens}> | summarized tool result`)
      }
    }

    return summarized
  }

  private _matchesToolTarget(block: ToolResultBlock, message: Message): boolean {
    if (this._target === 'toolResults') return block.status !== 'error'
    if (this._target === 'toolResultErrors') return block.status === 'error'

    const toolName = resolveToolName(block, message)
    if (!toolName) return this._toolFilter === undefined && this._excludeFilter === undefined

    if (this._excludeFilter) return !this._excludeFilter.has(toolName)
    if (this._toolFilter) return this._toolFilter.has(toolName)

    return true
  }
}

// --- Builder ---

function createDropBuilder(target: OffloadTarget): StrategyBuilder {
  const strategy = new OffloadDropStrategy(target)
  return {
    get name(): string {
      return strategy.name
    },
    init: strategy.init.bind(strategy),
    apply: strategy.apply.bind(strategy),
    when(conditions: WhenConditions): ContextStrategy {
      return new OffloadDropStrategy(target, conditions)
    },
  }
}

function createTruncateBuilder(target: OffloadTarget, config?: TruncateConfig): StrategyBuilder {
  const strategy = new OffloadTruncateStrategy(target, config)
  return {
    get name(): string {
      return strategy.name
    },
    init: strategy.init.bind(strategy),
    apply: strategy.apply.bind(strategy),
    when(conditions: WhenConditions): ContextStrategy {
      return new OffloadTruncateStrategy(target, config, conditions)
    },
  }
}

function createSummarizeBuilder(target: OffloadTarget, config?: SummarizeConfig): StrategyBuilder {
  const strategy = new OffloadSummarizeStrategy(target, config)
  return {
    get name(): string {
      return strategy.name
    },
    init: strategy.init.bind(strategy),
    apply: strategy.apply.bind(strategy),
    when(conditions: WhenConditions): ContextStrategy {
      return new OffloadSummarizeStrategy(target, config, conditions)
    },
  }
}

/**
 * Offload strategy builder namespace.
 *
 * - `Offload(target)` — drop matching content from L0 entirely
 * - `Offload.truncate(target, config)` — replace with head-tail preview
 * - `Offload.summarize(target, config)` — replace with LLM-generated summary
 */
export interface OffloadNamespace {
  /** Drop matching content from L0 entirely. */
  (target: OffloadTarget): StrategyBuilder

  /** Replace oversized content with a head-tail preview. */
  truncate(target: OffloadTarget, config?: TruncateConfig): StrategyBuilder

  /** Replace oversized content with an LLM-generated summary. */
  summarize(target: OffloadTarget, config?: SummarizeConfig): StrategyBuilder
}

function offloadFn(target: OffloadTarget): StrategyBuilder {
  return createDropBuilder(target)
}

offloadFn.truncate = function truncate(target: OffloadTarget, config?: TruncateConfig): StrategyBuilder {
  return createTruncateBuilder(target, config)
}

offloadFn.summarize = function summarize(target: OffloadTarget, config?: SummarizeConfig): StrategyBuilder {
  return createSummarizeBuilder(target, config)
}

/**
 * Builder for offload strategies — reduces content in L0.
 *
 * @example
 * ```typescript
 * Offload.truncate("toolResults", { previewTokens: 750 }).when({ threshold: 1500 })
 * Offload.summarize("toolResults", { ratio: 0.3 }).when({ threshold: 2000 })
 * Offload("toolResultErrors")  // drop from L0 entirely
 * ```
 */
export const Offload: OffloadNamespace = offloadFn as OffloadNamespace

// --- Helpers ---

function resolveToolName(block: ToolResultBlock, message: Message): string | undefined {
  for (const content of message.content) {
    if ('toolUseId' in content && 'name' in content && (content as { toolUseId: string }).toolUseId === block.toolUseId) {
      return (content as { name: string }).name
    }
  }
  return undefined
}

function resolveToolFilter(target: OffloadTarget): { include?: Set<string>; exclude?: Set<string> } {
  if (typeof target === 'string') return {}
  if (!Array.isArray(target)) return {}

  const includes: string[] = []
  const excludes: string[] = []

  for (const entry of target) {
    if (entry.startsWith('!')) {
      excludes.push(entry.slice(1))
    } else {
      includes.push(entry)
    }
  }

  if (excludes.length > 0 && includes.length > 0) {
    return { include: new Set(includes) }
  }
  if (excludes.length > 0) {
    return { exclude: new Set(excludes) }
  }
  if (includes.length > 0) {
    return { include: new Set(includes) }
  }

  return {}
}
