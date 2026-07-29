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
 *     // Replace tool results over 1500 tokens with a 750-token head-tail preview
 *     Offload.truncate("toolResults", { previewTokens: 750 })
 *       .when({ threshold: 1500 }),
 *
 *     // Aggressively truncate noisy tools — small preview, low threshold
 *     Offload.truncate(["bash", "list_files"], { previewTokens: 200 })
 *       .when({ threshold: 500 }),
 *
 *     // LLM-summarize all large blocks when context is 85% full, keep last 2 messages intact
 *     Offload.summarize()
 *       .when({ utilization: 0.85, preserveRecent: 2 }),
 *
 *     // Drop all error tool results from context immediately
 *     Offload("toolResultErrors"),
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
  DROPPED_MARKER,
  SUMMARIZED_PREFIX,
  estimateBlockTokens,
  estimateTextBlockTokens,
  extractBlockText,
  truncateToolResultBlock,
  truncateTextBlock,
  type TruncateConfig,
} from './methods/truncate.js'
import { summarizeText, type SummarizeConfig } from './methods/summarize.js'

/**
 * Target for offload operations.
 *
 * - `"toolResults"` — all successful tool result blocks
 * - `"toolResultErrors"` — all failed tool result blocks
 * - `"assistantMessages"` — text blocks in assistant messages
 * - `"userMessages"` — text blocks in user messages (excluding tool results)
 * - `string[]` — tool results from specific tools; prefix with `!` to exclude
 * - `undefined` — all content in the context window (tool results + text blocks)
 */
export type OffloadTarget = 'toolResults' | 'toolResultErrors' | 'assistantMessages' | 'userMessages' | string[]

/**
 * Conditions that determine when an offload strategy fires.
 *
 * All fields are optional. Omitting a condition means that dimension is not gated:
 * - No `threshold` → process blocks at any size
 * - No `utilization` → don't gate on context fullness
 * - No `preserveRecent` → don't protect any messages
 */
export interface OffloadConditions {
  /** Token threshold above which individual blocks are offloaded. Omit to process at any size. */
  threshold?: number

  /** Context utilization ratio (0-1) above which the strategy fires. Omit to always fire. */
  utilization?: number

  /** Number of most recent matching messages to leave untouched. Omit to protect nothing. */
  preserveRecent?: number
}

/**
 * Intermediate builder result that allows chaining `.when()` conditions.
 * Also implements `ContextStrategy` directly so it can be used without `.when()`.
 */
export interface OffloadStrategyBuilder extends ContextStrategy {
  /** Add conditions that determine when this strategy fires. */
  when(conditions: OffloadConditions): ContextStrategy
}

// --- Shared helpers ---

/**
 * Checks whether a ToolResultBlock matches the given offload target.
 * Handles status-based targets (toolResults/toolResultErrors) and name-based targets (string[]).
 */
function matchesToolTarget(
  block: ToolResultBlock,
  target: OffloadTarget,
  messages: Message[],
  toolFilter: Set<string> | undefined,
  excludeFilter: Set<string> | undefined
): boolean {
  if (target === 'toolResults') return block.status !== 'error'
  if (target === 'toolResultErrors') return block.status === 'error'

  const toolName = resolveToolName(block, messages)
  if (!toolName) return toolFilter === undefined && excludeFilter === undefined

  if (excludeFilter) return !excludeFilter.has(toolName)
  if (toolFilter) return toolFilter.has(toolName)

  return true
}

// --- Offload bare: drop from L0 entirely ---

/** Replaces matching content with a [Dropped] marker. */
class OffloadDropStrategy implements ContextStrategy {
  readonly name = 'offload:drop'

  private readonly _target: OffloadTarget | undefined
  private readonly _threshold: number
  private readonly _preserveRecent: number
  private readonly _toolFilter: Set<string> | undefined
  private readonly _excludeFilter: Set<string> | undefined

  constructor(target?: OffloadTarget, conditions?: OffloadConditions) {
    this._target = target
    this._threshold = conditions?.threshold ?? 0
    this._preserveRecent = conditions?.preserveRecent ?? 0

    const resolved = resolveToolFilter(target)
    this._toolFilter = resolved.include
    this._excludeFilter = resolved.exclude
  }

  /** Registers eager hook to drop content on arrival (disabled when preserveRecent \> 0). */
  init(context: StrategyInitContext): void {
    if (this._preserveRecent > 0) return
    const { agent } = context
    agent.addHook(MessageAddedEvent, async (event) => {
      const message = event.message
      this._processMessage(message, [message])
    })
  }

  /** Drops eligible blocks across all messages, respecting preserveRecent. */
  async apply(context: StrategyContext): Promise<boolean> {
    const { messages } = context
    const eligible =
      this._preserveRecent > 0
        ? excludeRecentMatches(messages, this._target, this._preserveRecent, this._toolFilter, this._excludeFilter)
        : messages
    let dropped = false

    for (const message of eligible) {
      if (this._processMessage(message, messages)) {
        dropped = true
      }
    }

    return dropped
  }

  /** Routes a single message to the appropriate drop handler based on target. */
  private _processMessage(message: Message, messages: Message[]): boolean {
    if (this._target === 'assistantMessages') {
      if (message.role !== 'assistant') return false
      return this._dropTextBlocks(message)
    }

    if (this._target === 'userMessages') {
      if (message.role !== 'user') return false
      return this._dropTextBlocks(message)
    }

    if (this._target === undefined) {
      let dropped = this._dropTextBlocks(message)
      if (message.role === 'user') {
        if (this._dropToolResults(message, messages)) dropped = true
      }
      return dropped
    }

    // Tool result targets
    if (message.role !== 'user') return false
    return this._dropToolResults(message, messages)
  }

  /** Replaces text blocks in a message with [Dropped] markers. */
  private _dropTextBlocks(message: Message): boolean {
    let dropped = false
    for (let blockIndex = message.content.length - 1; blockIndex >= 0; blockIndex--) {
      const block = message.content[blockIndex]!
      if (!(block instanceof TextBlock)) continue
      if (this._threshold > 0 && estimateTextBlockTokens(block) <= this._threshold) continue
      ;(message.content as unknown[])[blockIndex] = new TextBlock(DROPPED_MARKER)
      dropped = true
      logger.debug(`trackingId=<${message.trackingId}> | dropped text block from L0`)
    }
    return dropped
  }

  /** Replaces tool result blocks in a message with [Dropped] markers. */
  private _dropToolResults(message: Message, messages: Message[]): boolean {
    let dropped = false
    for (let blockIndex = message.content.length - 1; blockIndex >= 0; blockIndex--) {
      const block = message.content[blockIndex]!
      if (!(block instanceof ToolResultBlock)) continue
      if (
        this._target !== undefined &&
        !matchesToolTarget(block, this._target, messages, this._toolFilter, this._excludeFilter)
      )
        continue

      if (this._threshold > 0) {
        const tokens = estimateBlockTokens(block)
        if (tokens <= this._threshold) continue
      }

      const replacement = new ToolResultBlock({
        toolUseId: block.toolUseId,
        status: block.status,
        content: [new TextBlock(DROPPED_MARKER)],
      })
      ;(message.content as unknown[])[blockIndex] = replacement
      dropped = true
      logger.debug(`toolUseId=<${block.toolUseId}> | dropped tool result from L0`)
    }
    return dropped
  }
}

// --- Offload + Truncate strategy ---

/** Replaces matching content with a head-tail preview. */
class OffloadTruncateStrategy implements ContextStrategy {
  readonly name = 'offload:truncate'

  private readonly _threshold: number
  private readonly _truncateConfig: TruncateConfig
  private readonly _target: OffloadTarget | undefined
  private readonly _preserveRecent: number
  private readonly _toolFilter: Set<string> | undefined
  private readonly _excludeFilter: Set<string> | undefined

  constructor(target?: OffloadTarget, config?: TruncateConfig, conditions?: OffloadConditions) {
    this._threshold = conditions?.threshold ?? 0
    this._truncateConfig = config ?? {}
    this._target = target
    this._preserveRecent = conditions?.preserveRecent ?? 0

    const resolved = resolveToolFilter(target)
    this._toolFilter = resolved.include
    this._excludeFilter = resolved.exclude
  }

  /** Registers eager hook to truncate content on arrival (disabled when preserveRecent \> 0). */
  init(context: StrategyInitContext): void {
    if (this._preserveRecent > 0) return
    const { agent } = context
    agent.addHook(MessageAddedEvent, async (event) => {
      const message = event.message
      this._processMessage(message, [message])
    })
  }

  /** Truncates eligible blocks across all messages, respecting preserveRecent. */
  async apply(context: StrategyContext): Promise<boolean> {
    const { messages } = context
    const eligible =
      this._preserveRecent > 0
        ? excludeRecentMatches(messages, this._target, this._preserveRecent, this._toolFilter, this._excludeFilter)
        : messages
    let truncated = false

    for (const message of eligible) {
      if (this._processMessage(message, messages)) {
        truncated = true
      }
    }

    return truncated
  }

  /** Routes a single message to the appropriate truncate handler based on target. */
  private _processMessage(message: Message, messages: Message[]): boolean {
    if (this._target === 'assistantMessages') {
      if (message.role !== 'assistant') return false
      return this._truncateTextBlocks(message)
    }

    if (this._target === 'userMessages') {
      if (message.role !== 'user') return false
      return this._truncateTextBlocks(message)
    }

    if (this._target === undefined) {
      let truncated = this._truncateTextBlocks(message)
      if (message.role === 'user') {
        if (this._truncateToolResultBlocks(message, messages)) truncated = true
      }
      return truncated
    }

    // Tool result targets
    if (message.role !== 'user') return false
    return this._truncateToolResultBlocks(message, messages)
  }

  /** Replaces oversized text blocks with a head-tail preview. */
  private _truncateTextBlocks(message: Message): boolean {
    let truncated = false
    for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
      const block = message.content[blockIndex]!
      if (!(block instanceof TextBlock)) continue
      const tokens = estimateTextBlockTokens(block)
      if (tokens <= this._threshold) continue
      ;(message.content as unknown[])[blockIndex] = truncateTextBlock(block, this._truncateConfig)
      truncated = true
      logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | truncated text block`)
    }
    return truncated
  }

  /** Replaces oversized tool result blocks with a head-tail preview. */
  private _truncateToolResultBlocks(message: Message, messages: Message[]): boolean {
    let truncated = false
    for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
      const block = message.content[blockIndex]!
      if (!(block instanceof ToolResultBlock)) continue
      if (
        this._target !== undefined &&
        !matchesToolTarget(block, this._target, messages, this._toolFilter, this._excludeFilter)
      )
        continue

      const estimatedTokens = estimateBlockTokens(block)
      if (estimatedTokens <= this._threshold) continue

      const replacement = truncateToolResultBlock(block, this._truncateConfig)
      ;(message.content as unknown[])[blockIndex] = replacement
      truncated = true
      logger.debug(`toolUseId=<${block.toolUseId}>, tokens=<${estimatedTokens}> | truncated tool result`)
    }
    return truncated
  }
}

// --- Offload + Summarize strategy (block-level) ---

/** Replaces matching content with an LLM-generated summary. */
class OffloadSummarizeStrategy implements ContextStrategy {
  readonly name = 'offload:summarize'

  private readonly _config: SummarizeConfig
  private readonly _target: OffloadTarget | undefined
  private readonly _threshold: number
  private readonly _utilization: number | undefined
  private readonly _preserveRecent: number
  private readonly _toolFilter: Set<string> | undefined
  private readonly _excludeFilter: Set<string> | undefined

  constructor(target?: OffloadTarget, config?: SummarizeConfig, conditions?: OffloadConditions) {
    this._config = config ?? {}
    this._target = target
    this._threshold = conditions?.threshold ?? 0
    this._utilization = conditions?.utilization
    this._preserveRecent = conditions?.preserveRecent ?? 0

    const resolved = resolveToolFilter(target)
    this._toolFilter = resolved.include
    this._excludeFilter = resolved.exclude
  }

  /** Registers eager hook (disabled when preserveRecent \> 0 or utilization is set). */
  init(context: StrategyInitContext): void {
    if (this._preserveRecent > 0) return
    if (this._utilization !== undefined) return
    const { agent } = context
    agent.addHook(MessageAddedEvent, async (event) => {
      const message = event.message
      const model = this._config.model ?? ((agent as unknown as Record<string, unknown>)['model'] as Model | undefined)
      if (!model) return
      await this._processMessage(message, [message], model)
    })
  }

  /** Summarizes eligible blocks across all messages, respecting utilization and preserveRecent. */
  async apply(context: StrategyContext): Promise<boolean> {
    if (this._utilization !== undefined && context.utilization < this._utilization) {
      logger.debug(
        `utilization=<${context.utilization}>, threshold=<${this._utilization}> | skipping summarization, below threshold`
      )
      return false
    }

    const { messages, agent } = context
    const model = this._config.model ?? ((agent as unknown as Record<string, unknown>)['model'] as Model | undefined)

    if (!model) {
      logger.warn('no model available for summarization')
      return false
    }

    if (messages.length === 0) return false

    const eligible =
      this._preserveRecent > 0
        ? excludeRecentMatches(messages, this._target, this._preserveRecent, this._toolFilter, this._excludeFilter)
        : messages

    let summarized = false

    for (const message of eligible) {
      if (await this._processMessage(message, messages, model)) {
        summarized = true
      }
    }

    return summarized
  }

  /** Routes a single message to the appropriate summarize handler based on target. */
  private async _processMessage(message: Message, messages: Message[], model: Model): Promise<boolean> {
    if (this._target === 'assistantMessages') {
      if (message.role !== 'assistant') return false
      return this._summarizeTextBlocks(message, model)
    }

    if (this._target === 'userMessages') {
      if (message.role !== 'user') return false
      return this._summarizeTextBlocks(message, model)
    }

    if (this._target === undefined) {
      let summarized = await this._summarizeTextBlocks(message, model)
      if (message.role === 'user') {
        if (await this._summarizeToolResultBlocks(message, messages, model)) summarized = true
      }
      return summarized
    }

    // Tool result targets
    if (message.role !== 'user') return false
    return this._summarizeToolResultBlocks(message, messages, model)
  }

  /** Replaces oversized text blocks with an LLM-generated summary. */
  private async _summarizeTextBlocks(message: Message, model: Model): Promise<boolean> {
    let summarized = false
    for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
      const block = message.content[blockIndex]!
      if (!(block instanceof TextBlock)) continue
      const tokens = estimateTextBlockTokens(block)
      if (tokens <= this._threshold) continue

      const summary = await summarizeText(block.text, model, this._config)
      if (summary) {
        ;(message.content as unknown[])[blockIndex] = new TextBlock(
          `${SUMMARIZED_PREFIX} ~${tokens.toLocaleString()} tokens]\n\n${summary}`
        )
        summarized = true
        logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | summarized text block`)
      }
    }
    return summarized
  }

  /** Replaces oversized tool result blocks with an LLM-generated summary. */
  private async _summarizeToolResultBlocks(message: Message, messages: Message[], model: Model): Promise<boolean> {
    let summarized = false
    for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
      const block = message.content[blockIndex]!
      if (!(block instanceof ToolResultBlock)) continue
      if (
        this._target !== undefined &&
        !matchesToolTarget(block, this._target, messages, this._toolFilter, this._excludeFilter)
      )
        continue

      const tokens = estimateBlockTokens(block)
      if (tokens <= this._threshold) continue

      const fullText = extractBlockText(block)
      const summary = await summarizeText(fullText, model, this._config)
      if (summary) {
        const replacement = new ToolResultBlock({
          toolUseId: block.toolUseId,
          status: block.status,
          content: [new TextBlock(`${SUMMARIZED_PREFIX} ~${tokens.toLocaleString()} tokens]\n\n${summary}`)],
        })
        ;(message.content as unknown[])[blockIndex] = replacement
        summarized = true
        logger.debug(`toolUseId=<${block.toolUseId}>, tokens=<${tokens}> | summarized tool result`)
      }
    }
    return summarized
  }
}

// --- preserveRecent helper ---

/**
 * Returns messages excluding the N most recent that match the target.
 * Walks messages from newest to oldest, counting matches, and excludes the first `count` matches.
 */
function excludeRecentMatches(
  messages: Message[],
  target: OffloadTarget | undefined,
  count: number,
  toolFilter: Set<string> | undefined,
  excludeFilter: Set<string> | undefined
): Message[] {
  const excluded = new Set<Message>()
  let remaining = count

  for (let index = messages.length - 1; index >= 0 && remaining > 0; index--) {
    const message = messages[index]!
    if (messageMatchesTarget(message, target, messages, toolFilter, excludeFilter)) {
      excluded.add(message)
      remaining--
    }
  }

  return messages.filter((message) => !excluded.has(message))
}

/**
 * Checks whether a message matches the given target for preserveRecent counting.
 * A message matches if it contains content that the target would select.
 */
function messageMatchesTarget(
  message: Message,
  target: OffloadTarget | undefined,
  messages: Message[],
  toolFilter: Set<string> | undefined,
  excludeFilter: Set<string> | undefined
): boolean {
  if (target === undefined) return true

  if (target === 'assistantMessages') return message.role === 'assistant'
  if (target === 'userMessages') return message.role === 'user'

  // Tool result targets — must be a user message with a matching tool result
  if (message.role !== 'user') return false
  for (const block of message.content) {
    if (block instanceof ToolResultBlock) {
      if (matchesToolTarget(block, target, messages, toolFilter, excludeFilter)) return true
    }
  }
  return false
}

// --- Builder ---

/** Creates a drop strategy builder with an optional target. */
function createDropBuilder(target?: OffloadTarget): OffloadStrategyBuilder {
  const strategy = new OffloadDropStrategy(target)
  return {
    get name(): string {
      return strategy.name
    },
    init: strategy.init.bind(strategy),
    apply: strategy.apply.bind(strategy),
    when(conditions: OffloadConditions): ContextStrategy {
      return new OffloadDropStrategy(target, conditions)
    },
  }
}

/** Creates a truncate strategy builder with an optional target and config. */
function createTruncateBuilder(target?: OffloadTarget, config?: TruncateConfig): OffloadStrategyBuilder {
  const strategy = new OffloadTruncateStrategy(target, config)
  return {
    get name(): string {
      return strategy.name
    },
    init: strategy.init.bind(strategy),
    apply: strategy.apply.bind(strategy),
    when(conditions: OffloadConditions): ContextStrategy {
      return new OffloadTruncateStrategy(target, config, conditions)
    },
  }
}

/** Creates a summarize strategy builder with an optional target and config. */
function createSummarizeBuilder(target?: OffloadTarget, config?: SummarizeConfig): OffloadStrategyBuilder {
  const strategy = new OffloadSummarizeStrategy(target, config)
  return {
    get name(): string {
      return strategy.name
    },
    init: strategy.init.bind(strategy),
    apply: strategy.apply.bind(strategy),
    when(conditions: OffloadConditions): ContextStrategy {
      return new OffloadSummarizeStrategy(target, config, conditions)
    },
  }
}

/**
 * Offload strategy builder namespace.
 *
 * - `Offload(target)` — drop matching content from L0 entirely
 * - `Offload.truncate(target, config)` — replace with a preview
 * - `Offload.summarize(target, config)` — replace with LLM-generated summary
 */
export interface OffloadNamespace {
  /** Drop matching content from L0 entirely. */
  (target?: OffloadTarget): OffloadStrategyBuilder

  /** Replace oversized content with a preview. */
  truncate(target?: OffloadTarget, config?: TruncateConfig): OffloadStrategyBuilder

  /** Replace oversized content with a preview (config-only, targets everything). */
  truncate(config: TruncateConfig): OffloadStrategyBuilder

  /** Replace oversized content with an LLM-generated summary. */
  summarize(target?: OffloadTarget, config?: SummarizeConfig): OffloadStrategyBuilder

  /** Replace oversized content with an LLM-generated summary (config-only, targets everything). */
  summarize(config: SummarizeConfig): OffloadStrategyBuilder
}

function offloadFn(target?: OffloadTarget): OffloadStrategyBuilder {
  return createDropBuilder(target)
}

offloadFn.truncate = function truncate(
  targetOrConfig?: OffloadTarget | TruncateConfig,
  config?: TruncateConfig
): OffloadStrategyBuilder {
  if (targetOrConfig === undefined) {
    return createTruncateBuilder(undefined, config)
  }
  if (isTruncateConfig(targetOrConfig)) {
    return createTruncateBuilder(undefined, targetOrConfig)
  }
  return createTruncateBuilder(targetOrConfig as OffloadTarget, config)
}

offloadFn.summarize = function summarize(
  targetOrConfig?: OffloadTarget | SummarizeConfig,
  config?: SummarizeConfig
): OffloadStrategyBuilder {
  if (targetOrConfig === undefined) {
    return createSummarizeBuilder(undefined, config)
  }
  if (isSummarizeConfig(targetOrConfig)) {
    return createSummarizeBuilder(undefined, targetOrConfig)
  }
  return createSummarizeBuilder(targetOrConfig as OffloadTarget, config)
}

/** Disambiguates the truncate overload: is the first arg a config object or a target? */
function isTruncateConfig(value: unknown): value is TruncateConfig {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const keys = Object.keys(value)
  if (keys.length === 0) return true
  return keys.some((key) => key === 'previewTokens' || key === 'preview')
}

/** Disambiguates the summarize overload: is the first arg a config object or a target? */
function isSummarizeConfig(value: unknown): value is SummarizeConfig {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const keys = Object.keys(value)
  if (keys.length === 0) return true
  return keys.some((key) => key === 'model' || key === 'systemPrompt')
}

/**
 * Builder for offload strategies — reduces content in L0.
 *
 * @example
 * ```typescript
 * Offload.truncate("toolResults", { previewTokens: 750 }).when({ threshold: 1500 })
 * Offload.summarize().when({ utilization: 0.85, preserveRecent: 2 })
 * Offload("toolResultErrors")  // drop from L0 entirely
 * ```
 */
export const Offload: OffloadNamespace = offloadFn as OffloadNamespace

// --- Helpers ---

/**
 * Resolves the tool name for a ToolResultBlock by finding the corresponding ToolUseBlock
 * in the preceding assistant message (where ToolUseBlocks live).
 */
function resolveToolName(block: ToolResultBlock, messages: Message[]): string | undefined {
  for (const message of messages) {
    if (message.role !== 'assistant') continue
    for (const content of message.content) {
      if (
        'toolUseId' in content &&
        'name' in content &&
        (content as { toolUseId: string }).toolUseId === block.toolUseId
      ) {
        return (content as { name: string }).name
      }
    }
  }
  return undefined
}

/**
 * Parses a string[] target into include/exclude filter sets.
 * Entries prefixed with `!` become excludes; all others become includes.
 */
function resolveToolFilter(target: OffloadTarget | undefined): { include?: Set<string>; exclude?: Set<string> } {
  if (target === undefined) return {}
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
    logger.warn('tool filter contains both includes and excludes, excludes will be ignored')
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
