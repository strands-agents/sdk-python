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
 *     // Drop error tool results over 500 tokens
 *     Offload("toolResultErrors").when({ threshold: 500 }),
 *   ],
 * })
 * ```
 */

import { logger } from '../../logging/logger.js'
import { MessageAddedEvent } from '../../hooks/events.js'
import { TextBlock, ToolResultBlock } from '../../types/messages.js'
import type { Message } from '../../types/messages.js'
import type { Model } from '../../models/model.js'
import type { LocalAgent } from '../../types/agent.js'
import type { ContextStrategy, StrategyContext } from '../types.js'
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
 * Target for offload operations. This union is intentionally extensible — new
 * string-literal members can be added freely as new content categories emerge.
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
  if (target === 'toolResults') return block.status === 'success'
  if (target === 'toolResultErrors') return block.status === 'error'

  const toolName = resolveToolName(block, messages)
  if (!toolName) return toolFilter === undefined && excludeFilter === undefined

  if (excludeFilter) return !excludeFilter.has(toolName)
  if (toolFilter) return toolFilter.has(toolName)

  return true
}

// --- Base strategy class ---

/** Shared offload logic: target routing, eager hooks, preserveRecent. */
abstract class BaseOffloadStrategy implements ContextStrategy {
  abstract readonly name: string

  protected readonly _target: OffloadTarget | undefined
  protected readonly _threshold: number
  protected readonly _preserveRecent: number
  protected readonly _toolFilter: Set<string> | undefined
  protected readonly _excludeFilter: Set<string> | undefined

  constructor(target?: OffloadTarget, conditions?: OffloadConditions) {
    this._target = target
    this._threshold = conditions?.threshold ?? 0
    this._preserveRecent = conditions?.preserveRecent ?? 0

    const resolved = resolveToolFilter(target)
    this._toolFilter = resolved.include
    this._excludeFilter = resolved.exclude
  }

  init(agent: LocalAgent): void {
    if (this._preserveRecent > 0) return
    if (!this._shouldRegisterEagerHook()) return
    agent.addHook(MessageAddedEvent, async (event) => {
      const messages = agent.messages
      const lastMessage = messages[messages.length - 1]
      if (event.message === lastMessage) return
      await this._processMessage(event.message, messages)
    })
  }

  async apply(context: StrategyContext): Promise<boolean> {
    if (!this._shouldApply(context)) return false

    const { messages } = context
    const eligible =
      this._preserveRecent > 0
        ? excludeRecentMatches(messages, this._target, this._preserveRecent, this._toolFilter, this._excludeFilter)
        : messages
    let acted = false

    for (const message of eligible) {
      if (await this._processMessage(message, messages)) {
        acted = true
      }
    }

    return acted
  }

  /** Override to add extra gates (e.g. utilization check for summarize). */
  protected _shouldApply(_context: StrategyContext): boolean {
    return true
  }

  /** Override to disable eager hook registration (e.g. when utilization is set). */
  protected _shouldRegisterEagerHook(): boolean {
    return true
  }

  /** Routes a single message to text block or tool result handlers based on target. */
  private async _processMessage(message: Message, messages: Message[]): Promise<boolean> {
    if (this._target === 'assistantMessages') {
      if (message.role !== 'assistant') return false
      return this._transformTextBlocks(message)
    }

    if (this._target === 'userMessages') {
      if (message.role !== 'user') return false
      return this._transformTextBlocks(message)
    }

    if (this._target === undefined) {
      let acted = await this._transformTextBlocks(message)
      if (message.role === 'user') {
        if (await this._transformToolResultBlocks(message, messages)) acted = true
      }
      return acted
    }

    // Tool result targets
    if (message.role !== 'user') return false
    return this._transformToolResultBlocks(message, messages)
  }

  /** Process text blocks in a message. Subclasses implement the transformation. */
  private async _transformTextBlocks(message: Message): Promise<boolean> {
    let acted = false
    for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
      const block = message.content[blockIndex]!
      if (!(block instanceof TextBlock)) continue
      const tokens = estimateTextBlockTokens(block)
      if (tokens <= this._threshold) continue

      const replacement = await this._replaceTextBlock(block, tokens, message)
      if (replacement && replacement.text !== block.text) {
        ;(message.content as unknown[])[blockIndex] = replacement
        acted = true
      }
    }
    return acted
  }

  /** Process tool result blocks in a message. Subclasses implement the transformation. */
  private async _transformToolResultBlocks(message: Message, messages: Message[]): Promise<boolean> {
    let acted = false
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

      const replacement = await this._replaceToolResultBlock(block, tokens)
      if (replacement) {
        ;(message.content as unknown[])[blockIndex] = replacement
        acted = true
      }
    }
    return acted
  }

  /** Transform a text block. Return the replacement, or null to skip. */
  protected abstract _replaceTextBlock(block: TextBlock, tokens: number, message: Message): Promise<TextBlock | null>

  /** Transform a tool result block. Return the replacement, or null to skip. */
  protected abstract _replaceToolResultBlock(block: ToolResultBlock, tokens: number): Promise<ToolResultBlock | null>
}

// --- Drop strategy ---

class OffloadDropStrategy extends BaseOffloadStrategy {
  readonly name = 'offload:drop'

  protected async _replaceTextBlock(block: TextBlock, _tokens: number, message: Message): Promise<TextBlock | null> {
    logger.debug(`trackingId=<${message.trackingId}> | dropped text block from L0`)
    return new TextBlock(DROPPED_MARKER)
  }

  protected async _replaceToolResultBlock(block: ToolResultBlock, _tokens: number): Promise<ToolResultBlock | null> {
    logger.debug(`toolUseId=<${block.toolUseId}> | dropped tool result from L0`)
    return new ToolResultBlock({
      toolUseId: block.toolUseId,
      status: block.status,
      content: [new TextBlock(DROPPED_MARKER)],
    })
  }
}

// --- Truncate strategy ---

class OffloadTruncateStrategy extends BaseOffloadStrategy {
  readonly name = 'offload:truncate'

  private readonly _truncateConfig: TruncateConfig

  constructor(target?: OffloadTarget, config?: TruncateConfig, conditions?: OffloadConditions) {
    super(target, conditions)
    this._truncateConfig = config ?? {}
  }

  protected async _replaceTextBlock(block: TextBlock, tokens: number, message: Message): Promise<TextBlock | null> {
    logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | truncated text block`)
    return truncateTextBlock(block, this._truncateConfig)
  }

  protected async _replaceToolResultBlock(block: ToolResultBlock, tokens: number): Promise<ToolResultBlock | null> {
    logger.debug(`toolUseId=<${block.toolUseId}>, tokens=<${tokens}> | truncated tool result`)
    return truncateToolResultBlock(block, this._truncateConfig)
  }
}

// --- Summarize strategy ---

class OffloadSummarizeStrategy extends BaseOffloadStrategy {
  readonly name = 'offload:summarize'

  private readonly _config: SummarizeConfig
  private readonly _utilization: number | undefined
  private _model: Model | undefined

  constructor(target?: OffloadTarget, config?: SummarizeConfig, conditions?: OffloadConditions) {
    super(target, conditions)
    this._config = config ?? {}
    this._utilization = conditions?.utilization
  }

  protected override _shouldRegisterEagerHook(): boolean {
    return this._utilization === undefined
  }

  protected override _shouldApply(context: StrategyContext): boolean {
    if (this._utilization !== undefined && context.utilization < this._utilization) {
      logger.debug(
        `utilization=<${context.utilization}>, threshold=<${this._utilization}> | skipping summarization, below threshold`
      )
      return false
    }

    const model = this._resolveModel(context.agent)
    if (!model) {
      logger.warn('no model available for summarization')
      return false
    }
    this._model = model

    return context.messages.length > 0
  }

  override async apply(context: StrategyContext): Promise<boolean> {
    this._model = this._resolveModel(context.agent)
    return super.apply(context)
  }

  protected async _replaceTextBlock(block: TextBlock, tokens: number, message: Message): Promise<TextBlock | null> {
    if (!this._model) return null

    const summary = await summarizeText(block.text, this._model, this._config)
    if (!summary) return null

    logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | summarized text block`)
    return new TextBlock(`${SUMMARIZED_PREFIX} ~${tokens.toLocaleString()} tokens]\n\n${summary}`)
  }

  protected async _replaceToolResultBlock(block: ToolResultBlock, tokens: number): Promise<ToolResultBlock | null> {
    if (!this._model) return null

    const fullText = extractBlockText(block)
    const summary = await summarizeText(fullText, this._model, this._config)
    if (!summary) return null

    logger.debug(`toolUseId=<${block.toolUseId}>, tokens=<${tokens}> | summarized tool result`)
    return new ToolResultBlock({
      toolUseId: block.toolUseId,
      status: block.status,
      content: [new TextBlock(`${SUMMARIZED_PREFIX} ~${tokens.toLocaleString()} tokens]\n\n${summary}`)],
    })
  }

  private _resolveModel(agent: LocalAgent): Model | undefined {
    return this._config.model ?? agent.model
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

/** Wraps a strategy instance as an OffloadStrategyBuilder with a `.when()` chain. */
function wrapAsBuilder(
  strategy: BaseOffloadStrategy,
  createWithConditions: (conditions: OffloadConditions) => BaseOffloadStrategy
): OffloadStrategyBuilder {
  return {
    get name(): string {
      return strategy.name
    },
    init: strategy.init.bind(strategy),
    apply: strategy.apply.bind(strategy),
    when(conditions: OffloadConditions): ContextStrategy {
      return createWithConditions(conditions)
    },
  }
}

/** Disambiguates whether the first argument is a config object or a target. */
function isConfigObject(value: unknown, configKeys: string[]): boolean {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const keys = Object.keys(value)
  if (keys.length === 0) return true
  return keys.some((key) => configKeys.includes(key))
}

/**
 * Offload strategy builder namespace.
 *
 * - `Offload(target)` — drop matching content from L0 entirely
 * - `Offload.truncate(target, config)` — replace with a preview
 * - `Offload.summarize(target, config)` — replace with LLM-generated summary
 */
interface OffloadNamespace {
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
  return wrapAsBuilder(new OffloadDropStrategy(target), (c) => new OffloadDropStrategy(target, c))
}

offloadFn.truncate = function truncate(
  targetOrConfig?: OffloadTarget | TruncateConfig,
  config?: TruncateConfig
): OffloadStrategyBuilder {
  let target: OffloadTarget | undefined
  let truncateConfig: TruncateConfig | undefined

  if (targetOrConfig === undefined) {
    truncateConfig = config
  } else if (isConfigObject(targetOrConfig, ['previewTokens', 'preview'])) {
    truncateConfig = targetOrConfig as TruncateConfig
  } else {
    target = targetOrConfig as OffloadTarget
    truncateConfig = config
  }

  return wrapAsBuilder(
    new OffloadTruncateStrategy(target, truncateConfig),
    (c) => new OffloadTruncateStrategy(target, truncateConfig, c)
  )
}

offloadFn.summarize = function summarize(
  targetOrConfig?: OffloadTarget | SummarizeConfig,
  config?: SummarizeConfig
): OffloadStrategyBuilder {
  let target: OffloadTarget | undefined
  let summarizeConfig: SummarizeConfig | undefined

  if (targetOrConfig === undefined) {
    summarizeConfig = config
  } else if (isConfigObject(targetOrConfig, ['model', 'systemPrompt'])) {
    summarizeConfig = targetOrConfig as SummarizeConfig
  } else {
    target = targetOrConfig as OffloadTarget
    summarizeConfig = config
  }

  return wrapAsBuilder(
    new OffloadSummarizeStrategy(target, summarizeConfig),
    (c) => new OffloadSummarizeStrategy(target, summarizeConfig, c)
  )
}

/**
 * Builder for offload strategies — reduces content in L0.
 *
 * @example
 * ```typescript
 * Offload.truncate("toolResults", { previewTokens: 750 }).when({ threshold: 1500 })
 * Offload.summarize().when({ utilization: 0.85, preserveRecent: 2 })
 * Offload("toolResultErrors").when({ threshold: 500 })
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
