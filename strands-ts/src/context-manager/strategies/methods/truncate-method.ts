/**
 * Truncate method: stores oversized tool results to in-memory storage and replaces
 * them with a head-tail preview in the message array.
 *
 * Operates eagerly: offloads on message arrival so the model never sees
 * oversized results. The retroactive `apply()` catches anything that
 * slipped through (e.g., messages added before the strategy was initialized).
 *
 * @internal
 */

import { logger } from '../../../logging/logger.js'
import { MessageAddedEvent } from '../../../hooks/events.js'
import { TextBlock, ToolResultBlock } from '../../../types/messages.js'
import type { Message } from '../../../types/messages.js'
import type { ContextStrategy, StrategyContext, StrategyInitContext } from '../../types.js'
import type { OffloadTarget, OffloadWhenConditions } from '../offload.js'

const DEFAULT_THRESHOLD = 2500
const DEFAULT_PREVIEW_TOKENS = 1000
const DEFAULT_SKIP_RECENT = 3
const CHARS_PER_TOKEN = 4

/**
 * Configuration for the truncate method.
 */
export interface TruncateMethodConfig {
  /** Token threshold above which tool results are offloaded. Defaults to 2,500. */
  threshold?: number

  /** Number of tokens to keep as preview text. Defaults to 1,000. */
  previewTokens?: number
}

/**
 * A context reduction method that offloads oversized tool results into storage,
 * replacing them with a head-tail preview and a storage reference.
 *
 * Not instantiated directly — use the `Offload.truncate()` builder.
 *
 * @internal
 */
export class TruncateMethod implements ContextStrategy {
  readonly name = 'offload:truncate'

  private readonly _threshold: number
  private readonly _previewTokens: number
  private readonly _skipRecent: number
  private readonly _target: OffloadTarget
  private readonly _toolFilter: Set<string> | undefined
  private readonly _excludeFilter: Set<string> | undefined

  constructor(target: OffloadTarget, config?: TruncateMethodConfig, conditions?: OffloadWhenConditions) {
    this._threshold = conditions?.threshold ?? config?.threshold ?? DEFAULT_THRESHOLD
    this._previewTokens = config?.previewTokens ?? DEFAULT_PREVIEW_TOKENS
    this._skipRecent = conditions?.skipRecent ?? DEFAULT_SKIP_RECENT
    this._target = target

    const resolved = resolveToolFilter(target)
    this._toolFilter = resolved.include
    this._excludeFilter = resolved.exclude
  }

  init(context: StrategyInitContext): void {
    const { agent, storage } = context
    agent.addHook(MessageAddedEvent, async (event) => {
      const message = event.message
      if (message.role !== 'user') return

      for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
        const block = message.content[blockIndex]!
        if (!(block instanceof ToolResultBlock)) continue
        if (!this._matchesTarget(block, message)) continue

        const estimatedTokens = this._estimateTokens(block)
        if (estimatedTokens <= this._threshold) continue

        const reference = await this._storeAndReplace(message, blockIndex, block, storage)
        if (reference) {
          logger.debug(
            `toolUseId=<${block.toolUseId}>, tokens=<${estimatedTokens}> | eagerly offloaded tool result to storage`
          )
        }
      }
    })
  }

  async apply(context: StrategyContext): Promise<boolean> {
    const { messages, storage } = context
    const eligible = messages.slice(0, Math.max(0, messages.length - this._skipRecent))

    let offloaded = false

    for (const message of eligible) {
      if (message.role !== 'user') continue

      for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
        const block = message.content[blockIndex]!
        if (!(block instanceof ToolResultBlock)) continue
        if (!this._matchesTarget(block, message)) continue
        if (this._isAlreadyOffloaded(block)) continue

        const estimatedTokens = this._estimateTokens(block)
        if (estimatedTokens <= this._threshold) continue

        const reference = await this._storeAndReplace(message, blockIndex, block, storage)
        if (reference) {
          offloaded = true
          logger.debug(
            `toolUseId=<${block.toolUseId}>, tokens=<${estimatedTokens}> | offloaded tool result to storage`
          )
        }
      }
    }

    return offloaded
  }

  private _matchesTarget(block: ToolResultBlock, message: Message): boolean {
    if (this._target === 'toolResults') {
      return block.status !== 'error'
    }
    if (this._target === 'toolResultErrors') {
      return block.status === 'error'
    }

    const toolName = this._resolveToolName(block, message)
    if (!toolName) return this._toolFilter === undefined && this._excludeFilter === undefined

    if (this._excludeFilter) {
      return !this._excludeFilter.has(toolName)
    }
    if (this._toolFilter) {
      return this._toolFilter.has(toolName)
    }

    return true
  }

  private _resolveToolName(block: ToolResultBlock, message: Message): string | undefined {
    for (const content of message.content) {
      if ('toolUseId' in content && 'name' in content && (content as { toolUseId: string }).toolUseId === block.toolUseId) {
        return (content as { name: string }).name
      }
    }
    return undefined
  }

  private _isAlreadyOffloaded(block: ToolResultBlock): boolean {
    if (block.content.length === 1 && block.content[0] instanceof TextBlock) {
      return block.content[0].text.startsWith('[Offloaded:')
    }
    return false
  }

  private _estimateTokens(block: ToolResultBlock): number {
    let chars = 0
    for (const content of block.content) {
      if (content instanceof TextBlock) {
        chars += content.text.length
      } else {
        chars += JSON.stringify(content.toJSON()).length
      }
    }
    return Math.ceil(chars / CHARS_PER_TOKEN)
  }

  private async _storeAndReplace(
    message: Message,
    blockIndex: number,
    block: ToolResultBlock,
    storage: import('../../../storage/storage.js').Storage
  ): Promise<string | null> {
    const fullText = this._extractText(block)
    const storageKey = `offload/${block.toolUseId}`

    try {
      await storage.write(storageKey, new TextEncoder().encode(fullText))
    } catch {
      logger.warn(`toolUseId=<${block.toolUseId}> | failed to store offloaded content`)
      return null
    }

    const preview = this._buildPreview(fullText, block, storageKey)
    const replacement = new ToolResultBlock({
      toolUseId: block.toolUseId,
      status: block.status,
      content: [new TextBlock(preview)],
    })

    ;(message.content as unknown[])[blockIndex] = replacement
    return storageKey
  }

  private _extractText(block: ToolResultBlock): string {
    const parts: string[] = []
    for (const content of block.content) {
      if (content instanceof TextBlock) {
        parts.push(content.text)
      } else {
        parts.push(JSON.stringify(content.toJSON(), null, 2))
      }
    }
    return parts.join('\n')
  }

  private _buildPreview(fullText: string, block: ToolResultBlock, reference: string): string {
    const previewChars = this._previewTokens * CHARS_PER_TOKEN
    const totalChars = fullText.length

    let preview: string
    if (totalChars <= previewChars) {
      preview = fullText
    } else {
      const headChars = Math.floor(previewChars * 0.6)
      const tailChars = previewChars - headChars
      const head = fullText.slice(0, headChars)
      const tail = fullText.slice(-tailChars)
      const elided = totalChars - headChars - tailChars
      preview = `${head}\n\n[... ${elided.toLocaleString()} chars elided ...]\n\n${tail}`
    }

    return (
      `[Offloaded: ${block.content.length} blocks, ~${Math.ceil(totalChars / CHARS_PER_TOKEN).toLocaleString()} tokens]\n` +
      `Full content available at storage reference "${reference}".\n\n` +
      preview
    )
  }
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
