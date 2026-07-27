/**
 * Offload pass: stores oversized tool results to the stash and replaces
 * them with a head-tail preview + storage reference.
 *
 * Operates eagerly by default: offloads on message arrival so the model never
 * sees oversized results. The retroactive `apply()` pass catches anything that
 * slipped through (e.g., messages added before the pass was initialized).
 *
 * @internal
 */

import { logger } from '../../logging/logger.js'
import { MessageAddedEvent } from '../../hooks/events.js'
import { TextBlock, ToolResultBlock } from '../../types/messages.js'
import type { Message } from '../../types/messages.js'
import type { ContextPass, PassContext, PassInitContext } from '../types.js'

const DEFAULT_MAX_RESULT_TOKENS = 2500
const DEFAULT_PREVIEW_TOKENS = 1000
const DEFAULT_SKIP_RECENT = 3
const CHARS_PER_TOKEN = 4

/**
 * Configuration for the offload pass.
 */
export interface OffloadStrategyConfig {
  /** Token threshold above which tool results are offloaded. Defaults to 2,500. */
  maxResultTokens?: number

  /** Number of tokens to keep as preview text. Defaults to 1,000. */
  previewTokens?: number

  /** Number of recent messages to skip during retroactive apply(). Defaults to 3. */
  skipRecent?: number
}

/**
 * Offloads oversized tool results from L0 into storage, replacing them with a
 * head-tail preview and a stash reference.
 *
 * Eagerly offloads on message arrival (the model never sees the full content).
 * The retroactive `apply()` pass catches results that existed before initialization.
 */
export class OffloadStrategy implements ContextPass {
  readonly name = 'offload'

  private readonly _maxResultTokens: number
  private readonly _previewTokens: number
  private readonly _skipRecent: number

  constructor(config?: OffloadStrategyConfig) {
    this._maxResultTokens = config?.maxResultTokens ?? DEFAULT_MAX_RESULT_TOKENS
    this._previewTokens = config?.previewTokens ?? DEFAULT_PREVIEW_TOKENS
    this._skipRecent = config?.skipRecent ?? DEFAULT_SKIP_RECENT
  }

  init(context: PassInitContext): void {
    const { agent, storage } = context
    agent.addHook(MessageAddedEvent, async (event) => {
      const message = event.message
      if (message.role !== 'user') return

      for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
        const block = message.content[blockIndex]!
        if (!(block instanceof ToolResultBlock)) continue
        if (block.status === 'error') continue

        const estimatedTokens = this._estimateTokens(block)
        if (estimatedTokens <= this._maxResultTokens) continue

        const reference = await this._storeAndReplace(message, blockIndex, block, storage)
        if (reference) {
          logger.debug(
            `toolUseId=<${block.toolUseId}>, tokens=<${estimatedTokens}> | eagerly offloaded tool result to stash`
          )
        }
      }
    })
  }

  async apply(context: PassContext): Promise<boolean> {
    const { messages, storage } = context
    const eligible = messages.slice(0, Math.max(0, messages.length - this._skipRecent))

    let offloaded = false

    for (const message of eligible) {
      if (message.role !== 'user') continue

      for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
        const block = message.content[blockIndex]!
        if (!(block instanceof ToolResultBlock)) continue
        if (block.status === 'error') continue
        if (this._isAlreadyOffloaded(block)) continue

        const estimatedTokens = this._estimateTokens(block)
        if (estimatedTokens <= this._maxResultTokens) continue

        const reference = await this._storeAndReplace(message, blockIndex, block, storage)
        if (reference) {
          offloaded = true
          logger.debug(
            `toolUseId=<${block.toolUseId}>, tokens=<${estimatedTokens}> | offloaded tool result to stash`
          )
        }
      }
    }

    return offloaded
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
    storage: import('../../storage/storage.js').Storage
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
