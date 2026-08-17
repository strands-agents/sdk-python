/**
 * Truncate strategy — replaces oversized content with a preview.
 *
 * @internal
 */

import { logger } from '../../../logging/logger.js'
import { Message, TextBlock, ToolResultBlock } from '../../../types/messages.js'
import type { ContentBlock } from '../../../types/messages.js'
import type { LocalAgent } from '../../../types/agent.js'
import type { ContextStrategy, ContextState } from '../../types.js'
import { truncateToolResultBlock, truncateTextBlock, type TruncateConfig } from '../methods/truncate.js'
import {
  BaseOffloadStrategy,
  spliceWithPairs,
  repairAlternation,
  type OffloadConditions,
  type OffloadTarget,
} from './base.js'

export class TruncateStrategy extends BaseOffloadStrategy {
  readonly name = 'offload:truncate'

  private readonly _truncateConfig: TruncateConfig

  constructor(target?: OffloadTarget, config?: TruncateConfig, conditions?: OffloadConditions) {
    super(target, conditions)
    this._truncateConfig = config ?? {}

    const previewTokens =
      typeof this._truncateConfig.previewTokens === 'number' && Number.isFinite(this._truncateConfig.previewTokens)
        ? this._truncateConfig.previewTokens
        : 1000
    if (
      conditions?.threshold !== undefined &&
      Number.isFinite(conditions.threshold) &&
      conditions.threshold <= previewTokens
    ) {
      throw new Error(
        `threshold (${conditions.threshold}) must be greater than previewTokens (${previewTokens}) to ensure truncation converges`
      )
    }
  }

  when(conditions: OffloadConditions): ContextStrategy {
    return new TruncateStrategy(this._target, this._truncateConfig, conditions)
  }

  protected override _makeRemovalMarker(count: number): string {
    return `[... ${count} ${count === 1 ? 'message' : 'messages'} elided ...]`
  }

  protected override async _applyPerMessage(context: ContextState): Promise<boolean> {
    const { messages } = context
    if (messages.length <= 1) return false

    const eligible = await this._getEligibleMessages(context)
    if (eligible.length === 0) return false

    // Determine head/tail split based on config (default: favor tail — 30% head, 70% tail)
    const previewMode = this._truncateConfig.preview ?? 'headTail'
    const headShare = { head: 1, tail: 0, headTail: 0.3 }[previewMode]
    const targetRemoval = Math.max(1, Math.floor(eligible.length * this._removalRatio))
    const keepCount = eligible.length - targetRemoval

    const headKeep = Math.floor(keepCount * headShare)
    const tailKeep = keepCount - headKeep

    // Split into head (keep), middle (remove), tail (keep)
    const headMessages = eligible.slice(0, headKeep)
    const middleMessages = eligible.slice(headKeep, eligible.length - (tailKeep || 0))

    if (middleMessages.length === 0) return false

    const insertIndex = headKeep > 0 ? messages.indexOf(headMessages[headKeep - 1]!) + 1 : 1
    const removed = spliceWithPairs(messages, middleMessages)
    if (removed === 0) return false

    const marker = this._makeRemovalMarker(removed)
    messages.splice(
      Math.min(insertIndex, messages.length),
      0,
      new Message({ role: 'user', content: [new TextBlock(marker)] })
    )

    repairAlternation(messages)
    return true
  }

  protected async _replaceBlock(
    block: TextBlock | ToolResultBlock,
    tokens: number,
    message: Message,
    _agent: LocalAgent
  ): Promise<ContentBlock | null> {
    if (block instanceof ToolResultBlock) {
      logger.debug(`toolUseId=<${block.toolUseId}>, tokens=<${tokens}> | truncated tool result`)
      return truncateToolResultBlock(block, this._truncateConfig)
    }
    logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | truncated text block`)
    return truncateTextBlock(block, this._truncateConfig)
  }
}
