/**
 * Summarize strategy — replaces oversized content with LLM-generated summaries.
 *
 * @internal
 */

import { logger } from '../../../logging/logger.js'
import { Message, TextBlock, ToolResultBlock } from '../../../types/messages.js'
import type { ContentBlock } from '../../../types/messages.js'
import type { Model } from '../../../models/model.js'
import type { LocalAgent } from '../../../types/agent.js'
import type { ContextStrategy, ContextState } from '../../types.js'
import {
  flattenMessagesToContent,
  SUMMARIZED_PREFIX,
  summarizeContent,
  toolResultToContentBlocks,
  type SummarizeConfig,
} from '../../methods/summarize.js'
import {
  BaseOffloadStrategy,
  collectRemovableWithPair,
  repairAlternation,
  type OffloadConditions,
  type OffloadTarget,
} from './base.js'

export class SummarizeStrategy extends BaseOffloadStrategy {
  readonly name = 'offload:summarize'

  private readonly _config: SummarizeConfig

  constructor(target?: OffloadTarget, config?: SummarizeConfig, conditions?: OffloadConditions) {
    super(target, conditions)
    this._config = config ?? {}
  }

  when(conditions: OffloadConditions): ContextStrategy {
    return new SummarizeStrategy(this._target, this._config, conditions)
  }

  override init(agent: LocalAgent): void {
    if (this._utilizationThreshold !== undefined) return
    super.init(agent)
  }

  override async apply(context: ContextState): Promise<boolean> {
    if (!this._resolveModel(context.agent)) {
      logger.warn('no model available for summarization')
      return false
    }
    if (context.messages.length === 0) return false
    return super.apply(context)
  }

  protected override async _applyPerMessage(context: ContextState): Promise<boolean> {
    const model = this._resolveModel(context.agent)
    if (!model) return false

    const { messages } = context
    if (messages.length <= 1) return false

    const eligible = await this._getEligibleMessages(context)
    if (eligible.length === 0) return false

    const summarizeCount = Math.max(1, Math.floor(eligible.length * this._removalRatio))
    const toSummarize = eligible.slice(0, summarizeCount)

    // Expand to include paired messages so we don't orphan tool pairs
    const safeSet = new Set<Message>()
    for (const message of toSummarize) {
      const index = messages.indexOf(message)
      if (index === -1) continue
      for (const removable of collectRemovableWithPair(messages, index)) {
        safeSet.add(removable)
      }
    }
    const safe = messages.filter((message) => safeSet.has(message))
    if (safe.length === 0) return false

    const contentBlocks = flattenMessagesToContent(safe)
    const summary = await summarizeContent(contentBlocks, model, this._config)
    if (!summary) return false

    const totalTokens = await model.countTokens(safe)
    const summaryMessage = new Message({
      role: 'user',
      content: [
        new TextBlock(
          `${SUMMARIZED_PREFIX} ${safe.length} messages, ~${totalTokens.toLocaleString()} tokens]\n\n${summary}`
        ),
      ],
    })

    // Remove summarized messages, tracking the lowest removal point for insertion
    let lowestIndex = messages.length
    for (const message of safe) {
      const index = messages.indexOf(message)
      if (index === -1) continue
      if (index < lowestIndex) lowestIndex = index
      messages.splice(index, 1)
    }

    const insertIndex = Math.max(1, Math.min(lowestIndex, messages.length))
    messages.splice(insertIndex, 0, summaryMessage)

    repairAlternation(messages)
    logger.debug(`summarized=<${safe.length}>, tokens=<${totalTokens}> | batched summarization complete`)
    return true
  }

  protected async _replaceBlock(
    block: TextBlock | ToolResultBlock,
    tokens: number,
    message: Message,
    agent: LocalAgent
  ): Promise<ContentBlock | null> {
    const model = this._resolveModel(agent)
    if (!model) return null

    if (block instanceof ToolResultBlock) {
      const summary = await summarizeContent(toolResultToContentBlocks(block.content), model, this._config)
      if (!summary) return null

      logger.debug(`toolUseId=<${block.toolUseId}>, tokens=<${tokens}> | summarized tool result`)
      return new ToolResultBlock({
        toolUseId: block.toolUseId,
        status: block.status,
        content: [new TextBlock(`${SUMMARIZED_PREFIX} ~${tokens.toLocaleString()} tokens]\n\n${summary}`)],
      })
    }

    const summary = await summarizeContent([new TextBlock(block.text)], model, this._config)
    if (!summary) return null

    logger.debug(`trackingId=<${message.trackingId}>, tokens=<${tokens}> | summarized text block`)
    return new TextBlock(`${SUMMARIZED_PREFIX} ~${tokens.toLocaleString()} tokens]\n\n${summary}`)
  }

  private _resolveModel(agent: LocalAgent): Model | undefined {
    return this._config.model ?? agent.model
  }
}
