/**
 * ContextManager: first-class agent component for strategy-driven context management.
 *
 * On overflow, runs strategies then truncates if utilization still exceeds the context window.
 */

import type { Plugin } from '../plugins/plugin.js'
import type { LocalAgent } from '../types/agent.js'
import type { Message } from '../types/messages.js'
import { AfterModelCallEvent, BeforeModelCallEvent } from '../hooks/events.js'
import { ContextWindowOverflowError } from '../errors.js'
import { logger } from '../logging/logger.js'
import type { ContextManagerConfig, ContextStrategy, ContextState } from './types.js'
import { Offload } from './strategies/offload.js'

/**
 * Manages context reduction for an agent's conversation.
 *
 * On context overflow, runs the strategy pipeline (offload, summarize) then
 * truncates the oldest messages if utilization still exceeds the context window.
 *
 * The ContextManager is a first-class agent component — pass it via the
 * `contextManager` parameter on the Agent constructor. When present, it owns
 * overflow recovery — no separate ConversationManager is needed.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { ContextManager } from '@strands-agents/sdk/context-manager'
 *
 * const agent = new Agent({
 *   model,
 *   contextManager: new ContextManager(),
 * })
 * ```
 */
export class ContextManager implements Plugin {
  readonly name = 'strands:context-manager'

  private readonly _strategies: ContextStrategy[]

  private _agent: LocalAgent | undefined
  private _agentId: string | undefined

  constructor(config?: ContextManagerConfig) {
    this._strategies = config?.strategies ?? [
      Offload.truncate('toolResults').when({ threshold: 2500 }),
      Offload.summarize('*').when({ threshold: 1000, utilization: 0.85 }),
    ]
  }

  initAgent(agent: LocalAgent): void {
    if (this._agent && this._agent !== agent) {
      throw new Error('ContextManager instance cannot be shared across multiple agents')
    }
    this._agent = agent
    this._agentId = agent.id

    for (const strategy of this._strategies) {
      strategy.init?.(agent)
    }

    agent.addHook(BeforeModelCallEvent, async (event) => {
      try {
        await this._runStrategies(event.projectedInputTokens)
      } catch (error) {
        logger.warn(`agentId=<${this._agentId}>, error=<${error}> | proactive strategy pipeline failed`)
      }
    })

    let overflowRetries = 0
    agent.addHook(AfterModelCallEvent, async (event) => {
      if (!(event.error instanceof ContextWindowOverflowError)) {
        overflowRetries = 0
        return
      }

      if (overflowRetries >= 3) {
        logger.warn(`agentId=<${this._agentId}> | overflow retry limit reached, giving up`)
        overflowRetries = 0
        return
      }

      try {
        await this._runStrategies()
      } catch (strategyError) {
        logger.warn(
          `agentId=<${this._agentId}>, error=<${strategyError}> | strategy pipeline failed, falling through to truncate`
        )
      }

      const postTokens = await this._agent!.model.countTokens(agent.messages)
      const postUtilization = this._agent!.model.estimateUtilization(postTokens)
      if (postUtilization >= 1.0) {
        try {
          this._truncate(agent.messages)
        } catch (truncateError) {
          logger.warn(`agentId=<${this._agentId}>, error=<${truncateError}> | truncation failed`)
        }
      }

      overflowRetries++
      event.retry = true
    })
  }

  private async _runStrategies(precomputedInputTokens?: number): Promise<void> {
    if (!this._agent) return

    const messages = this._agent.messages
    const inputTokens = precomputedInputTokens ?? (await this._agent.model.countTokens(messages))
    const utilization = this._agent.model.estimateUtilization(inputTokens)

    const strategyContext: ContextState = {
      messages,
      agent: this._agent,
      utilization,
    }

    for (const strategy of this._strategies) {
      const acted = await strategy.apply(strategyContext)
      if (acted) {
        logger.debug(`strategy=<${strategy.name}>, agentId=<${this._agentId}> | strategy applied`)
      }
    }
  }

  /**
   * Unconditional truncation: drop the oldest messages (preserving the first message
   * and respecting tool-use/tool-result pair boundaries).
   */
  private _truncate(messages: Message[]): void {
    if (messages.length <= 3) return

    const startIndex = this._findSafeStartIndex(messages)
    if (startIndex >= messages.length - 1) return

    const targetRemoval = Math.max(2, Math.floor(messages.length * 0.2))
    const targetSplitIndex = Math.min(startIndex + targetRemoval, messages.length - 1)

    const validSplitIndex = findValidTrimPoint(messages, targetSplitIndex)
    const splitIndex = validSplitIndex < messages.length ? validSplitIndex : targetSplitIndex

    const removeCount = splitIndex - startIndex
    if (removeCount <= 0) return

    messages.splice(startIndex, removeCount)
    logger.debug(`agentId=<${this._agentId}>, removed=<${removeCount}> | truncated oldest messages on overflow`)
  }

  /**
   * Finds a safe start index for truncation that doesn't orphan tool-use/tool-result pairs.
   * Walks forward from index 1, skipping any message whose tool results pair with a tool-use
   * in the preceding preserved messages.
   */
  private _findSafeStartIndex(messages: Message[]): number {
    let startIndex = 2

    while (startIndex < messages.length - 1) {
      if (!this._messageHasToolResultPairedWithPreceding(messages, startIndex)) break
      startIndex++
    }

    return startIndex
  }

  private _messageHasToolResultPairedWithPreceding(messages: Message[], index: number): boolean {
    const message = messages[index]!
    const toolResultIds = new Set<string>()
    for (const block of message.content) {
      if (block.type === 'toolResultBlock' && 'toolUseId' in block) {
        toolResultIds.add((block as { toolUseId: string }).toolUseId)
      }
    }
    if (toolResultIds.size === 0) return false

    for (let preceding = 0; preceding < index; preceding++) {
      for (const block of messages[preceding]!.content) {
        if ('toolUseId' in block && 'name' in block) {
          if (toolResultIds.has((block as { toolUseId: string }).toolUseId)) return true
        }
      }
    }
    return false
  }
}

// --- Truncation helpers (inlined to avoid depending on conversation-manager) ---

/**
 * Finds a valid trim point: a user message that isn't an orphaned tool result.
 * Returns `messages.length` if no valid point exists.
 */
function findValidTrimPoint(messages: Message[], startIndex: number): number {
  let trimIndex = startIndex

  while (trimIndex < messages.length) {
    const message = messages[trimIndex]
    if (!message) break

    if (message.role !== 'user') {
      trimIndex++
      continue
    }

    const hasToolResult = message.content.some((block) => block.type === 'toolResultBlock')
    if (hasToolResult) {
      trimIndex++
      continue
    }

    const hasToolUse = message.content.some((block) => block.type === 'toolUseBlock')
    if (hasToolUse) {
      const nextMessage = messages[trimIndex + 1]
      const nextHasToolResult = nextMessage?.content.some((block) => block.type === 'toolResultBlock')
      if (!nextHasToolResult) {
        trimIndex++
        continue
      }
    }

    break
  }

  return trimIndex
}
