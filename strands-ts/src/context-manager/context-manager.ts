/**
 * ContextManager: first-class agent component for strategy-driven context management.
 *
 * On overflow, runs strategies then always truncates to guarantee the agent can continue.
 */

import type { Plugin } from '../plugins/plugin.js'
import type { LocalAgent } from '../types/agent.js'
import type { Message } from '../types/messages.js'
import { AfterModelCallEvent } from '../hooks/events.js'
import { ContextWindowOverflowError } from '../errors.js'
import { logger } from '../logging/logger.js'
import { adjustSplitPointForToolPairs } from '../conversation-manager/compression/context-compression.js'
import type { ContextManagerConfig, ContextStrategy, ContextState } from './types.js'
import { Offload } from './strategies/offload.js'

/**
 * Manages context reduction for an agent's conversation.
 *
 * On context overflow, runs the strategy pipeline (offload, summarize) then
 * unconditionally truncates the oldest messages as a safety net.
 *
 * The ContextManager is a first-class agent component — pass it via the
 * `contextManager` parameter on the Agent constructor. When present, it owns
 * overflow recovery — no separate ConversationManager is needed.
 *
 * @example
 * ```typescript
 * import { Agent, ContextManager } from '@strands-agents/sdk'
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
  private readonly _defaultStrategies: ContextStrategy[]

  private _agent: LocalAgent | undefined
  private _agentId: string | undefined

  constructor(config?: ContextManagerConfig) {
    this._strategies = config?.strategies ?? []
    this._defaultStrategies = [
      Offload.truncate('toolResults').when({ threshold: 2500 }),
      Offload.summarize('toolResults').when({ threshold: 2500, utilization: 0.85 }),
    ]
  }

  initAgent(agent: LocalAgent): void {
    if (this._agent && this._agent !== agent) {
      throw new Error('ContextManager instance cannot be shared across multiple agents')
    }
    this._agent = agent
    this._agentId = agent.id

    const strategies = this._strategies.length > 0 ? this._strategies : this._defaultStrategies
    for (const strategy of strategies) {
      strategy.init?.(agent)
    }

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

      try {
        this._truncate(agent.messages)
      } catch (truncateError) {
        logger.warn(`agentId=<${this._agentId}>, error=<${truncateError}> | truncation failed`)
      }

      overflowRetries++
      event.retry = true
    })
  }

  /**
   * Run the strategy pipeline to reduce context.
   *
   * Strategies are applied in order; each decides whether to act.
   */
  async apply(): Promise<void> {
    if (!this._agent) {
      throw new Error('ContextManager.apply() called before initAgent()')
    }

    await this._runStrategies()
  }

  private async _runStrategies(): Promise<void> {
    if (!this._agent) return

    const messages = this._agent.messages
    const utilization = await this._estimateUtilization()

    const strategyContext: ContextState = {
      messages,
      agent: this._agent,
      utilization,
    }

    const strategies = this._strategies.length > 0 ? this._strategies : this._defaultStrategies

    for (const strategy of strategies) {
      const acted = await strategy.apply(strategyContext)
      if (acted) {
        logger.debug(`strategy=<${strategy.name}>, agentId=<${this._agentId}> | strategy applied`)
      }
    }
  }

  private async _estimateUtilization(): Promise<number> {
    if (!this._agent) return 0
    const model = this._agent.model
    const config = model.getConfig()
    if (!config.contextWindowLimit) return 0

    const tokens = await model.countTokens(this._agent.messages)
    return tokens / config.contextWindowLimit
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

    let validSplitIndex: number
    try {
      validSplitIndex = adjustSplitPointForToolPairs(messages, targetSplitIndex)
    } catch {
      logger.warn(`agentId=<${this._agentId}> | no valid split point found, skipping truncation`)
      return
    }

    const removeCount = validSplitIndex - startIndex
    if (removeCount <= 0) return

    messages.splice(startIndex, removeCount)
    logger.debug(`agentId=<${this._agentId}>, removed=<${removeCount}> | truncated oldest messages on overflow`)
  }

  /**
   * Finds a safe start index for truncation that doesn't orphan tool-use/tool-result pairs
   * anchored in the preserved head messages.
   */
  private _findSafeStartIndex(messages: Message[]): number {
    let startIndex = 1
    const headMessage = messages[0]
    if (!headMessage) return startIndex

    const headHasToolUse = headMessage.content.some((block) => 'toolUseId' in block && 'name' in block)
    if (headHasToolUse) {
      // Skip past the tool result message that pairs with the head's tool use
      while (startIndex < messages.length) {
        const message = messages[startIndex]!
        const hasToolResult = message.content.some((block) => block.type === 'toolResultBlock')
        if (!hasToolResult) break
        startIndex++
      }
    }

    return startIndex
  }
}
