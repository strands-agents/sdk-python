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
import type {
  ContextManagerConfig,
  ContextStrategy,
  StrategyContext,
  StrategyInitContext,
} from './types.js'
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
    this._agent = agent
    this._agentId = agent.id

    const initContext: StrategyInitContext = {
      agent,
    }
    const strategies = this._strategies.length > 0 ? this._strategies : this._defaultStrategies
    for (const strategy of strategies) {
      strategy.init?.(initContext)
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

      await this._runStrategies()
      this._truncate(agent.messages)

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

    const strategyContext: StrategyContext = {
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
    const targetRemoval = Math.max(1, Math.floor(messages.length * 0.2))
    if (messages.length <= 3) return

    let safeSplitPoint: number
    try {
      safeSplitPoint = adjustSplitPointForToolPairs(messages, Math.min(targetRemoval + 1, messages.length - 2))
    } catch {
      safeSplitPoint = Math.min(targetRemoval + 1, messages.length - 2)
    }
    const removeCount = Math.max(1, safeSplitPoint - 1)

    messages.splice(1, removeCount)
    logger.debug(`agentId=<${this._agentId}>, removed=<${removeCount}> | truncated oldest messages on overflow`)
  }
}
