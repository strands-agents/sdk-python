/**
 * ContextManager: first-class agent component for strategy-driven context management.
 *
 * On overflow, runs the strategy pipeline (including an emergency truncation as the final step).
 */

import type { Plugin } from '../plugins/plugin.js'
import type { LocalAgent } from '../types/agent.js'
import { AfterModelCallEvent, BeforeModelCallEvent } from '../hooks/events.js'
import { ContextWindowOverflowError } from '../errors.js'
import { logger } from '../logging/logger.js'
import type { ContextManagerConfig, ContextStrategy, ContextState } from './types.js'
import { EmergencyTruncateStrategy, Offload } from './strategies/offload/index.js'

/**
 * Manages context reduction for an agent's conversation.
 *
 * On context overflow, runs the strategy pipeline (offload, summarize, emergency truncate).
 * The emergency truncation is always appended as the final strategy — it recomputes
 * utilization and only fires if the window is still overflowing after user strategies.
 *
 * The ContextManager is a first-class agent component — pass it via the
 * `contextManager` parameter on the Agent constructor. When present, it owns
 * overflow recovery — no separate ConversationManager is needed.
 *
 * @experimental
 * @internal
 */
export class ContextManager implements Plugin {
  readonly name = 'strands:context-manager'

  private static readonly _claimed = new WeakMap<LocalAgent, ContextManager>()

  private readonly _strategies: ContextStrategy[]

  constructor(config?: ContextManagerConfig) {
    this._strategies = [
      ...(config?.strategies ?? [
        Offload.truncate('toolResults').when({ threshold: 2500 }),
        Offload.summarize('*').when({ threshold: 1000, utilization: 0.85 }),
      ]),
      new EmergencyTruncateStrategy(),
    ]
  }

  initAgent(agent: LocalAgent): void {
    const existing = ContextManager._claimed.get(agent)
    if (existing && existing !== this) {
      logger.warn('duplicate ContextManager detected, this instance will be ignored')
      return
    }
    ContextManager._claimed.set(agent, this)

    for (const strategy of this._strategies) {
      strategy.init?.(agent)
    }

    agent.addHook(BeforeModelCallEvent, async (event) => {
      await this._runStrategies(event.agent, event.projectedInputTokens)
    })

    let overflowRetries = 0
    agent.addHook(AfterModelCallEvent, async (event) => {
      if (!(event.error instanceof ContextWindowOverflowError)) {
        overflowRetries = 0
        return
      }

      if (overflowRetries >= 3) {
        logger.warn(`agentId=<${event.agent.id}> | overflow retry limit reached, giving up`)
        overflowRetries = 0
        return
      }

      const acted = await this._runStrategies(event.agent)
      if (!acted) {
        logger.warn(`agentId=<${event.agent.id}> | no strategy made progress, skipping retry`)
        return
      }

      overflowRetries++
      event.retry = true
    })
  }

  private async _runStrategies(agent: LocalAgent, precomputedInputTokens?: number): Promise<boolean> {
    const messages = agent.messages
    const inputTokens = precomputedInputTokens ?? (await agent.model.countTokens(messages))
    const utilization = agent.model.estimateUtilization(inputTokens)

    const strategyContext: ContextState = {
      messages,
      agent,
      utilization,
    }

    let anyActed = false
    for (const strategy of this._strategies) {
      try {
        const acted = await strategy.apply(strategyContext)
        if (acted) {
          anyActed = true
          logger.debug(`strategy=<${strategy.name}>, agentId=<${agent.id}> | strategy applied`)
        }
      } catch (error) {
        logger.warn(
          `strategy=<${strategy.name}>, agentId=<${agent.id}>, error=<${error}> | strategy failed, continuing`
        )
      }
    }
    return anyActed
  }
}
