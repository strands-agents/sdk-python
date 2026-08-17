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
import { EmergencyTruncateStrategy, Offload } from './strategies/offload.js'

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
    for (const strategy of this._strategies) {
      strategy.init?.(agent)
    }

    agent.addHook(BeforeModelCallEvent, async (event) => {
      try {
        await this._runStrategies(event.agent, event.projectedInputTokens)
      } catch (error) {
        logger.warn(`agentId=<${event.agent.id}>, error=<${error}> | proactive strategy pipeline failed`)
      }
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

      try {
        await this._runStrategies(event.agent)
      } catch (strategyError) {
        logger.warn(`agentId=<${event.agent.id}>, error=<${strategyError}> | strategy pipeline failed`)
        return
      }

      overflowRetries++
      event.retry = true
    })
  }

  private async _runStrategies(agent: LocalAgent, precomputedInputTokens?: number): Promise<void> {
    const messages = agent.messages
    const inputTokens = precomputedInputTokens ?? (await agent.model.countTokens(messages))
    const utilization = agent.model.estimateUtilization(inputTokens)

    const strategyContext: ContextState = {
      messages,
      agent,
      utilization,
    }

    for (const strategy of this._strategies) {
      const acted = await strategy.apply(strategyContext)
      if (acted) {
        logger.debug(`strategy=<${strategy.name}>, agentId=<${agent.id}> | strategy applied`)
      }
    }
  }
}
