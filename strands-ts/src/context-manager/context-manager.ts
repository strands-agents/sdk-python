/**
 * ContextManager: first-class agent component for strategy-driven context management.
 *
 * On overflow, runs the strategy pipeline (including an emergency truncation as the final step).
 */

import type { Plugin } from '../plugins/plugin.js'
import type { Tool } from '../tools/tool.js'
import type { LocalAgent } from '../types/agent.js'
import { AfterModelCallEvent, BeforeModelCallEvent, MessageAddedEvent } from '../hooks/events.js'
import { ContextWindowOverflowError } from '../errors.js'
import { InMemoryStorage } from '../storage/in-memory-storage.js'
import type { Storage } from '../storage/storage.js'
import { logger } from '../logging/logger.js'
import type { ContextManagerConfig, ContextStrategy, ContextState } from './types.js'
import { EmergencyTruncateStrategy, Offload } from './strategies/offload/index.js'
import { Stash } from './stash.js'
import { createRetrievalTool, trackRetrievalToolUseIds } from './retrieval-tool.js'

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
  private readonly _stashStorage: Storage | false
  private readonly _enableRetrievalTool: boolean
  private readonly _retrievalToolUseIds = new Set<string>()
  private _stash: Stash | undefined
  private _retrievalTool: Tool | undefined

  constructor(config?: ContextManagerConfig) {
    this._strategies = [
      ...(config?.strategies ?? [
        Offload.truncate('toolResults').when({ threshold: 2500 }),
        Offload.summarize('*').when({ threshold: 1000, utilization: 0.85 }),
      ]),
      new EmergencyTruncateStrategy(),
    ]
    this._stashStorage = config?.stash === false ? false : (config?.stash?.storage ?? new InMemoryStorage())
    this._enableRetrievalTool = config?.stash !== false && config?.stash?.retrievalTool !== false
  }

  getTools(): Tool[] {
    if (!this._enableRetrievalTool) return []
    if (!this._stash) return []
    if (!this._retrievalTool) {
      this._retrievalTool = createRetrievalTool(this._stash)
    }
    return [this._retrievalTool]
  }

  initAgent(agent: LocalAgent): void {
    if (this._stashStorage !== false) {
      this._stash = new Stash(this._stashStorage, agent.sessionId)
    }

    if (this._stash) {
      const stash = this._stash
      const skipSet = this._retrievalToolUseIds
      agent.addHook(MessageAddedEvent, async (event) => {
        trackRetrievalToolUseIds(event.message, skipSet)
        await stash.storeMessage(event.message, skipSet)
      })
    }

    for (const strategy of this._strategies) {
      strategy.init?.(agent, this._stash)
    }

    agent.addHook(BeforeModelCallEvent, async (event) => {
      await this._runStrategies(event.agent, event.projectedInputTokens)
    })

    // Assumes sequential invocations on this agent (no concurrent calls)
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

    const strategyContext: ContextState = {
      messages,
      agent,
      utilization: agent.model.estimateUtilization(inputTokens),
      ...(this._stash ? { stash: this._stash } : {}),
    }

    let anyActed = false
    for (const strategy of this._strategies) {
      try {
        const acted = await strategy.apply(strategyContext)
        if (acted) {
          anyActed = true
          strategyContext.utilization = agent.model.estimateUtilization(await agent.model.countTokens(messages))
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
