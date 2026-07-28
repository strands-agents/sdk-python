/**
 * ContextManager: first-class agent component for durable context management.
 *
 * Writes every message to L1 (the stash) on arrival.
 * L0 (agent.messages) is the compressed working set; L1 is the source of truth.
 * On overflow, always truncates to guarantee the agent can continue.
 */

import type { Plugin } from '../plugins/plugin.js'
import type { LocalAgent } from '../types/agent.js'
import type { Message } from '../types/messages.js'
import { InMemoryStorage } from '../storage/in-memory-storage.js'
import { AfterModelCallEvent, MessageAddedEvent } from '../hooks/events.js'
import { ContextWindowOverflowError } from '../errors.js'
import { logger } from '../logging/logger.js'
import { ToolResultBlock } from '../types/messages.js'
import { adjustSplitPointForToolPairs } from '../conversation-manager/compression/context-compression.js'
import { Stash } from './stash.js'
import type {
  ContextManagerConfig,
  ContextStrategy,
  MessageCategory,
  StashConfig,
  StrategyContext,
  StrategyInitContext,
} from './types.js'
import { TruncateMethod } from './strategies/methods/truncate-method.js'
import { SummarizeMethod } from './strategies/methods/summarize-method.js'

/**
 * Manages the L1 stash and context reduction for an agent's conversation.
 *
 * Every message is written to the stash on arrival via the `MessageAddedEvent` hook.
 * L0 (`agent.messages`) remains the in-memory working set that the model sees;
 * the stash holds the full, uncompressed history and serves as the source of truth
 * for later retrieval and restore operations.
 *
 * On context overflow, always truncates the oldest messages to guarantee the agent
 * can continue. Strategies (offload, summarize) run first for smarter compression,
 * but truncation is the unconditional safety net.
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

  private readonly _storage: InMemoryStorage
  private readonly _stashEnabled: boolean
  private readonly _stashFilter: MessageCategory[] | undefined
  private readonly _stashFilterMode: 'include' | 'exclude' | undefined
  private readonly _strategies: ContextStrategy[]
  private readonly _defaultStrategies: ContextStrategy[]

  private _stash: Stash | undefined
  private _agent: LocalAgent | undefined
  private _agentId: string | undefined

  constructor(config?: ContextManagerConfig) {
    this._storage = new InMemoryStorage()
    this._stashEnabled = resolveStashEnabled(config?.stash)
    this._strategies = config?.strategies ?? []
    this._defaultStrategies = [new TruncateMethod('toolResults'), new SummarizeMethod()]

    const stashConfig = typeof config?.stash === 'object' ? config.stash : undefined
    if (stashConfig?.include && stashConfig?.exclude) {
      throw new Error('StashConfig: include and exclude are mutually exclusive')
    }

    if (stashConfig?.include) {
      const raw = Array.isArray(stashConfig.include) ? stashConfig.include : [stashConfig.include]
      this._stashFilter = expandCategories(raw)
      this._stashFilterMode = 'include'
    } else if (stashConfig?.exclude) {
      const raw = Array.isArray(stashConfig.exclude) ? stashConfig.exclude : [stashConfig.exclude]
      this._stashFilter = expandCategories(raw)
      this._stashFilterMode = 'exclude'
    } else {
      this._stashFilter = undefined
      this._stashFilterMode = undefined
    }
  }

  /** Whether L1 writes are active. */
  get stashEnabled(): boolean {
    return this._stashEnabled
  }

  /** The L1 stash writer, or undefined if stash is disabled. */
  get stash(): Stash | undefined {
    return this._stash
  }

  initAgent(agent: LocalAgent): void {
    this._agent = agent
    this._agentId = agent.id

    if (this._stashEnabled) {
      this._stash = new Stash(this._storage)
    }

    if (!this._stashEnabled) {
      logger.info(`agentId=<${this._agentId}> | L1 stash disabled, offload operations will be destructive`)
    }

    const initContext: StrategyInitContext = {
      agent,
      storage: this._storage,
    }
    const strategies = this._strategies.length > 0 ? this._strategies : this._defaultStrategies
    for (const strategy of strategies) {
      strategy.init?.(initContext)
    }

    agent.addHook(MessageAddedEvent, (event) => this._onMessageAdded(event))

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
   * Called proactively when utilization exceeds a threshold. Strategies are
   * applied in order; each decides whether to act.
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
      storage: this._storage,
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

    const safeSplitPoint = adjustSplitPointForToolPairs(messages, Math.min(targetRemoval + 1, messages.length - 2))
    const removeCount = Math.max(1, safeSplitPoint - 1)

    messages.splice(1, removeCount)
    logger.debug(`agentId=<${this._agentId}>, removed=<${removeCount}> | truncated oldest messages on overflow`)
  }

  private async _onMessageAdded(event: MessageAddedEvent): Promise<void> {
    if (this._stash === undefined) return

    if (!this._shouldStash(event.message)) return

    await this._stash.writeMessage(event.message)
  }

  private _shouldStash(message: Message): boolean {
    if (!this._stashFilter) return true

    const categories = messageCategories(message)
    categories.add(message.role)

    if (this._stashFilterMode === 'include') {
      return this._stashFilter.some((category) => categories.has(category))
    }

    return !this._stashFilter.some((category) => categories.has(category))
  }
}

const MEDIA_TYPES = new Set(['imageBlock', 'videoBlock', 'documentBlock'])
const MEDIA_CATEGORIES: Set<MessageCategory> = new Set(['image', 'video', 'document'])

const BLOCK_TYPE_TO_CATEGORY: Record<string, MessageCategory> = {
  textBlock: 'text',
  toolUseBlock: 'toolUse',
  reasoningBlock: 'reasoning',
  imageBlock: 'image',
  videoBlock: 'video',
  documentBlock: 'document',
  citationsBlock: 'citations',
  cachePointBlock: 'cachePoint',
  guardContentBlock: 'guardContent',
}

function messageCategories(message: Message): Set<MessageCategory> {
  const categories = new Set<MessageCategory>()

  for (const block of message.content) {
    if (block instanceof ToolResultBlock) {
      categories.add(block.status === 'error' ? 'toolError' : 'toolResult')
    } else {
      const category = BLOCK_TYPE_TO_CATEGORY[block.type]
      if (category) {
        categories.add(category)
        if (MEDIA_TYPES.has(block.type)) {
          categories.add('media')
        }
      }
    }
  }

  return categories
}

function expandCategories(categories: MessageCategory[]): MessageCategory[] {
  if (!categories.includes('media')) return categories
  return [...new Set([...categories, ...MEDIA_CATEGORIES])]
}

function resolveStashEnabled(stash: StashConfig | boolean | undefined): boolean {
  if (stash === undefined || stash === true) return true
  if (stash === false) return false
  return stash.enabled ?? true
}
