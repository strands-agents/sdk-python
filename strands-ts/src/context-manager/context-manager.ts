/**
 * ContextManager: first-class agent component for durable context management.
 *
 * Writes every message to L1 (the stash) on arrival.
 * L0 (agent.messages) is the compressed working set; L1 is the source of truth.
 * Applies a strategy pipeline for proactive and reactive context reduction.
 */

import type { Plugin } from '../plugins/plugin.js'
import type { LocalAgent } from '../types/agent.js'
import type { Storage } from '../storage/storage.js'
import type { Message } from '../types/messages.js'
import { NAMESPACED, namespace } from '../storage/storage.js'
import { InMemoryStorage } from '../storage/in-memory-storage.js'
import { AfterModelCallEvent, MessageAddedEvent } from '../hooks/events.js'
import { ContextWindowOverflowError } from '../errors.js'
import { logger } from '../logging/logger.js'
import { Stash } from './stash.js'
import type { ContextManagerConfig, ContextStrategy, StashConfig, StrategyContext } from './types.js'
import { OffloadStrategy } from './strategies/offload-strategy.js'
import { SummarizeStrategy } from './strategies/summarize-strategy.js'

const STORAGE_PREFIX = 'context'

/**
 * Manages the L1 stash and context reduction for an agent's conversation.
 *
 * Every message is written to the stash on arrival via the `MessageAddedEvent` hook.
 * L0 (`agent.messages`) remains the in-memory working set that the model sees;
 * the stash holds the full, uncompressed history and serves as the source of truth
 * for later retrieval and restore operations.
 *
 * On context overflow (or proactive threshold), `apply()` runs the strategy pipeline.
 * When no strategies are configured, uses a default pipeline (offload → summarize).
 * An internal floor truncation guarantees `apply()` always reduces something.
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
 *   contextManager: new ContextManager({ storage }),
 * })
 * ```
 */
export class ContextManager implements Plugin {
  readonly name = 'strands:context-manager'

  private readonly _storage: Storage
  private readonly _stashEnabled: boolean
  private readonly _strategies: ContextStrategy[]

  private _stash: Stash | undefined
  private _agent: LocalAgent | undefined
  private _agentId: string | undefined
  private _sessionId: string | undefined

  constructor(config?: ContextManagerConfig) {
    this._storage = config?.storage ?? new InMemoryStorage()
    this._stashEnabled = resolveStashEnabled(config?.stash)
    this._strategies = config?.strategies ?? []
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
    this._sessionId = this._resolveSessionId(agent)
    this._stash = this._buildStash()

    if (!this._stashEnabled) {
      logger.info(`agentId=<${this._agentId}> | L1 stash disabled, offload operations will be destructive`)
    }

    agent.addHook(MessageAddedEvent, (event) => this._onMessageAdded(event))

    // Reactive: handle context overflow by applying strategies and retrying
    agent.addHook(AfterModelCallEvent, async (event) => {
      if (event.error instanceof ContextWindowOverflowError) {
        await this.apply()
        event.retry = true
      }
    })
  }

  /**
   * Run the strategy pipeline to reduce context.
   *
   * Called reactively on context overflow or proactively when utilization
   * exceeds a threshold. Strategies are applied in order; each decides
   * whether to act. If no strategy reduces anything, a floor truncation
   * drops the oldest non-pinned messages to guarantee progress.
   */
  async apply(): Promise<void> {
    if (!this._agent) {
      throw new Error('ContextManager.apply() called before initAgent()')
    }

    const messages = this._agent.messages
    const utilization = await this._estimateUtilization()

    const strategyContext: StrategyContext = {
      messages,
      agent: this._agent,
      utilization,
      storage: this._scopeStorage(),
    }

    const strategies = this._strategies.length > 0 ? this._strategies : this._defaultStrategies()
    let reduced = false

    for (const strategy of strategies) {
      const acted = await strategy.apply(strategyContext)
      if (acted) {
        reduced = true
        logger.debug(`strategy=<${strategy.name}>, agentId=<${this._agentId}> | strategy applied`)
      }
    }

    if (!reduced) {
      this._floorTruncate(messages)
      logger.warn(`agentId=<${this._agentId}> | floor truncation applied, no strategy reduced context`)
    }
  }

  private _defaultStrategies(): ContextStrategy[] {
    return [new OffloadStrategy(), new SummarizeStrategy()]
  }

  private async _estimateUtilization(): Promise<number> {
    if (!this._agent) return 0
    const model = (this._agent as unknown as Record<string, unknown>)['model'] as
      | { contextWindowLimit?: number; countTokens?: (messages: Message[]) => Promise<number> }
      | undefined
    if (!model?.contextWindowLimit || !model?.countTokens) return 0

    const tokens = await model.countTokens(this._agent.messages)
    return tokens / model.contextWindowLimit
  }

  /**
   * Last-resort truncation: drop the oldest messages (preserving the first user message)
   * until at least 20% of messages are removed.
   */
  private _floorTruncate(messages: Message[]): void {
    const targetRemoval = Math.max(1, Math.floor(messages.length * 0.2))
    let removed = 0
    let index = 1 // skip first message (first user message)

    while (removed < targetRemoval && index < messages.length - 2) {
      messages.splice(index, 1)
      removed++
    }
  }

  private _resolveSessionId(agent: LocalAgent): string {
    const agentRecord = agent as unknown as Record<string, unknown>
    const sessionManager = agentRecord['sessionManager'] as { _sessionId?: string } | undefined
    if (sessionManager?._sessionId) {
      return sessionManager._sessionId
    }
    return agent.id
  }

  private _buildStash(): Stash | undefined {
    if (!this._stashEnabled) return undefined

    const scopedStorage = this._scopeStorage()
    return new Stash(scopedStorage)
  }

  private _scopeStorage(): Storage {
    if (NAMESPACED in this._storage) {
      return this._storage
    }

    const prefix = `${STORAGE_PREFIX}/${this._sessionId}/scopes/agent/${this._agentId}`

    if (this._storage.namespace) {
      return this._storage.namespace(prefix)
    }

    return namespace(this._storage, prefix)
  }

  private async _onMessageAdded(event: MessageAddedEvent): Promise<void> {
    if (this._stash === undefined) return

    await this._stash.writeMessage(event.message)
  }
}

function resolveStashEnabled(stash: StashConfig | boolean | undefined): boolean {
  if (stash === undefined || stash === true) return true
  if (stash === false) return false
  return stash.enabled ?? true
}
