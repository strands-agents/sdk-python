/**
 * ContextManager: first-class agent component for durable context management.
 *
 * Writes every message to L1 (the durable transcript) on arrival.
 * L0 (agent.messages) is the compressed working set; L1 is the source of truth.
 * Future PRs add strategy-driven offloading, injection, and retrieval.
 */

import type { Plugin } from '../plugins/plugin.js'
import type { LocalAgent } from '../types/agent.js'
import type { Storage } from '../storage/storage.js'
import { NAMESPACED, namespace } from '../storage/storage.js'
import { InMemoryStorage } from '../storage/in-memory-storage.js'
import { MessageAddedEvent } from '../hooks/events.js'
import { logger } from '../logging/logger.js'
import { Transcript } from './transcript.js'
import type { ContextManagerConfig, StashConfig } from './types.js'

const STORAGE_PREFIX = 'context'

/**
 * Manages the L1 durable transcript for an agent's conversation.
 *
 * Every message is written to L1 on arrival via the `MessageAddedEvent` hook.
 * L0 (`agent.messages`) remains the in-memory working set that the model sees;
 * L1 holds the full, uncompressed history and serves as the source of truth for
 * later retrieval and restore operations.
 *
 * The ContextManager is a Plugin — attach it via `plugins: [new ContextManager(...)]`
 * on the Agent constructor.
 *
 * @example
 * ```typescript
 * import { Agent } from 'strands-agents'
 * import { ContextManager } from 'strands-agents/context-manager'
 * import { LocalFileStorage } from 'strands-agents/storage'
 *
 * const agent = new Agent({
 *   model,
 *   plugins: [new ContextManager({ storage: new LocalFileStorage('./.strands/') })],
 * })
 * ```
 */
export class ContextManager implements Plugin {
  readonly name = 'strands:context-manager'

  private readonly _storage: Storage
  private readonly _stashEnabled: boolean
  private readonly _strategies: unknown[]

  private _transcript: Transcript | undefined
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

  /** The L1 transcript writer, or undefined if stash is disabled. */
  get transcript(): Transcript | undefined {
    return this._transcript
  }

  initAgent(agent: LocalAgent): void {
    this._agentId = agent.id
    this._sessionId = this._resolveSessionId(agent)
    this._transcript = this._buildTranscript()

    if (!this._stashEnabled) {
      logger.info(`agentId=<${this._agentId}> | L1 stash disabled, offload operations will be destructive`)
    }

    agent.addHook(MessageAddedEvent, (event) => this._onMessageAdded(event))
  }

  private _resolveSessionId(agent: LocalAgent): string {
    const agentRecord = agent as unknown as Record<string, unknown>
    const sessionManager = agentRecord['sessionManager'] as { _sessionId?: string } | undefined
    if (sessionManager?._sessionId) {
      return sessionManager._sessionId
    }
    return agent.id
  }

  private _buildTranscript(): Transcript | undefined {
    if (!this._stashEnabled) return undefined

    const scopedStorage = this._scopeStorage()
    return new Transcript(scopedStorage)
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
    if (this._transcript === undefined) return

    await this._transcript.writeMessage(event.message)
  }
}

function resolveStashEnabled(stash: StashConfig | boolean | undefined): boolean {
  if (stash === undefined || stash === true) return true
  if (stash === false) return false
  return stash.enabled ?? true
}
