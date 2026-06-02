/**
 * Base A2A server that manages agent card and request handler setup.
 *
 * This module is browser-compatible. For Express-based HTTP serving,
 * see {@link A2AExpressServer} in `./express-server.ts`.
 *
 * The A2A protocol is experimental, so breaking changes in the underlying SDK
 * may require breaking changes in this module.
 */

import type { AgentCard, AgentSkill } from '@a2a-js/sdk'
import type { TaskStore, A2ARequestHandler } from '@a2a-js/sdk/server'
import { DefaultRequestHandler, InMemoryTaskStore } from '@a2a-js/sdk/server'
import type { InvokableAgent } from '../types/agent.js'
import { A2AExecutor } from './executor.js'

/**
 * Configuration options for creating an A2AServer.
 */
export interface A2AServerConfig {
  /** The Strands Agent to serve via A2A protocol */
  agent: InvokableAgent
  /** Human-readable name for the agent */
  name: string
  /** Optional description of the agent's purpose */
  description?: string
  /** Public URL override for the agent card */
  httpUrl?: string
  /** Version string for the agent card (default: '0.0.1') */
  version?: string
  /** Skills to advertise in the agent card */
  skills?: AgentSkill[]
  /** Task store for persisting task state */
  taskStore?: TaskStore
}

/**
 * Base A2A server that manages agent card and request handler setup.
 *
 * Subclass this to integrate with different HTTP frameworks. For Express,
 * use {@link A2AExpressServer}.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { A2AExpressServer } from '@strands-agents/sdk/a2a/express'
 *
 * const agent = new Agent({ model: 'my-model' })
 * const server = new A2AExpressServer({
 *   agent,
 *   name: 'My Agent',
 *   description: 'An agent that helps with tasks',
 * })
 *
 * await server.serve()
 * ```
 */
export class A2AServer {
  protected _agentCard: AgentCard
  protected _requestHandler: A2ARequestHandler

  /**
   * Creates a new A2AServer.
   *
   * @param config - Configuration for the server
   */
  constructor(config: A2AServerConfig) {
    const httpUrl = config.httpUrl ?? ''

    this._agentCard = {
      name: config.name,
      description: config.description ?? '',
      version: config.version ?? '0.0.1',
      protocolVersion: '0.2.0',
      url: httpUrl,
      defaultInputModes: ['text/plain'],
      defaultOutputModes: ['text/plain'],
      skills: config.skills ?? [],
      capabilities: {
        streaming: true,
      },
    }

    const taskStore = config.taskStore ?? new InMemoryTaskStore()
    const executor = new A2AExecutor(config.agent)
    this._requestHandler = new DefaultRequestHandler(this._agentCard, taskStore, executor)
  }

  /**
   * Returns the agent card for this server.
   */
  get agentCard(): AgentCard {
    return this._agentCard
  }
}
