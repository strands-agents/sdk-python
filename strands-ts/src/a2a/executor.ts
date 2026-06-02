/**
 * A2A executor that bridges a Strands Agent into the A2A protocol.
 *
 * Implements the AgentExecutor interface from `@a2a-js/sdk/server` to allow
 * a Strands Agent to handle A2A JSON-RPC requests.
 */

import type { ExecutionEventBus, RequestContext } from '@a2a-js/sdk/server'
import type { AgentExecutor } from '@a2a-js/sdk/server'
import { A2AError } from '@a2a-js/sdk/server'
import type { InvokableAgent } from '../types/agent.js'
import { ModelStreamUpdateEvent, ContentBlockEvent } from '../hooks/events.js'
import { contentBlocksToParts, partsToContentBlocks } from './adapters.js'
import { normalizeError } from '../errors.js'
import { logger } from '../logging/logger.js'

/**
 * Bridges a Strands Agent into the A2A protocol as an AgentExecutor.
 *
 * Converts A2A message parts to Strands content blocks, streams the agent
 * execution, and publishes text deltas as artifact updates through the A2A
 * event bus. Text chunks are appended to a single artifact as they arrive,
 * implementing A2A-compliant streaming behavior.
 *
 * ## Invocation state
 *
 * The executor populates the agent's `invocationState` with the incoming A2A
 * {@link RequestContext} under the reserved key `a2aRequestContext`. Hooks and
 * tools running inside the agent can read `event.invocationState.a2aRequestContext`
 * to correlate with the A2A request (taskId, contextId, user message metadata)
 * for logging, metrics, or audit.
 *
 * Because the A2A framework (not user code) drives `execute()`, there is no
 * per-request path for the user to supply their own `invocationState`. If a
 * user hook writes to the `a2aRequestContext` key, it will be overwritten on
 * the next request.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { A2AExecutor } from '@strands-agents/sdk/a2a'
 *
 * const agent = new Agent({ model: 'my-model' })
 * const executor = new A2AExecutor(agent)
 * ```
 */
export class A2AExecutor implements AgentExecutor {
  private _agent: InvokableAgent

  /**
   * Creates a new A2AExecutor.
   *
   * @param agent - The agent to execute for incoming A2A requests
   */
  constructor(agent: InvokableAgent) {
    this._agent = agent
  }

  /**
   * Executes the agent in response to an A2A message.
   *
   * Converts A2A message parts to Strands content blocks, then streams the
   * agent execution. Text deltas are streamed incrementally into a single
   * artifact; non-text content blocks (images, videos, documents) are each
   * published as separate complete artifacts. A final artifact with
   * `lastChunk: true` signals the end of the text artifact, followed by a
   * completed status update.
   *
   * @param context - The A2A request context containing the user message
   * @param eventBus - The event bus for publishing A2A artifact and status events
   */
  async execute(context: RequestContext, eventBus: ExecutionEventBus): Promise<void> {
    const { taskId, contextId, userMessage } = context
    const contentBlocks = partsToContentBlocks(userMessage.parts)
    if (contentBlocks.length === 0) {
      throw A2AError.invalidRequest('No content blocks available')
    }

    // Publish initial task event to register the task with the ResultManager.
    // Without this, artifact and status events are ignored as "unknown task".
    eventBus.publish({ kind: 'task', id: taskId, contextId, status: { state: 'working' } })

    const artifactId = globalThis.crypto.randomUUID()
    let isFirstChunk = true

    try {
      // Forward the A2A RequestContext to the agent under a reserved key so
      // hooks and tools can correlate with the A2A request (taskId, contextId,
      // user message metadata).
      const stream = this._agent.stream(contentBlocks, {
        invocationState: { a2aRequestContext: context },
      })
      let next = await stream.next()

      while (!next.done) {
        const event = next.value

        // Stream text deltas incrementally into the text artifact
        if (
          event instanceof ModelStreamUpdateEvent &&
          event.event.type === 'modelContentBlockDeltaEvent' &&
          event.event.delta.type === 'textDelta'
        ) {
          eventBus.publish({
            kind: 'artifact-update',
            taskId,
            contextId,
            artifact: {
              artifactId,
              parts: [{ kind: 'text', text: event.event.delta.text }],
            },
            append: !isFirstChunk,
          })
          isFirstChunk = false
        }

        // Publish non-text content blocks (images, videos, documents) as separate artifacts
        if (event instanceof ContentBlockEvent && event.contentBlock.type !== 'textBlock') {
          const parts = contentBlocksToParts([event.contentBlock])
          if (parts.length > 0) {
            eventBus.publish({
              kind: 'artifact-update',
              taskId,
              contextId,
              artifact: { artifactId: globalThis.crypto.randomUUID(), parts },
              append: false,
              lastChunk: true,
            })
          }
        }

        next = await stream.next()
      }

      // Publish final artifact chunk to signal end of artifact
      eventBus.publish({
        kind: 'artifact-update',
        taskId,
        contextId,
        artifact: {
          artifactId,
          // If no deltas were streamed, publish the full result; otherwise empty to close the artifact
          parts: [{ kind: 'text', text: isFirstChunk && next.value ? next.value.toString() : '' }],
        },
        append: !isFirstChunk, // false for new artifact, true to append to streamed chunks
        lastChunk: true, // Always true — this runs after the stream loop ends
      })

      eventBus.publish({ kind: 'status-update', taskId, contextId, status: { state: 'completed' }, final: true })
    } catch (error) {
      logger.error(`task_id=<${taskId}> | error in streaming execution`, normalizeError(error))
      throw error
    }
  }

  /**
   * Cancels a running task. Not supported by this executor.
   *
   * @param taskId - The ID of the task to cancel
   * @param eventBus - The event bus for publishing status events
   */
  async cancelTask(_taskId: string, _eventBus: ExecutionEventBus): Promise<void> {
    throw A2AError.unsupportedOperation('Task cancellation is not supported')
  }
}
