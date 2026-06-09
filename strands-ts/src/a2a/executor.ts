/**
 * A2A executor that bridges a Strands Agent into the A2A protocol.
 *
 * Implements the AgentExecutor interface from `@a2a-js/sdk/server` to allow
 * a Strands Agent to handle A2A JSON-RPC requests.
 */

import type { ExecutionEventBus, RequestContext } from '@a2a-js/sdk/server'
import type { AgentExecutor } from '@a2a-js/sdk/server'
import { A2AError } from '@a2a-js/sdk/server'
import type { InvokableAgent, LocalAgent } from '../types/agent.js'
import type { Snapshot } from '../types/snapshot.js'
import { ModelStreamUpdateEvent, ContentBlockEvent } from '../hooks/events.js'
import { contentBlocksToParts, partsToContentBlocks } from './adapters.js'
import { normalizeError } from '../errors.js'
import { logger } from '../logging/logger.js'
import { AsyncLock } from './async-lock.js'

/** Deep-copies a snapshot. Snapshots are JSON-serializable, so a round-trip is sufficient. */
function cloneSnapshot(snapshot: Snapshot): Snapshot {
  return JSON.parse(JSON.stringify(snapshot)) as Snapshot
}

/**
 * A factory that builds a fresh {@link Agent} for a given A2A `contextId`.
 *
 * Invoked once per context. Each returned agent owns an independent conversation
 * and runs under its own lock, so different contexts execute concurrently and
 * never share state. The factory is also where per-context concerns such as a
 * `sessionManager` are wired.
 */
export type AgentFactory = (contextId: string) => InvokableAgent

/**
 * Cap on concurrently tracked A2A contexts. Beyond this, the least-recently-used
 * context is evicted to bound memory in long-running servers.
 */
export const DEFAULT_MAX_CONTEXTS = 1000

/**
 * Per-context bookkeeping for factory mode: a dedicated agent and its serializing lock.
 *
 * Keeping the agent and lock in one entry means the LRU map has a single source of
 * truth; there is no second map to keep in sync on insert, reorder, or eviction.
 */
interface ContextEntry {
  agent: InvokableAgent
  lock: AsyncLock
}

/**
 * Options for constructing an {@link A2AExecutor}.
 *
 * Provide exactly one of `agent` (deprecated) or `agentFactory`.
 */
export interface A2AExecutorOptions {
  /** A single agent reused across contexts. Deprecated; prefer `agentFactory`. */
  agent?: InvokableAgent
  /**
   * Callable that takes a `contextId` and returns a fresh agent per context. Recommended.
   */
  agentFactory?: AgentFactory
  /**
   * Maximum number of contexts to retain concurrently; the least-recently-used is
   * evicted beyond this. Must be at least 1. Defaults to {@link DEFAULT_MAX_CONTEXTS}.
   */
  maxContexts?: number
}

/**
 * An agent that can both be invoked and snapshotted — i.e. a full Strands `Agent`.
 * Single-agent mode requires both: `stream` to run, and `takeSnapshot`/`loadSnapshot`
 * to swap per-context state on and off the shared instance.
 */
type SnapshotAgent = InvokableAgent & LocalAgent

/**
 * Narrows an {@link InvokableAgent} to a {@link SnapshotAgent}, which exposes the
 * snapshot APIs needed for single-agent-mode state swapping.
 */
function asSnapshotAgent(agent: InvokableAgent): SnapshotAgent {
  const candidate = agent as Partial<LocalAgent>
  if (typeof candidate.takeSnapshot !== 'function' || typeof candidate.loadSnapshot !== 'function') {
    throw new Error(
      'A2AExecutor requires an Agent that supports snapshots (takeSnapshot/loadSnapshot). ' +
        'Pass a Strands Agent instance.'
    )
  }
  return agent as unknown as SnapshotAgent
}

/**
 * Whether an agent has a configured `sessionManager`. Single-agent mode rejects this:
 * snapshot-swapping one shared instance would interleave every context into one session.
 * `sessionManager` is an `Agent` field not declared on {@link InvokableAgent}, so it is
 * read defensively here.
 */
function hasSessionManager(agent: InvokableAgent): boolean {
  return (agent as { sessionManager?: unknown }).sessionManager !== undefined
}

/**
 * Bridges a Strands Agent into the A2A protocol as an AgentExecutor.
 *
 * Converts A2A message parts to Strands content blocks, streams the agent
 * execution, and publishes text deltas as artifact updates through the A2A
 * event bus. Text chunks are appended to a single artifact as they arrive,
 * implementing A2A-compliant streaming behavior.
 *
 * ## Conversation isolation
 *
 * Conversation state is isolated per A2A `contextId` so callers in different
 * contexts cannot read or influence each other's history. There are two modes:
 *
 * - **`agentFactory`** (recommended): a callable that takes a `contextId` and returns
 *   a dedicated agent, invoked once per context. Each context owns an independent
 *   agent and runs under its own lock, so different contexts execute concurrently
 *   and never share state. The factory is also where per-context concerns such as
 *   a `sessionManager` are wired.
 * - **`agent`** (deprecated): a single agent reused across contexts. Each context's
 *   conversation state is swapped on/off this instance under a lock, so requests are
 *   serialized. Not multi-tenant safe for concurrency; prefer `agentFactory`.
 *
 * Contexts are keyed on the client-supplied `contextId`, which is **not** an
 * authentication boundary. A caller that knows another caller's `contextId` can
 * attach to that conversation. Multi-tenant deployments must enforce authenticated
 * identity at the transport/gateway layer.
 *
 * At most `maxContexts` contexts are retained; beyond that the least-recently-used
 * is evicted and a later request reusing that `contextId` starts fresh.
 *
 * ## Invocation state
 *
 * The executor populates the agent's `invocationState` with the incoming A2A
 * {@link RequestContext} under the reserved key `a2aRequestContext`. Hooks and
 * tools running inside the agent can read `event.invocationState.a2aRequestContext`
 * to correlate with the A2A request (taskId, contextId, user message metadata).
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { A2AExecutor } from '@strands-agents/sdk/a2a'
 *
 * // Recommended: a dedicated agent per context
 * const executor = new A2AExecutor({ agentFactory: (contextId) => new Agent({ model: 'my-model' }) })
 * ```
 */
export class A2AExecutor implements AgentExecutor {
  private readonly _agentFactory?: AgentFactory
  private readonly _maxContexts: number

  /** Guards the per-context bookkeeping maps below. */
  private readonly _contextsLock = new AsyncLock()

  // Factory mode: a dedicated agent and lock per context.
  private readonly _contexts = new Map<string, ContextEntry>()

  // Single-agent mode: one shared agent, swapping each context's snapshot on/off it.
  private readonly _agent?: SnapshotAgent
  private readonly _templateSnapshot?: Snapshot
  private readonly _snapshots = new Map<string, Snapshot>()

  /**
   * Creates a new A2AExecutor.
   *
   * Provide exactly one of `agent` (deprecated) or `agentFactory`.
   *
   * @param options - Executor options: `agent` or `agentFactory`, plus optional `maxContexts`.
   */
  constructor(options: A2AExecutorOptions = {}) {
    const { agent, agentFactory, maxContexts = DEFAULT_MAX_CONTEXTS } = options

    if (maxContexts < 1) {
      throw new Error(`maxContexts must be at least 1, got ${maxContexts}`)
    }
    if ((agent === undefined) === (agentFactory === undefined)) {
      throw new Error("Provide exactly one of 'agent' or 'agentFactory'.")
    }

    this._maxContexts = maxContexts

    if (agentFactory !== undefined) {
      this._agentFactory = agentFactory
    } else {
      // Single-agent mode: reuse one agent, swapping each context's snapshot on/off it.
      const sharedAgent = asSnapshotAgent(agent!)
      if (hasSessionManager(sharedAgent)) {
        throw new Error(
          "A single 'agent' with a sessionManager is not supported: the session manager " +
            "persists every context's messages into one interleaved session. Use " +
            "'agentFactory' to build a per-context agent with its own sessionManager."
        )
      }
      logger.warn(
        "Passing a single 'agent' to A2AExecutor is deprecated and will be removed in a future " +
          'version. A single agent serializes all requests; pass an agentFactory (a callable taking ' +
          'the contextId) instead to isolate conversations per context.'
      )
      this._agent = sharedAgent
      // The template snapshot is the agent's clean state, captured before any request mutates it.
      this._templateSnapshot = this._captureState(sharedAgent)
    }
  }

  /** Snapshot an agent's session state. */
  private _captureState(agent: LocalAgent): Snapshot {
    return agent.takeSnapshot({ preset: 'session' })
  }

  /**
   * Load a snapshot into an agent, restoring its session state.
   *
   * Deep-copies once at the boundary so a run never mutates a stored or template
   * snapshot in place (snapshots, including the template, are restored repeatedly).
   */
  private _restoreState(agent: LocalAgent, snapshot: Snapshot): void {
    agent.loadSnapshot(cloneSnapshot(snapshot))
  }

  /** Evict least-recently-used contexts beyond `maxContexts`. Caller holds the contexts lock. */
  private _evictExcessContexts(): void {
    const contexts: Map<string, unknown> = this._agentFactory !== undefined ? this._contexts : this._snapshots
    // Map preserves insertion order; the first key is the least-recently-used because
    // touched contexts are re-inserted at the end (delete-then-set on access).
    while (contexts.size > this._maxContexts) {
      const evictedId = contexts.keys().next().value as string
      contexts.delete(evictedId)
      logger.debug(`context_id=<${evictedId}> | evicted least-recently-used A2A context`)
    }
  }

  /** Return the dedicated agent and lock for a context, building it on first use (factory mode). */
  private async _acquireContextAgent(contextId: string): Promise<ContextEntry> {
    const release = await this._contextsLock.acquire()
    try {
      let entry = this._contexts.get(contextId)
      if (entry === undefined) {
        entry = { agent: this._agentFactory!(contextId), lock: new AsyncLock() }
        this._contexts.set(contextId, entry)
        this._evictExcessContexts()
      } else {
        // Mark most-recently-used: delete-then-set moves the entry to the Map's end.
        this._contexts.delete(contextId)
        this._contexts.set(contextId, entry)
      }
      return entry
    } finally {
      release()
    }
  }

  /**
   * Executes the agent in response to an A2A message.
   *
   * Routes to the per-context agent (factory mode) or the shared agent
   * (single-agent mode), isolating conversation state by `contextId`.
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

    if (this._agentFactory !== undefined) {
      await this._runWithContextAgent(context, contentBlocks, eventBus)
    } else {
      await this._runWithSharedAgent(context, contentBlocks, eventBus)
    }
  }

  /**
   * Factory mode: run against this context's dedicated agent, serialized only per context.
   */
  private async _runWithContextAgent(
    context: RequestContext,
    contentBlocks: ReturnType<typeof partsToContentBlocks>,
    eventBus: ExecutionEventBus
  ): Promise<void> {
    const { agent, lock } = await this._acquireContextAgent(context.contextId)
    const release = await lock.acquire()
    try {
      await this._streamAgent(agent, context, contentBlocks, eventBus)
    } finally {
      release()
    }
  }

  /**
   * Single-agent mode: swap this context's snapshot on/off the shared agent under a lock.
   *
   * The lock serializes all requests (a single agent cannot be invoked concurrently). The
   * agent is reset to the template afterward so no context's data lingers on it.
   */
  private async _runWithSharedAgent(
    context: RequestContext,
    contentBlocks: ReturnType<typeof partsToContentBlocks>,
    eventBus: ExecutionEventBus
  ): Promise<void> {
    const agent = this._agent!
    const release = await this._contextsLock.acquire()
    try {
      this._restoreState(agent, this._snapshots.get(context.contextId) ?? this._templateSnapshot!)
      try {
        await this._streamAgent(agent, context, contentBlocks, eventBus)
      } finally {
        // Persist this context's updated history (even on error, to retain partial turns),
        // evict beyond the cap, then reset the shared agent for the next caller. Delete-then-set
        // places the entry at the most-recently-used end whether or not it already existed.
        this._snapshots.delete(context.contextId)
        this._snapshots.set(context.contextId, this._captureState(agent))
        this._evictExcessContexts()
        this._restoreState(agent, this._templateSnapshot!)
      }
    } finally {
      release()
    }
  }

  /**
   * Streams one agent invocation and translates its events to A2A artifact updates.
   */
  private async _streamAgent(
    agent: InvokableAgent,
    context: RequestContext,
    contentBlocks: ReturnType<typeof partsToContentBlocks>,
    eventBus: ExecutionEventBus
  ): Promise<void> {
    const { taskId, contextId } = context
    const artifactId = globalThis.crypto.randomUUID()
    let isFirstChunk = true

    try {
      // Forward the A2A RequestContext to the agent under a reserved key so
      // hooks and tools can correlate with the A2A request (taskId, contextId,
      // user message metadata).
      const stream = agent.stream(contentBlocks, {
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
