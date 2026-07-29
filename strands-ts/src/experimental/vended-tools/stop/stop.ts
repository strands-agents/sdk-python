import { tool } from '../../../tools/tool-factory.js'
import { AfterToolsEvent } from '../../../hooks/events.js'
import type { LocalAgent } from '../../../types/agent.js'
import {
  DEFAULT_MAX_STOP_MESSAGE_LENGTH,
  DEFAULT_STOP_DESCRIPTION,
  DEFAULT_STOP_MESSAGE,
  STOP_INVOCATION_STATE_KEY,
  buildStopInputSchema,
} from './types.js'

/**
 * Tracks agents on which the terminal `AfterToolsEvent` hook has been
 * installed, so we install it at most once per agent instance regardless of
 * how many times the model calls `stop`. Keyed weakly so garbage-collected
 * agents drop out automatically.
 */
const installedAgents = new WeakSet<LocalAgent>()

/**
 * Options accepted by {@link makeStop}.
 */
export interface MakeStopOptions {
  /**
   * Tool name shown to the model. Defaults to `'stop'`.
   */
  name?: string
  /**
   * Tool description shown to the model. Defaults to {@link DEFAULT_STOP_DESCRIPTION}.
   */
  description?: string
  /**
   * Maximum accepted length for the model-supplied `message` argument, in
   * characters. Must be a positive integer. Defaults to
   * {@link DEFAULT_MAX_STOP_MESSAGE_LENGTH} (4096).
   */
  maxMessageLength?: number
}

/**
 * Shape of the stop marker written to `invocationState`.
 *
 * @internal
 */
interface StopMarker {
  /** The final message the tool returned, or a default if none was provided. */
  message: string
}

/**
 * Installs a one-time `AfterToolsEvent` hook that ends the agent loop when the
 * stop tool has set its marker on `invocationState`. Reads the marker fresh
 * on every event so subsequent invocations of the same agent can call `stop`
 * again without leaking state across turns.
 */
function ensureHookInstalled(agent: LocalAgent): void {
  if (installedAgents.has(agent)) return

  // Register the hook FIRST — if `addHook` throws, we haven't marked the agent
  // as installed, so a later call still gets a chance to install one. Marking
  // before registering would silently no-op every future stop() on this agent.
  agent.addHook(AfterToolsEvent, (event) => {
    const marker = event.invocationState[STOP_INVOCATION_STATE_KEY] as StopMarker | null | undefined
    // JSON round-tripping `invocationState` can turn an absent field into
    // `null`; treat both as "no stop requested".
    if (marker == null) return
    // Consume the marker to avoid re-firing if the caller reuses `invocationState` across invocations.
    delete event.invocationState[STOP_INVOCATION_STATE_KEY]
    event.endTurn = marker.message
  })

  installedAgents.add(agent)
}

/**
 * Create a stop tool that gracefully ends the agent loop.
 *
 * **Experimental** — this tool is subject to change in future revisions without notice.
 *
 * Shims onto the SDK's existing `AfterToolsEvent.endTurn` primitive: the tool
 * records the model's optional final message on `invocationState`, then a
 * lazily-installed hook on the agent reads that marker on the terminating
 * `AfterToolsEvent` and sets `endTurn` to the message, which the agent loop
 * already treats as a cooperative stop signal.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { stop } from '@strands-agents/sdk/experimental/vended-tools/stop'
 *
 * const agent = new Agent({ model, tools: [stop] })
 * ```
 */
export function makeStop(options?: MakeStopOptions): ReturnType<typeof tool> {
  const maxMessageLength = options?.maxMessageLength ?? DEFAULT_MAX_STOP_MESSAGE_LENGTH
  if (!Number.isInteger(maxMessageLength) || maxMessageLength <= 0) {
    throw new Error(`maxMessageLength must be a positive integer, got ${String(maxMessageLength)}`)
  }
  return tool({
    name: options?.name ?? 'stop',
    description: options?.description ?? DEFAULT_STOP_DESCRIPTION,
    inputSchema: buildStopInputSchema(maxMessageLength),
    callback: (input, context) => {
      if (!context) {
        throw new Error('Tool context is required for stop operations')
      }

      // Fall back to the default when message is absent (undefined or null,
      // e.g. from providers that serialize omitted fields as null) OR empty.
      // The agent loop treats `endTurn === ''` as falsy, so an empty string
      // would fail to halt the loop even though the model called `stop`.
      const message = input.message != null && input.message.length > 0 ? input.message : DEFAULT_STOP_MESSAGE

      ensureHookInstalled(context.agent)
      const marker: StopMarker = { message }
      context.invocationState[STOP_INVOCATION_STATE_KEY] = marker

      return message
    },
  })
}

/**
 * Default stop tool.
 *
 * **Experimental** — this tool is subject to change in future revisions without notice.
 *
 * Ends the agent loop cooperatively when called by the model. Any tools the
 * model requested alongside `stop` in the same turn still run to completion —
 * the loop halts after the batch, without calling the model again.
 */
export const stop = makeStop()
