/**
 * Types for the a2a_client vended tool.
 */

import type { A2AAgentConfig } from '../../a2a/a2a-agent.js'

/**
 * Subset of the resolved remote agent card echoed back in the result.
 */
export interface A2AClientRemoteCard {
  /** Human-readable agent name from the resolved agent card. */
  name: string
  /** Human-readable agent description from the resolved agent card. */
  description: string
  /** URL of the resolved agent (from the card, or the request URL as fallback). */
  url: string
}

/**
 * Output of an a2a_client invocation.
 *
 * Follows the shared multi-agent tool result shape defined in
 * `_multiagent-conventions.md`: a top-level `status` / `output` /
 * `executionTimeMs` triple, plus a `remoteCard` addendum describing the
 * resolved remote endpoint.
 *
 * `status` is only ever `'success'`. Cancellation and timeout raise a
 * `DOMException` with `name === 'AbortError'` rather than returning a
 * `'cancelled'` variant, matching the other network tools (`http_request`,
 * `web_fetch`). See the a2a_client addendum in `_multiagent-conventions.md`
 * for the rationale.
 */
export interface A2AClientOutput {
  /** Result status. Always `"success"` on a normal return. */
  status: 'success'
  /** Text produced by the remote agent, concatenated from all text parts. */
  output: string
  /** Total wall-clock time for the tool call, in milliseconds. */
  executionTimeMs: number
  /** Subset of the resolved remote agent card. */
  remoteCard: A2AClientRemoteCard
}

/**
 * Developer-time options for the `makeA2AClient` factory.
 *
 * Every field is bound at construction time by the developer. The model
 * can only supply `url` and `message` at call time.
 */
export interface MakeA2AClientOptions {
  /** Tool name. Defaults to `'a2a_client'`. */
  name?: string
  /** Tool description shown to the model. */
  description?: string
  /**
   * Optional developer-supplied URL allowlist. When set, the model-provided
   * `url` must start with one of these prefixes.
   */
  allowedUrlPrefixes?: readonly string[]
  /**
   * Wall-clock cap on the entire tool call, in seconds. Bounds both card
   * discovery and message send. Default: 60.
   */
  timeoutSeconds?: number
  /**
   * Cap on the agent-card size in bytes (serialized JSON). Cards larger than
   * this are rejected. Default: 262144 (256 KiB).
   */
  maxCardBytes?: number
  /**
   * Cap on the returned response text size in bytes. Text beyond the cap is
   * truncated with a `... [truncated]` marker. Default: 262144 (256 KiB).
   */
  maxResponseBytes?: number
  /**
   * Extra static options for the underlying `A2AAgent`. Auth material such
   * as a custom `clientFactory` for signed requests goes here. Never
   * touched by the model.
   */
  agentConfig?: Omit<A2AAgentConfig, 'url'>
  /**
   * Cap on the shared multi-agent recursion depth counter. A parent agent
   * that calls `a2a_client` counts as depth+1. The tool refuses to run once
   * the counter reaches the cap. Not propagated across the wire — the
   * remote agent's depth resets from its perspective. Default: 3.
   */
  multiagentDepthCap?: number
}
