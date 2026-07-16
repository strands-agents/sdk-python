/**
 * Types for the use_agent vended tool.
 */

/**
 * Input parameters accepted by the use_agent tool.
 *
 * The Zod input schema is built inside {@link makeUseAgent}'s closure so its
 * describe strings can reference the factory-configurable caps; this
 * hand-written interface is the single source of truth for the external
 * shape. Keep the two aligned if you add or rename a field.
 */
export interface UseAgentInput {
  /** System prompt for the nested agent. Non-empty; capped in UTF-8 bytes. */
  systemPrompt: string
  /** Task to hand the nested agent. Non-empty; capped in UTF-8 bytes. */
  task: string
  /**
   * Exact-name allowlist of tools to expose to the nested agent. Every entry
   * must be a tool that exists in the parent agent's registry. Wildcards and
   * multi-agent tool names ('use_agent', 'swarm', 'graph', 'a2a_client') are
   * rejected.
   */
  tools?: string[]
}

/**
 * Status values returned by the use_agent tool. Values are the lower-cased
 * SDK Status enum vocabulary shared across every multi-agent vended tool
 * (see `_multiagent-conventions.md`).
 *
 * `cancelled` never appears in a returned result on TypeScript. The tool
 * re-raises an AbortError on cancellation, matching the sibling http-request
 * tool's idiom; the value is kept in the union for documentation of the
 * shared shape.
 */
export type UseAgentStatus = 'completed' | 'failed' | 'cancelled' | 'interrupted'

/**
 * Result returned by the use_agent tool. Matches the shared multi-agent
 * result shape from `_multiagent-conventions.md`.
 *
 * `completed` maps to a child that finished with `stopReason === 'endTurn'`.
 * `interrupted` maps to `stopReason === 'interrupt'`. Any other non-cancelled
 * stop reason (`limitTurns`, `contentFiltered`, `maxTokens`,
 * `guardrailIntervened`, ...) maps to `failed` so the parent can distinguish
 * a completed delegation from one that hit a policy or limit.
 */
export interface UseAgentResult {
  status: UseAgentStatus
  output: string
  executionTimeMs: number
}
