/**
 * Result status vocabulary from the shared multi-agent result dialect.
 *
 * Deliberately distinct from the SDK's `Status` enum: tool consumers see a
 * stable `success`/`error`/`cancelled` regardless of which underlying multi-agent
 * pattern was used. See `_multiagent-conventions.md`.
 */
export type SwarmToolStatus = 'success' | 'error' | 'cancelled'

/**
 * Result shape returned by the swarm vended tool.
 *
 * Shared with sibling multi-agent tools (`use-agent`, `graph`, `a2a-client`).
 * See `_multiagent-conventions.md` for the full dialect.
 */
export interface SwarmToolResult {
  /**
   * Result status in the shared multi-agent dialect.
   *
   * Translated from the SDK's execution vocabulary so callers get a stable
   * contract across every multi-agent tool.
   */
  status: SwarmToolStatus
  /** Concatenated text from the terminal agent's content blocks. Empty if none. */
  output: string
  /** Ordered ids of the agents that produced results, in completion order. */
  nodeHistory: string[]
  /** Total number of node results captured. */
  executionCount: number
  /** Total wall-clock duration in milliseconds. */
  executionTimeMs: number
  /** Token usage aggregated across all child agents. */
  usage: {
    inputTokens: number
    outputTokens: number
    totalTokens: number
  }
}

/**
 * Description shown to the model for the swarm tool.
 */
export const SWARM_TOOL_DESCRIPTION =
  'Spin up a small team of sub-agents that hand off to each other to complete a task. ' +
  'Use when the task splits cleanly into specialized roles and you want the sub-agents ' +
  'to decide amongst themselves who does what. ' +
  'Each sub-agent is defined by a name, systemPrompt, and a (possibly empty) tools list. ' +
  'Sub-agents inherit your model. Child sub-agents may only use tools you allowlist from ' +
  'your own tool registry. Returns the final response plus which sub-agents ran.'
