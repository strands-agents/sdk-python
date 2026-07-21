/**
 * Type definitions, caps, and description constants for the graph vended tool.
 */

/**
 * Maximum number of nodes the tool will accept in one call.
 * Sized to keep the DAG a coordination surface, not a job runner. Anything
 * meaningfully larger belongs in a user-authored `Graph` outside the tool.
 */
export const MAX_NODES = 20

/** Maximum number of edges the tool will accept in one call. */
export const MAX_EDGES = 40

/** Maximum length of a node id (characters). */
export const MAX_ID_LENGTH = 64

/** Maximum length (characters) of any node's `systemPrompt`. */
export const MAX_SYSTEM_PROMPT_LENGTH = 8_000

/** Maximum length (characters) of the `initialInput` passed to the graph. */
export const MAX_INITIAL_INPUT_LENGTH = 32_000

/** Wall-clock budget for the whole graph invocation, in milliseconds. */
export const DEFAULT_TIMEOUT_MS = 300_000

/** Wall-clock budget for each node in the graph, in milliseconds. */
export const DEFAULT_NODE_TIMEOUT_MS = 120_000

/** Maximum steps (node executions) the underlying `Graph` is allowed to take. */
export const MAX_STEPS = 40

/** Maximum number of tool names a single node may list in its allow-list. */
export const MAX_TOOLS_PER_NODE = 64

/** Tool description shown to the model. */
export const GRAPH_DESCRIPTION =
  'Runs a deterministic directed acyclic graph (DAG) of sub-agents. ' +
  'You describe the nodes (each with an optional system prompt and tool allow-list) ' +
  'and the edges (dependency order); the tool executes the graph, feeding each node ' +
  'the outputs of its dependencies, and returns the results keyed by node id. ' +
  'Use when a task has a fixed pipeline shape with clear dependencies; use plain tool ' +
  'calls for single-step work.'

/**
 * One node's result in the aggregated graph output.
 */
export interface GraphNodeResult {
  /**
   * Node status. Lower-case single-word string ("completed", "failed",
   * "cancelled", "interrupted"). Matches the Python side byte-for-byte per
   * the shared multi-agent dialect.
   */
  status: string
  /** Text output the node produced. Empty when the node failed before producing content. */
  output: string
  /** Node wall-clock time in milliseconds. */
  executionTimeMs: number
}

/**
 * Aggregated result of a graph invocation.
 *
 * Shape follows the shared multi-agent dialect: `status`, `output`, and
 * `executionTimeMs` are the common contract; `executionOrder` and `results`
 * are the graph-specific extensions.
 *
 * `output` is the concatenation of every terminal (leaf) node's text, joined
 * by a blank line — see `_multiagent-conventions.md` for the rationale.
 */
export interface GraphOutput {
  /**
   * Overall graph status. Lower-case single-word string ("completed",
   * "failed", "cancelled", "interrupted") matching the Python side.
   */
  status: string
  /** Concatenated text of every terminal (leaf) node's output. */
  output: string
  /** Node ids in the order they completed. */
  executionOrder: string[]
  /** Per-node results keyed by node id. */
  results: Record<string, GraphNodeResult>
  /** Total wall-clock time for the whole graph, in milliseconds. */
  executionTimeMs: number
  /** Allow indexing with string keys for JSONValue compatibility. */
  [key: string]: unknown
}
