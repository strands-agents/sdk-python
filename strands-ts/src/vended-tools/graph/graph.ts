/**
 * Graph tool: deterministic DAG orchestration over the SDK's Graph primitive.
 *
 * Shim over `Graph` in `@strands-agents/sdk/multiagent`. The caller agent
 * describes a DAG of sub-agents inline; this tool constructs a `Graph` from
 * those specs and invokes it, returning per-node text results.
 *
 * Design notes:
 * - **Shim only.** All orchestration lives in `multiagent/graph.ts`. This module
 *   validates the model-supplied topology, builds sub-agents, wires them into a
 *   `Graph`, and formats the result.
 * - **No edge conditions.** Conditional routing would require executing
 *   model-supplied code at every edge; not appropriate for a vended tool.
 * - **Tools are name-allow-listed** from the parent agent's registry. The model
 *   cannot conjure new tools inside a node — only reference tools the caller
 *   already has. Mirrors the `use_agent` / `swarm` convention.
 * - **Cycles are rejected** at validation time. `Graph` also requires reachability
 *   from source nodes; the tool detects cycles explicitly before construction so
 *   the failure message points at the actual problem.
 */

import { z } from 'zod'
import { Agent } from '../../agent/agent.js'
import { Graph } from '../../multiagent/graph.js'
import { AgentNode } from '../../multiagent/nodes.js'
import type { InvokableTool, Tool } from '../../tools/tool.js'
import { tool } from '../../tools/tool-factory.js'
import { TextBlock } from '../../types/messages.js'
import type { ContentBlock } from '../../types/messages.js'
import {
  DEFAULT_NODE_TIMEOUT_MS,
  DEFAULT_TIMEOUT_MS,
  GRAPH_DESCRIPTION,
  MAX_EDGES,
  MAX_ID_LENGTH,
  MAX_INITIAL_INPUT_LENGTH,
  MAX_NODES,
  MAX_STEPS,
  MAX_SYSTEM_PROMPT_LENGTH,
  MAX_TOOLS_PER_NODE,
  type GraphNodeResult,
  type GraphOutput,
} from './types.js'

/**
 * Shared multi-agent recursion depth counter — see `_multiagent-conventions.md`.
 * Every multi-agent vended tool (`use_agent`, `swarm`, `graph`, `a2a_client`)
 * reads/increments the same key so a chain crossing tool boundaries is still
 * capped.
 */
export const MULTIAGENT_DEPTH_KEY = 'multiagentDepth'
export const MAX_MULTIAGENT_DEPTH = 3

/**
 * Multi-agent tool names that must never appear in a child node's allow-list.
 * Depth-cap participation already blocks unbounded recursion, but the shared
 * convention (`_multiagent-conventions.md`) also mandates an explicit reject at
 * the tool boundary as defense-in-depth.
 */
const MULTIAGENT_TOOL_NAMES = new Set(['use_agent', 'swarm', 'graph', 'a2a_client'])

function currentDepth(invocationState: Record<string, unknown> | undefined): number {
  const raw = invocationState?.[MULTIAGENT_DEPTH_KEY]
  if (typeof raw !== 'number' || !Number.isInteger(raw) || raw < 0) return 0
  return raw
}

/**
 * Build a `cancelled` result. Spec (`_multiagent-conventions.md`): a cancelled
 * tool call returns `{status: 'cancelled'}` — it does not raise past the tool
 * boundary. A cancelled result reaching the loop is a signal to the parent,
 * not an exception to propagate.
 */
function cancelledOutput(startedAt: number, partial = ''): GraphOutput {
  return {
    status: 'cancelled',
    output: partial,
    executionOrder: [],
    results: {},
    executionTimeMs: Date.now() - startedAt,
  }
}

const nodeIdPattern = /^[A-Za-z0-9_-]+$/

const nodeSchema = z.object({
  id: z
    .string()
    .min(1, 'Node id must be non-empty.')
    .max(MAX_ID_LENGTH, `Node id must be at most ${MAX_ID_LENGTH} characters.`)
    .regex(nodeIdPattern, 'Node id must match [A-Za-z0-9_-].')
    .describe('Unique short identifier for the node.'),
  systemPrompt: z
    .string()
    .max(MAX_SYSTEM_PROMPT_LENGTH, `systemPrompt must be at most ${MAX_SYSTEM_PROMPT_LENGTH} characters.`)
    .optional()
    .describe("Optional system prompt for this node's sub-agent."),
  tools: z
    .array(z.string().min(1))
    .max(MAX_TOOLS_PER_NODE, `tools list must have at most ${MAX_TOOLS_PER_NODE} entries.`)
    .optional()
    .describe(
      "Tool names to expose to this node, drawn from the parent agent's tool " +
        'registry. Wildcards are not allowed; each tool must be listed by name.'
    ),
})

const edgeSchema = z.object({
  fromId: z.string().min(1).describe('Source node id.'),
  toId: z.string().min(1).describe('Target node id.'),
})

const graphInputSchema = z.object({
  nodes: z
    .array(nodeSchema)
    .min(1, "'nodes' must be a non-empty list.")
    .max(MAX_NODES, `Too many nodes (max ${MAX_NODES}).`)
    .describe('Nodes in the graph.'),
  edges: z
    .array(edgeSchema)
    .max(MAX_EDGES, `Too many edges (max ${MAX_EDGES}).`)
    .describe('Directed edges. Each entry declares that toId depends on fromId.'),
  initialInput: z
    .string()
    .max(MAX_INITIAL_INPUT_LENGTH, `initialInput must be at most ${MAX_INITIAL_INPUT_LENGTH} characters.`)
    .describe('The task passed to entry-point nodes (nodes with no incoming edges).'),
})

type NodeSpec = z.infer<typeof nodeSchema>
type EdgeSpec = z.infer<typeof edgeSchema>

/**
 * Verifies the graph has no cycles using Kahn's algorithm. Returns nothing on
 * success; throws when a cycle is detected. `Graph` is a DAG by design — cycle
 * detection at the tool boundary makes the failure message point at the
 * actual problem instead of a generic "unreachable node" error from the SDK.
 */
function assertNoCycle(nodes: NodeSpec[], edges: EdgeSpec[]): void {
  const ids = new Set(nodes.map((n) => n.id))
  const adjacency = new Map<string, string[]>()
  const inDegree = new Map<string, number>()
  for (const id of ids) inDegree.set(id, 0)

  for (const edge of edges) {
    if (!ids.has(edge.fromId)) {
      throw new Error(`Edge references unknown source node '${edge.fromId}'.`)
    }
    if (!ids.has(edge.toId)) {
      throw new Error(`Edge references unknown target node '${edge.toId}'.`)
    }
    if (edge.fromId === edge.toId) {
      throw new Error(`Self-loop on node '${edge.fromId}' is not allowed; the graph must be a DAG.`)
    }
    const targets = adjacency.get(edge.fromId) ?? []
    targets.push(edge.toId)
    adjacency.set(edge.fromId, targets)
    inDegree.set(edge.toId, (inDegree.get(edge.toId) ?? 0) + 1)
  }

  const queue: string[] = []
  for (const [id, deg] of inDegree.entries()) {
    if (deg === 0) queue.push(id)
  }
  let processed = 0
  while (queue.length > 0) {
    const current = queue.shift()!
    processed++
    for (const target of adjacency.get(current) ?? []) {
      const next = (inDegree.get(target) ?? 0) - 1
      inDegree.set(target, next)
      if (next === 0) queue.push(target)
    }
  }
  if (processed !== ids.size) {
    throw new Error('Graph contains a cycle; graphs must be acyclic (DAG).')
  }
}

/**
 * Assert unique ids in the flat `nodes` list. Zod validates each entry
 * individually but doesn't cross-check.
 */
function assertUniqueNodeIds(nodes: NodeSpec[]): void {
  const seen = new Set<string>()
  for (const node of nodes) {
    if (seen.has(node.id)) {
      throw new Error(`Duplicate node id '${node.id}'.`)
    }
    seen.add(node.id)
  }
}

/**
 * Resolve a node's tool allow-list against the parent registry. Rejects
 * wildcards, empty strings, and unknown names. Deduplicates.
 */
function resolveTools(names: string[] | undefined, parentTools: Map<string, Tool>, nodeId: string): Tool[] {
  if (!names || names.length === 0) return []
  const resolved: Tool[] = []
  const seen = new Set<string>()
  for (const raw of names) {
    if (raw === '*' || raw.includes('*')) {
      throw new Error(
        `Node '${nodeId}': tool '${raw}' looks like a wildcard. ` + 'List each tool by name; wildcards are not allowed.'
      )
    }
    // Reject multi-agent tools by name — defense-in-depth on top of the shared
    // depth cap. See `_multiagent-conventions.md`.
    if (MULTIAGENT_TOOL_NAMES.has(raw)) {
      throw new Error(
        `Node '${nodeId}': tool '${raw}' is a multi-agent tool and may not be ` +
          "used inside a graph node's allow-list."
      )
    }
    if (seen.has(raw)) continue
    seen.add(raw)
    const t = parentTools.get(raw)
    if (!t) {
      throw new Error(`Node '${nodeId}': tool '${raw}' is not registered on the parent agent.`)
    }
    resolved.push(t)
  }
  return resolved
}

/**
 * Coerce a node's ContentBlock output to a single text string. Non-text blocks
 * (images, tool uses, reasoning) are dropped — the tool's result shape is text
 * keyed by node id, which is what a downstream model will consume.
 */
function contentToText(blocks: ContentBlock[]): string {
  const parts: string[] = []
  for (const block of blocks) {
    if (block instanceof TextBlock) {
      parts.push(block.text)
    }
  }
  return parts.join('\n').trim()
}

/**
 * Options accepted by {@link makeGraph}. All fields are optional; each falls
 * back to the shared spec's default. Factory-only — none of these knobs are
 * surfaced to the model.
 */
export interface MakeGraphOptions {
  /** Tool name shown to the model. Defaults to `"graph"`. */
  name?: string
  /** Tool description shown to the model. */
  description?: string
  /** Shared multi-agent recursion cap. Defaults to {@link MAX_MULTIAGENT_DEPTH}. */
  maxDepth?: number
}

/**
 * Create a graph tool. Mirrors Python's `make_graph`. Use when you want a
 * different `name`, `description`, or `maxDepth` from the pre-built {@link graph}.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { makeGraph } from '@strands-agents/sdk/vended-tools/graph'
 *
 * const shallowGraph = makeGraph({ maxDepth: 2 })
 * const agent = new Agent({ tools: [shallowGraph] })
 * ```
 */
export function makeGraph(
  options: MakeGraphOptions = {}
): InvokableTool<z.infer<typeof graphInputSchema>, GraphOutput> {
  const name = options.name ?? 'graph'
  const description = options.description ?? GRAPH_DESCRIPTION
  const maxDepth = options.maxDepth ?? MAX_MULTIAGENT_DEPTH

  return tool({
    name,
    description,
    inputSchema: graphInputSchema,
    callback: async (input, context): Promise<GraphOutput> => {
      if (!context) {
        throw new Error('graph tool requires a tool context.')
      }

      const invocationState = context.invocationState as Record<string, unknown> | undefined
      const depth = currentDepth(invocationState)
      if (depth >= maxDepth) {
        throw new Error(
          `graph refused: multi-agent recursion depth cap of ${maxDepth} reached (current depth ${depth})`
        )
      }

      const { nodes: nodeSpecs, edges: edgeSpecs, initialInput } = input

      assertUniqueNodeIds(nodeSpecs)
      assertNoCycle(nodeSpecs, edgeSpecs)

      const startedAt = Date.now()

      if (context.agent.cancelSignal.aborted) {
        // Pre-flight cancellation: return the cancelled sentinel instead of
        // raising. Spec (`_multiagent-conventions.md`): the tool returns
        // `{status: 'cancelled'}`, does not raise past the tool boundary.
        return cancelledOutput(startedAt)
      }

      const parentToolMap = new Map<string, Tool>()
      for (const t of context.agent.toolRegistry.list()) {
        parentToolMap.set(t.name, t)
      }

      const parentModel = context.agent.model

      const agentNodes: AgentNode[] = nodeSpecs.map((spec) => {
        const nodeTools = resolveTools(spec.tools, parentToolMap, spec.id)
        const subAgent = new Agent({
          id: spec.id,
          model: parentModel,
          printer: false,
          ...(spec.systemPrompt !== undefined && { systemPrompt: spec.systemPrompt }),
          tools: nodeTools,
        })
        return new AgentNode({ agent: subAgent })
      })

      const g = new Graph({
        nodes: agentNodes,
        edges: edgeSpecs.map((e) => ({ source: e.fromId, target: e.toId })),
        timeout: DEFAULT_TIMEOUT_MS,
        nodeTimeout: DEFAULT_NODE_TIMEOUT_MS,
        maxSteps: MAX_STEPS,
      })

      const childInvocationState: Record<string, unknown> = { [MULTIAGENT_DEPTH_KEY]: depth + 1 }

      // Hard wall-clock ceiling. The SDK `Graph`'s own `timeout` is checked at
      // node boundaries, so an N-node chain of long-running nodes can silently
      // exceed the cap. Composing an `AbortSignal.timeout` with the parent's
      // cancel signal gives a strict upper bound that fires regardless of where
      // execution is inside the graph — parity with the Python side's
      // `asyncio.wait_for` guard.
      const hardCeiling = AbortSignal.timeout(DEFAULT_TIMEOUT_MS)
      const composedSignal = AbortSignal.any([context.agent.cancelSignal, hardCeiling])

      let result
      try {
        result = await g.invoke(initialInput, {
          cancelSignal: composedSignal,
          invocationState: childInvocationState,
        })
      } catch (err) {
        // The SDK's `Graph` throws when its `cancelSignal` aborts mid-flight
        // (see `graph.ts:343`). Map that back to a cancelled result — spec
        // forbids raising past the tool boundary. Anything else is a real
        // error and should propagate. The hard-ceiling `AbortSignal.timeout`
        // is treated the same way: no useful result once the cap fires.
        if (composedSignal.aborted) {
          return cancelledOutput(startedAt)
        }
        throw err
      }

      const results: Record<string, GraphNodeResult> = {}
      const executionOrder: string[] = []
      for (const nodeResult of result.results) {
        results[nodeResult.nodeId] = {
          // Lowercase the SDK enum value so the tool result matches the Python
          // side and the shared multi-agent dialect (see
          // `_multiagent-conventions.md`), which specifies single-word lowercase
          // status strings.
          status: nodeResult.status.toLowerCase(),
          output: nodeResult.error ? `error: ${nodeResult.error.message}` : contentToText(nodeResult.content),
          executionTimeMs: nodeResult.duration,
        }
        executionOrder.push(nodeResult.nodeId)
      }

      const terminalOutput = aggregateTerminalOutput(nodeSpecs, edgeSpecs, results)

      return {
        status: result.status.toLowerCase(),
        output: terminalOutput,
        executionOrder,
        results,
        executionTimeMs: result.duration,
      }
    },
  })
}

/**
 * Pre-built graph tool. Register on an `Agent` alongside any tools you want
 * the graph's nodes to be able to call.
 *
 * @example
 * ```typescript
 * import { Agent } from '@strands-agents/sdk'
 * import { graph } from '@strands-agents/sdk/vended-tools/graph'
 *
 * const agent = new Agent({ tools: [graph] })
 * ```
 */
export const graph = makeGraph()

/**
 * Concatenate every terminal (leaf) node's output text, joined by a blank
 * line, in graph declaration order. A graph without a single named sink
 * typically returns multiple final outputs; dropping any of them would
 * silently truncate the tool's answer.
 */
function aggregateTerminalOutput(
  nodes: NodeSpec[],
  edges: EdgeSpec[],
  results: Record<string, GraphNodeResult>
): string {
  const withOutgoing = new Set<string>(edges.map((e) => e.fromId))
  const parts: string[] = []
  for (const spec of nodes) {
    if (withOutgoing.has(spec.id)) continue
    const nodeResult = results[spec.id]
    if (!nodeResult) continue
    if (nodeResult.output) parts.push(nodeResult.output)
  }
  return parts.join('\n\n')
}
