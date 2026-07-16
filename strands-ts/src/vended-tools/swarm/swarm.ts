/**
 * Swarm vended tool.
 *
 * Thin shim over {@link Swarm}: validates a spec of child agent definitions,
 * constructs the child agents (all inheriting the parent's model), builds a
 * `Swarm` with fixed safety caps, invokes it, and normalizes the result into
 * the shared multi-agent result dialect (see `_multiagent-conventions.md`).
 */

import { z } from 'zod'
import { Agent } from '../../agent/agent.js'
import { Swarm } from '../../multiagent/swarm.js'
import { Status } from '../../multiagent/state.js'
import type { MultiAgentResult } from '../../multiagent/state.js'
import { tool } from '../../tools/tool-factory.js'
import type { InvokableTool, Tool } from '../../tools/tool.js'
import { TextBlock } from '../../types/messages.js'
import type { SwarmToolStatus } from './types.js'
import { SWARM_TOOL_DESCRIPTION, type SwarmToolResult } from './types.js'

/**
 * Default cap on child-agent count. Not model-configurable.
 *
 * The `Swarm` class itself doesn't cap agent count. We pick 5 because past
 * that point a swarm is almost always the wrong tool — a `Graph` or hand-authored
 * orchestration models the shape better.
 */
const DEFAULT_MAX_AGENTS = 5

/** Default total wall-clock ceiling passed to `Swarm.timeout`, in milliseconds. */
const DEFAULT_EXECUTION_TIMEOUT_MS = 300_000

/** Default per-node wall-clock ceiling passed to `Swarm.nodeTimeout`, in milliseconds. */
const DEFAULT_NODE_TIMEOUT_MS = 120_000

/** Default cap on total node executions (and, by proxy, handoffs). */
const DEFAULT_MAX_STEPS = 10

/**
 * Shared with the sibling multi-agent tools. See `_multiagent-conventions.md`.
 */
export const MULTIAGENT_DEPTH_KEY = 'multiagentDepth'
export const MAX_MULTIAGENT_DEPTH = 3

/** Default size caps from the shared multi-agent dialect (UTF-8 bytes). */
const DEFAULT_MAX_INITIAL_INPUT_BYTES = 32 * 1024
const DEFAULT_MAX_SYSTEM_PROMPT_BYTES = 8 * 1024

/** `name` regex and max-char cap from the shared multi-agent dialect. */
const NAME_REGEX = /^[a-zA-Z_][a-zA-Z0-9_]{0,63}$/
const MAX_NAME_CHARS = 64

/** Max entries in a child spec's `tools` allowlist (shared multi-agent dialect). */
const MAX_TOOLS_PER_SPEC = 64

/**
 * Raised when the shared multi-agent recursion counter has already reached the
 * configured cap. Kept separate from a plain `Error` so callers that want to
 * catch the depth-cap case can do so without brittle string matching.
 */
export class MultiagentDepthExceededError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'MultiagentDepthExceededError'
  }
}

/**
 * Names of the multi-agent tools themselves. Children may not list any of
 * these in their `tools` allowlist — that would let a compromised model bypass
 * the shared depth counter by having a child re-invoke `swarm`/`graph`/etc.
 */
const MULTIAGENT_TOOL_NAMES: ReadonlySet<string> = new Set(['use_agent', 'swarm', 'graph', 'a2a_client'])

function byteLength(value: string): number {
  return new TextEncoder().encode(value).length
}

function currentDepth(invocationState: Record<string, unknown> | undefined): number {
  const raw = invocationState?.[MULTIAGENT_DEPTH_KEY]
  if (typeof raw !== 'number' || !Number.isInteger(raw) || raw < 0) return 0
  return raw
}

/**
 * Options for {@link makeSwarm}. All caps default to the shared multi-agent
 * dialect values and are fixed at tool construction — the model cannot alter
 * them at call time.
 */
export interface MakeSwarmOptions {
  /** Tool name. Defaults to `"swarm"`. */
  name?: string
  /** Description shown to the model. Defaults to {@link SWARM_TOOL_DESCRIPTION}. */
  description?: string
  /** Maximum number of child agents allowed in a single invocation. */
  maxAgents?: number
  /** Total wall-clock timeout for the swarm, in milliseconds. */
  executionTimeoutMs?: number
  /** Per-node wall-clock timeout, in milliseconds. */
  nodeTimeoutMs?: number
  /** Cap on total node executions (also caps handoffs). */
  maxSteps?: number
  /**
   * Cap on the shared multi-agent recursion depth. Defaults to
   * {@link MAX_MULTIAGENT_DEPTH}.
   */
  maxMultiagentDepth?: number
  /** UTF-8 byte cap on `initialInput`. */
  maxInitialInputBytes?: number
  /** UTF-8 byte cap on each spec's `systemPrompt`. */
  maxSystemPromptBytes?: number
}

/**
 * Build a swarm vended tool with configurable safety caps.
 *
 * The default caps match the shared multi-agent dialect. They are exposed here
 * so callers can construct swarms with tighter envelopes at wire-up time; the
 * model cannot reach them at call time, exactly like `maxIterations` on a
 * plain `Agent`.
 */
export function makeSwarm(options: MakeSwarmOptions = {}): InvokableTool<unknown, SwarmToolResult> {
  const {
    name = 'swarm',
    description = SWARM_TOOL_DESCRIPTION,
    maxAgents = DEFAULT_MAX_AGENTS,
    executionTimeoutMs = DEFAULT_EXECUTION_TIMEOUT_MS,
    nodeTimeoutMs = DEFAULT_NODE_TIMEOUT_MS,
    maxSteps = DEFAULT_MAX_STEPS,
    maxMultiagentDepth = MAX_MULTIAGENT_DEPTH,
    maxInitialInputBytes = DEFAULT_MAX_INITIAL_INPUT_BYTES,
    maxSystemPromptBytes = DEFAULT_MAX_SYSTEM_PROMPT_BYTES,
  } = options

  // Both the top-level object and each agent spec are `.strict()` so unknown
  // fields throw at the boundary. Defense-in-depth against the model inventing
  // knobs like `model`, `hooks`, or `credentials` at either level.
  const agentSpecSchema = z
    .object({
      name: z
        .string()
        .min(1)
        .max(MAX_NAME_CHARS)
        .regex(NAME_REGEX, 'name must match [a-zA-Z_][a-zA-Z0-9_]{0,63}')
        .describe('Unique identifier for this sub-agent. Referenced during handoffs.'),
      systemPrompt: z.string().describe('System prompt for this sub-agent.'),
      description: z
        .string()
        .optional()
        .describe('Human-readable description shown to sibling agents so they know when to hand off to this one.'),
      tools: z
        .array(z.string())
        .max(MAX_TOOLS_PER_SPEC)
        .describe(
          `Tool names to expose to this sub-agent (may be empty). Must be a subset of your own registered tools. No wildcards. Max ${MAX_TOOLS_PER_SPEC} entries.`
        ),
    })
    .strict()

  const swarmInputSchema = z
    .object({
      agents: z
        .array(agentSpecSchema)
        .min(1)
        .max(maxAgents)
        .describe(`List of sub-agent specs (at least 1, at most ${maxAgents}).`),
      initialInput: z.string().min(1).describe('Task to hand to the entry agent.'),
      entryAgent: z
        .string()
        .optional()
        .describe('Name of the agent to start with. Defaults to the first entry in `agents`.'),
    })
    .strict()

  type AgentSpec = z.infer<typeof agentSpecSchema>

  return tool({
    name,
    description,
    inputSchema: swarmInputSchema,
    callback: async (input, context): Promise<SwarmToolResult> => {
      if (!context) {
        throw new Error('Tool context is required for swarm operations')
      }

      // Depth cap first — shared across the multi-agent tool family via
      // `invocationState`. Cheapest way to shut down runaway delegation chains.
      const parentInvocationState = (context.invocationState ?? {}) as Record<string, unknown>
      const depth = currentDepth(parentInvocationState)
      if (depth >= maxMultiagentDepth) {
        throw new MultiagentDepthExceededError(
          `swarm refused: multi-agent recursion depth cap of ${maxMultiagentDepth} reached (current depth ${depth})`
        )
      }

      // Size caps at the tool boundary, before we construct anything.
      const initialSize = byteLength(input.initialInput)
      if (initialSize > maxInitialInputBytes) {
        throw new Error(`initialInput exceeds size cap: ${initialSize} bytes > ${maxInitialInputBytes} bytes`)
      }
      for (const spec of input.agents) {
        const size = byteLength(spec.systemPrompt)
        if (size > maxSystemPromptBytes) {
          throw new Error(
            `agent '${spec.name}' systemPrompt exceeds size cap: ${size} bytes > ${maxSystemPromptBytes} bytes`
          )
        }
      }

      const specs = input.agents as AgentSpec[]

      // Duplicate names would trip the SDK's own check inside `Swarm._resolveNodes`.
      // We front-run it for a tighter error message.
      const seen = new Set<string>()
      for (const spec of specs) {
        if (seen.has(spec.name)) {
          throw new Error(`agent name '${spec.name}' is a duplicate; names must be unique`)
        }
        seen.add(spec.name)
      }

      const parentAgent = context.agent
      const childAgents = specs.map((spec) => buildChildAgent(spec, parentAgent))

      let start: string | undefined
      if (input.entryAgent !== undefined) {
        const match = childAgents.find((a) => a.name === input.entryAgent)
        if (!match) {
          const available = childAgents.map((a) => a.name)
          throw new Error(
            `entryAgent '${input.entryAgent}' not in agents list. Available: ${JSON.stringify(available)}`
          )
        }
        start = match.name
      }

      const childSwarm = new Swarm({
        nodes: childAgents,
        ...(start !== undefined && { start }),
        maxSteps,
        timeout: executionTimeoutMs,
        nodeTimeout: nodeTimeoutMs,
      })

      // Preserve the parent's invocationState so tracing / telemetry / per-run
      // keys flow through to the child. Only override the shared depth counter
      // with the incremented value.
      const childInvocationState: Record<string, unknown> = {
        ...parentInvocationState,
        [MULTIAGENT_DEPTH_KEY]: depth + 1,
      }

      const cancelSignal = parentAgent.cancelSignal as AbortSignal | undefined
      let result: MultiAgentResult
      try {
        result = await childSwarm.invoke(input.initialInput, {
          ...(cancelSignal !== undefined && { cancelSignal }),
          invocationState: childInvocationState,
        })
      } catch (error) {
        // The underlying `Swarm` throws when its external cancel signal fires.
        // Translate into an `AbortError` at the tool boundary so callers can
        // distinguish cancellation from other failures via
        // `error.name === 'AbortError'`, matching the sibling `http-request`
        // tool. This is a deliberate TS-side deviation from the shared
        // multi-agent conventions doc, which specifies a returned
        // `{status: "cancelled"}`; the doc records the deviation.
        if (cancelSignal?.aborted) {
          throw cancelSignal.reason ?? new DOMException('swarm cancelled by parent agent', 'AbortError')
        }
        throw error
      }

      return mapResult(result)
    },
  })
}

/**
 * Default swarm tool with the shared multi-agent dialect's caps. Use
 * {@link makeSwarm} to construct one with a custom envelope.
 */
export const swarm = makeSwarm()

/**
 * Structural shape of a validated child spec — matches
 * `z.infer<typeof agentSpecSchema>` inside `makeSwarm`. Broken out here so
 * `buildChildAgent` doesn't need access to the closure-scoped Zod schema.
 */
interface ChildSpec {
  name: string
  systemPrompt: string
  tools: string[]
  description?: string | undefined
}

/**
 * Build one child Agent from a validated spec.
 *
 * Child agents inherit the parent's model. Tools are name-resolved against the
 * parent's registered tools; unknown names throw so the model can't hoist tools
 * the caller didn't hand it.
 */
function buildChildAgent(spec: ChildSpec, parentAgent: unknown): Agent {
  const parent = parentAgent as { model: unknown; toolRegistry: { get(name: string): Tool | undefined } }

  const childTools: Tool[] = []
  for (const toolName of spec.tools) {
    // Defense-in-depth: even if the parent has a multi-agent tool registered,
    // the model cannot grant it to a child. Otherwise a child could re-enter
    // `swarm`/`graph`/etc. and bypass the shared depth counter.
    if (MULTIAGENT_TOOL_NAMES.has(toolName)) {
      throw new Error(
        `agent '${spec.name}' requested multi-agent tool '${toolName}'; multi-agent tools may not be listed in a child spec's tools`
      )
    }
    const resolved = parent.toolRegistry.get(toolName)
    if (!resolved) {
      throw new Error(`agent '${spec.name}' requested unknown tool '${toolName}' (not in parent tool registry)`)
    }
    childTools.push(resolved)
  }

  return new Agent({
    name: spec.name,
    model: parent.model as Agent['model'],
    tools: childTools,
    // The parent already renders its own event stream; keep the child quiet.
    printer: false,
    systemPrompt: spec.systemPrompt,
    ...(spec.description !== undefined && { description: spec.description }),
  })
}

/**
 * Map a MultiAgentResult from `Swarm` into the shared multi-agent result dialect.
 *
 * Translates the SDK's execution vocabulary (`COMPLETED`/`FAILED`/`INTERRUPTED`
 * /`CANCELLED`) into the shared dialect's `success`/`error`/`cancelled` so
 * downstream models see a stable contract across every multi-agent tool.
 */
function mapResult(result: MultiAgentResult): SwarmToolResult {
  const nodeHistory = result.results.map((r) => r.nodeId)

  const output = extractOutput(result)

  const usage = result.usage
  return {
    status: mapStatus(result.status),
    output,
    nodeHistory,
    executionCount: result.results.length,
    executionTimeMs: result.duration,
    usage: {
      inputTokens: usage?.inputTokens ?? 0,
      outputTokens: usage?.outputTokens ?? 0,
      totalTokens: usage?.totalTokens ?? 0,
    },
  }
}

/**
 * Map the SDK's execution `Status` onto the shared multi-agent result dialect.
 *
 * `COMPLETED` maps to `success`, `INTERRUPTED` / `CANCELLED` map to
 * `cancelled`, and everything else (including `FAILED` and any unrecognized
 * state) maps to `error`. Better to surface an unknown state as an error than
 * silently paper it over.
 */
function mapStatus(status: Status): SwarmToolStatus {
  switch (status) {
    case Status.COMPLETED:
      return 'success'
    case Status.INTERRUPTED:
    case Status.CANCELLED:
      return 'cancelled'
    default:
      return 'error'
  }
}

/**
 * Pull the final text from a MultiAgentResult.
 *
 * `MultiAgentResult.content` is populated from the terminal node by `Swarm`.
 * Fall back to an empty string if the swarm never produced content (e.g.
 * failed before any agent could speak).
 */
function extractOutput(result: MultiAgentResult): string {
  if (result.status === Status.FAILED && result.results.length === 0) {
    return ''
  }

  const parts: string[] = []
  for (const block of result.content) {
    if (block instanceof TextBlock) {
      parts.push(block.text)
    }
  }
  return parts.join('')
}
