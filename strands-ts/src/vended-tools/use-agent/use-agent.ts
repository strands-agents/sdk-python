/**
 * use_agent vended tool: delegate a task to a nested agent.
 *
 * The calling agent constructs a fresh Agent with the given system prompt
 * and an allowlisted subset of the parent's tools, then hands it a single
 * task. The nested agent's final text response is returned to the parent.
 *
 * This is a shim over Agent and its normal construction path; it does
 * not reinvent agent lifecycle. It only enforces the security surface of
 * runtime delegation (allowlist, size caps, recursion cap, cancellation
 * propagation).
 */

import { z } from 'zod'

import { Agent } from '../../agent/agent.js'
import { tool } from '../../tools/tool-factory.js'
import type { InvokableTool } from '../../tools/tool.js'
import type { UseAgentResult, UseAgentStatus } from './types.js'

const DEFAULT_MAX_SYSTEM_PROMPT_BYTES = 8 * 1024
const DEFAULT_MAX_TASK_BYTES = 32 * 1024
const DEFAULT_MAX_TOOL_ALLOWLIST = 64

export const MULTIAGENT_DEPTH_KEY = 'multiagentDepth'
export const MAX_MULTIAGENT_DEPTH = 3

const WILDCARD_TOOL_NAMES: ReadonlySet<string> = new Set(['*', '**', 'all', 'any', ''])

/**
 * Defense-in-depth: child agents cannot invoke multi-agent tools directly. A
 * developer who wants that must register the variants at construction time.
 */
const MULTIAGENT_TOOL_NAMES: ReadonlySet<string> = new Set(['use_agent', 'swarm', 'graph', 'a2a_client'])

const USE_AGENT_DESCRIPTION =
  'Delegate a single task to a nested agent that you construct at call time. ' +
  "You provide the child agent's systemPrompt, an explicit allowlist of tool " +
  'names to expose (drawn from your own tools), and the task itself. The child ' +
  'runs with the same model as you and a fresh conversation, and returns its ' +
  'final text response. Prefer this for scoped sub-tasks that benefit from a ' +
  'different system prompt or a narrower tool surface than your own.'

/**
 * Raised when the shared multi-agent recursion counter has already reached the
 * configured cap. Kept separate from a plain Error so callers that want to
 * catch the depth-cap case can do so without brittle string matching.
 */
export class MultiagentDepthExceededError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'MultiagentDepthExceededError'
  }
}

function byteLength(value: string): number {
  return new TextEncoder().encode(value).length
}

function validatePositiveIntCap(value: number, name: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || !Number.isInteger(value) || value <= 0) {
    throw new Error(`${name} must be a positive integer, got ${value}`)
  }
  return value
}

function validateBoundedString(value: unknown, name: string, maxBytes: number): string {
  if (typeof value !== 'string') {
    throw new Error(`${name} must be a string, got ${typeof value}`)
  }
  if (value.trim().length === 0) {
    throw new Error(`${name} must be non-empty`)
  }
  const size = byteLength(value)
  if (size > maxBytes) {
    throw new Error(`${name} exceeds size cap: ${size} bytes > ${maxBytes} bytes`)
  }
  return value
}

function validateToolAllowlist(tools: unknown, parentToolNames: Set<string>, maxEntries: number): string[] {
  if (tools === undefined || tools === null) return []
  if (!Array.isArray(tools)) {
    throw new Error(`tools must be an array of tool names, got ${typeof tools}`)
  }
  if (tools.length > maxEntries) {
    throw new Error(`tools allowlist exceeds cap of ${maxEntries} entries`)
  }

  const seen = new Set<string>()
  const resolved: string[] = []
  for (const entry of tools) {
    if (typeof entry !== 'string') {
      throw new Error(`tools entries must be strings, got ${typeof entry}`)
    }
    const stripped = entry.trim()
    if (WILDCARD_TOOL_NAMES.has(stripped.toLowerCase())) {
      throw new Error(`tools entry ${JSON.stringify(entry)} is a wildcard; every child tool must be named explicitly`)
    }
    if (MULTIAGENT_TOOL_NAMES.has(stripped)) {
      throw new Error(
        `tools entry ${JSON.stringify(stripped)} is a multi-agent tool and cannot be nested inside a child agent`
      )
    }
    if (seen.has(stripped)) continue
    if (!parentToolNames.has(stripped)) {
      throw new Error(`tools entry ${JSON.stringify(stripped)} is not present in the parent agent's tool registry`)
    }
    seen.add(stripped)
    resolved.push(stripped)
  }
  return resolved
}

function currentDepth(invocationState: Record<string, unknown> | undefined): number {
  const raw = invocationState?.[MULTIAGENT_DEPTH_KEY]
  if (typeof raw !== 'number' || !Number.isFinite(raw) || !Number.isInteger(raw) || raw < 0) return 0
  return raw
}

function mapStopReason(stopReason: string): UseAgentStatus {
  // Shared multi-agent dialect uses the lower-cased SDK Status enum values.
  // Any non-endTurn, non-cancelled, non-interrupt stop reason surfaces as
  // failed so the parent can distinguish a delivered delegation from one
  // that hit a policy or limit.
  if (stopReason === 'endTurn') return 'completed'
  if (stopReason === 'cancelled') return 'cancelled'
  if (stopReason === 'interrupt') return 'interrupted'
  return 'failed'
}

/**
 * Options for {@link makeUseAgent}. All caps default to the shared multi-agent
 * dialect values and are fixed at tool construction; the model cannot alter
 * them at call time.
 */
export interface MakeUseAgentOptions {
  /** Tool name. Defaults to `"use_agent"`. */
  name?: string
  /** Description shown to the model. */
  description?: string
  /**
   * Cap on the shared multi-agent recursion counter. Defaults to
   * {@link MAX_MULTIAGENT_DEPTH}.
   */
  maxDepth?: number
  /** UTF-8 byte cap on `systemPrompt`. */
  maxSystemPromptBytes?: number
  /** UTF-8 byte cap on `task`. */
  maxTaskBytes?: number
  /** Cap on the number of entries in the tool allowlist. */
  maxToolAllowlist?: number
}

/**
 * Build a use_agent vended tool with configurable safety caps.
 *
 * The default caps match the shared multi-agent dialect. They are exposed here
 * so callers can construct the tool with a tighter envelope at wire-up time;
 * the model cannot reach them at call time.
 */
export function makeUseAgent(options: MakeUseAgentOptions = {}): InvokableTool<unknown, UseAgentResult> {
  const {
    name = 'use_agent',
    description = USE_AGENT_DESCRIPTION,
    maxDepth = MAX_MULTIAGENT_DEPTH,
    maxSystemPromptBytes = DEFAULT_MAX_SYSTEM_PROMPT_BYTES,
    maxTaskBytes = DEFAULT_MAX_TASK_BYTES,
    maxToolAllowlist = DEFAULT_MAX_TOOL_ALLOWLIST,
  } = options

  validatePositiveIntCap(maxDepth, 'maxDepth')
  validatePositiveIntCap(maxSystemPromptBytes, 'maxSystemPromptBytes')
  validatePositiveIntCap(maxTaskBytes, 'maxTaskBytes')
  validatePositiveIntCap(maxToolAllowlist, 'maxToolAllowlist')

  const useAgentInputSchema = z
    .object({
      systemPrompt: z
        .string()
        .describe(`System prompt for the nested agent. Non-empty; capped at ${maxSystemPromptBytes} UTF-8 bytes.`),
      task: z.string().describe(`The task to hand the nested agent. Non-empty; capped at ${maxTaskBytes} UTF-8 bytes.`),
      tools: z
        .array(z.string())
        .optional()
        .describe(
          'Exact-name allowlist of tools to expose to the nested agent. Every entry must be a tool that exists ' +
            "in your own tool registry. Wildcards and multi-agent tool names ('use_agent', 'swarm', 'graph', " +
            "'a2a_client') are rejected."
        ),
    })
    .strict()

  return tool({
    name,
    description,
    inputSchema: useAgentInputSchema,
    callback: async (input, context): Promise<UseAgentResult> => {
      if (!context) {
        throw new Error('use_agent requires a tool context')
      }

      const parentAgent = context.agent
      const parentInvocationState = (context.invocationState ?? {}) as Record<string, unknown>

      const depth = currentDepth(parentInvocationState)
      if (depth >= maxDepth) {
        throw new MultiagentDepthExceededError(
          `use_agent refused: recursion depth cap of ${maxDepth} reached (current depth ${depth})`
        )
      }

      const systemPrompt = validateBoundedString(input.systemPrompt, 'systemPrompt', maxSystemPromptBytes)
      const task = validateBoundedString(input.task, 'task', maxTaskBytes)

      const parentToolNames = new Set(parentAgent.toolRegistry.list().map((t) => t.name))
      const toolNames = validateToolAllowlist(input.tools, parentToolNames, maxToolAllowlist)
      const childTools = toolNames.map((toolName) => parentAgent.toolRegistry.get(toolName)!)

      // Child inherits the parent's model instance. Shipped providers are
      // stateless on `this` per invocation, so sharing is safe today; if a
      // future provider retains per-invocation state on the instance, this
      // assumption needs revisiting (e.g. by cloning before delegation).
      //
      // LocalAgent doesn't expose `name` on the shared interface, but the
      // concrete Agent class always sets one (falls back to a default). Read
      // it defensively so this stays compatible with any LocalAgent.
      const parentName = (parentAgent as { name?: string }).name ?? 'agent'
      const child = new Agent({
        model: parentAgent.model,
        systemPrompt,
        tools: childTools,
        name: `${parentName}::use_agent`,
        printer: false,
      })

      // Preserve the parent's invocationState so tracing / telemetry /
      // per-run keys flow through to the child. Only override the shared
      // depth counter with the incremented value.
      const childInvocationState: Record<string, unknown> = {
        ...parentInvocationState,
        [MULTIAGENT_DEPTH_KEY]: depth + 1,
      }

      const cancelSignal = parentAgent.cancelSignal as AbortSignal | undefined

      const start = Date.now()
      const result = await child.invoke(task, {
        invocationState: childInvocationState,
        ...(cancelSignal !== undefined && { cancelSignal }),
      })
      const executionTimeMs = Date.now() - start

      // The child agent surfaces cancellation as stopReason 'cancelled'. Per
      // the shared multi-agent conventions, the TypeScript side re-raises an
      // AbortError so callers can distinguish cancellation from other failures
      // via `error.name === 'AbortError'`, matching the sibling http-request
      // tool. This is a deliberate divergence from Python (which returns
      // `{status: "cancelled"}`); the conventions doc records the asymmetry.
      if (result.stopReason === 'cancelled') {
        const reason = cancelSignal?.reason
        if (reason instanceof Error) {
          throw reason
        }
        throw new DOMException('use_agent cancelled by parent agent', 'AbortError')
      }

      return {
        status: mapStopReason(result.stopReason),
        output: result.toString(),
        executionTimeMs,
      }
    },
  })
}

/**
 * Default use_agent tool with the shared multi-agent dialect's caps. Use
 * {@link makeUseAgent} to construct one with a custom envelope.
 */
export const useAgent = makeUseAgent()
