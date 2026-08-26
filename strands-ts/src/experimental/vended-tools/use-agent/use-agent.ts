import { z } from 'zod'

import { Agent, inheritPreToolHooksSymbol } from '../../../agent/agent.js'
import { DefaultNotConfiguredError, normalizeError } from '../../../errors.js'
import { BeforeToolCallEvent } from '../../../hooks/events.js'
import { Interrupt, InterruptError } from '../../../interrupt.js'
import { JsonBlock, ToolResultBlock } from '../../../types/messages.js'
import { InterruptResponseContent } from '../../../types/interrupt.js'
import { Tool } from '../../../tools/tool.js'
import type { ToolContext, ToolStreamGenerator } from '../../../tools/tool.js'
import type { ToolSpec } from '../../../tools/types.js'
import { zodSchemaToJsonSchema } from '../../../tools/zod-utils.js'
import type { JSONValue } from '../../../types/json.js'
import type { AgentResult, InvocationState } from '../../../types/agent.js'

const USE_AGENT_TOOL_NAME = 'use_agent'
const DEFAULT_LIMITS = {
  turns: 50,
  totalTokens: 100_000,
  depth: 3,
  timeoutSeconds: 300,
} as const satisfies Required<UseAgentLimits>

/**
 * Developer-controlled limits for the experimental `use_agent` tool.
 */
export interface UseAgentLimits {
  /** Maximum child turns. Defaults to and cannot exceed 50. */
  turns?: number
  /** Soft cumulative child-token threshold. A model turn can overshoot it. Defaults to and cannot exceed 100000. */
  totalTokens?: number
  /** Maximum nested `use_agent` depth. Defaults to and cannot exceed 3. */
  depth?: number
  /** Maximum child execution time in seconds. Defaults to and cannot exceed 300. */
  timeoutSeconds?: number
}

/**
 * Options for {@link makeUseAgent}.
 */
export interface MakeUseAgentOptions {
  /** Child execution limits. */
  limits?: UseAgentLimits
}

const useAgentInputSchema = z
  .object({
    task: z.string().min(1).describe('Task for the child agent.'),
    instructions: z.string().min(1).optional().describe('Optional child-specific instructions.'),
    tools: z.array(z.string().min(1)).optional().describe('Exact parent tool names to grant. Omit for no tools.'),
  })
  .strict()

interface UseAgentResult {
  status: 'completed' | 'failed' | 'cancelled'
  error?: string
  output?: string
}

interface PendingChild {
  child: Agent
  task: string
  parentInterruptState: object
  remainingMs: number
  remainingTurns: number
  remainingTotalTokens: number
  invocationState: InvocationState
  interruptGeneration: number
  interrupts: { childId: string; outwardId: string }[]
}

class UseAgentTool extends Tool {
  readonly name = USE_AGENT_TOOL_NAME
  readonly description =
    'Runs a task in a fresh child agent. The child receives only the exact parent tools named in tools; omit tools for no child tools.'
  readonly toolSpec: ToolSpec

  private readonly _limits: Required<UseAgentLimits>
  private readonly _pending = new WeakMap<Agent, Map<string, PendingChild>>()
  private readonly _depths = new WeakMap<Agent, number>()

  constructor(options: MakeUseAgentOptions) {
    super()
    this._limits = resolveLimits(options.limits)
    this.toolSpec = {
      name: this.name,
      description: this.description,
      inputSchema: zodSchemaToJsonSchema(useAgentInputSchema),
    }
  }

  // eslint-disable-next-line require-yield
  async *stream(context: ToolContext): ToolStreamGenerator {
    const parent = context.agent
    if (!(parent instanceof Agent)) {
      return resultBlock(context.toolUse.toolUseId, {
        status: 'failed',
        error: 'use_agent requires a local Agent context',
      })
    }

    const pending = this._pending.get(parent)?.get(context.toolUse.toolUseId)
    const timeoutMs = pending?.remainingMs ?? this._limits.timeoutSeconds * 1000
    const deadline = Date.now() + timeoutMs
    const cancelSignal = AbortSignal.any([context.cancelSignal, AbortSignal.timeout(timeoutMs)])
    try {
      const interruptState = parent._interruptState
      const restored =
        (pending && pending.parentInterruptState !== interruptState) ||
        (!pending &&
          Object.keys(interruptState.interrupts).some((id) => id.startsWith(`${context.toolUse.toolUseId}:`)))
      if (restored) {
        throw new Error('use_agent cannot resume an interrupted child after the parent or tool instance was restored')
      }
      const childState =
        pending ?? (await this._createChild(parent, context.toolUse.input, context.invocationState, cancelSignal))
      const stopReason =
        pending && childState.remainingTurns <= 0
          ? 'limitTurns'
          : pending && childState.remainingTotalTokens <= 0
            ? 'limitTotalTokens'
            : undefined
      if (stopReason) {
        this._pending.get(parent)?.delete(context.toolUse.toolUseId)
        return resultBlock(context.toolUse.toolUseId, stoppedResult(stopReason))
      }

      const prompt = pending ? this._interruptResponses(parent, pending) : childState.task
      const result = await waitForCancellation(
        childState.child.invoke(prompt, {
          invocationState: childState.invocationState,
          cancelSignal,
          limits: {
            turns: childState.remainingTurns,
            totalTokens: childState.remainingTotalTokens,
          },
        }),
        cancelSignal
      )
      if (result.stopReason === 'interrupt' && result.interrupts?.length) {
        const invocation = result.metrics?.latestAgentInvocation
        if (invocation) {
          childState.remainingTurns -= invocation.cycles.length
          childState.remainingTotalTokens -= invocation.usage.totalTokens
        }
        childState.remainingMs = Math.max(0, deadline - Date.now())
        childState.interruptGeneration += 1
        childState.interrupts = result.interrupts.map((interrupt) => ({
          childId: interrupt.id,
          outwardId: `${context.toolUse.toolUseId}:${childState.interruptGeneration}:${interrupt.id}`,
        }))
        const pendingChildren = this._pending.get(parent) ?? new Map()
        pendingChildren.set(context.toolUse.toolUseId, childState)
        this._pending.set(parent, pendingChildren)
        throw new InterruptError(
          result.interrupts.map(
            (interrupt, index) =>
              new Interrupt({
                ...interrupt.toJSON(),
                id: childState.interrupts[index]!.outwardId,
              })
          )
        )
      }

      this._pending.get(parent)?.delete(context.toolUse.toolUseId)
      return resultBlock(context.toolUse.toolUseId, terminalResult(result))
    } catch (error) {
      if (error instanceof InterruptError) {
        throw error
      }
      this._pending.get(parent)?.delete(context.toolUse.toolUseId)
      const normalized = normalizeError(error)
      const cancelled =
        context.cancelSignal.aborted || (normalized instanceof DOMException && normalized.name === 'AbortError')
      const message =
        normalized instanceof DOMException && normalized.name === 'TimeoutError'
          ? 'use_agent child exceeded its execution timeout'
          : normalized.message
      return resultBlock(context.toolUse.toolUseId, {
        status: cancelled ? 'cancelled' : 'failed',
        error: message,
      })
    }
  }

  private async _createChild(
    parent: Agent,
    rawInput: unknown,
    parentState: InvocationState,
    cancelSignal: AbortSignal
  ): Promise<PendingChild> {
    const input = useAgentInputSchema.parse(rawInput)
    const depth = this._depths.get(parent) ?? 0
    if (depth >= this._limits.depth) {
      throw new Error(`use_agent recursion depth limit of ${this._limits.depth} reached`)
    }
    if (hasProviderNativeTools(parent)) {
      throw new Error(
        'use_agent is not supported with provider-native model tools; ' +
          'register governed SDK tools on the parent agent instead'
      )
    }

    const selectedTools = resolveTools(parent, input.tools ?? [], this)
    const sandbox = resolveParentSandbox(parent)
    const child = new Agent({
      model: parent.model,
      ...(input.instructions !== undefined && { systemPrompt: input.instructions }),
      ...(sandbox !== undefined && { sandbox }),
      printer: false,
    })
    this._depths.set(child, depth + 1)
    child[inheritPreToolHooksSymbol](parent)
    await waitForCancellation(child.initialize(), cancelSignal)
    // Agent initialization may vend sandbox tools; exact grants must replace them.
    child.toolRegistry.clear()
    child.toolRegistry.add(selectedTools)
    const allowedTools = new Set(selectedTools)
    child.addHook(
      BeforeToolCallEvent,
      (event) => {
        const selectedTool = event.selectedTool ?? child.toolRegistry.get(event.toolUse.name)
        if (!event.cancel && selectedTool && !allowedTools.has(selectedTool)) {
          event.cancel = 'use_agent blocked a tool outside the child grant set'
        }
      },
      { order: Number.POSITIVE_INFINITY }
    )
    return {
      child,
      task: input.task,
      parentInterruptState: parent._interruptState,
      remainingMs: this._limits.timeoutSeconds * 1000,
      remainingTurns: this._limits.turns,
      remainingTotalTokens: this._limits.totalTokens,
      invocationState: { ...parentState },
      interruptGeneration: 0,
      interrupts: [],
    }
  }

  private _interruptResponses(parent: Agent, pending: PendingChild): InterruptResponseContent[] {
    const parentInterrupts = parent._interruptState.interrupts
    return pending.interrupts.map(({ childId, outwardId }) => {
      const interrupt = parentInterrupts[outwardId]
      if (!interrupt || interrupt.response === undefined) {
        throw new Error(`use_agent interrupt '${outwardId}' has no response`)
      }
      return new InterruptResponseContent({ interruptId: childId, response: interrupt.response })
    })
  }
}

/**
 * Creates an experimental governed `use_agent` tool.
 *
 * This tool is subject to change in future revisions without notice.
 * @param options - Developer-controlled child execution limits
 * @returns A streaming tool that creates bounded child agents
 * @throws TypeError if a child execution limit is outside its supported range
 */
export function makeUseAgent(options: MakeUseAgentOptions = {}): Tool {
  return new UseAgentTool(options)
}

/**
 * Default experimental `use_agent` tool.
 */
export const useAgent = makeUseAgent()

function stoppedResult(stopReason: AgentResult['stopReason']): UseAgentResult {
  return {
    status: 'failed',
    error: `use_agent child stopped before completion: ${stopReason}`,
  }
}

function terminalResult(result: AgentResult): UseAgentResult {
  if (result.stopReason === 'cancelled') {
    return { status: 'cancelled', error: 'use_agent child was cancelled' }
  }
  if (result.stopReason !== 'endTurn' && result.stopReason !== 'stopSequence') {
    return stoppedResult(result.stopReason)
  }
  return {
    status: 'completed',
    output: result.toString(),
  }
}

function resolveParentSandbox(parent: Agent): Agent['sandbox'] | undefined {
  try {
    return parent.sandbox
  } catch (error) {
    if (error instanceof DefaultNotConfiguredError) return undefined
    throw error
  }
}

function resolveTools(parent: Agent, requestedNames: string[], executingTool: Tool): Tool[] {
  return [...new Set(requestedNames)].map((name) => {
    const selectedTool = parent.toolRegistry.get(name)
    if (!selectedTool) {
      throw new Error(`Tool '${name}' was not found on the parent agent`)
    }
    if (name === USE_AGENT_TOOL_NAME && selectedTool !== executingTool) {
      throw new Error('A child can receive only the currently executing use_agent tool')
    }
    return selectedTool
  })
}

function resolveLimits(limits: UseAgentLimits | undefined): Required<UseAgentLimits> {
  const resolved = { ...DEFAULT_LIMITS, ...limits }
  for (const name of Object.keys(DEFAULT_LIMITS) as (keyof UseAgentLimits)[]) {
    const value = resolved[name]
    const maximum = DEFAULT_LIMITS[name]
    if (!Number.isSafeInteger(value) || value < 1 || value > maximum) {
      throw new TypeError(`limits.${name} must be a safe integer between 1 and ${maximum}, got ${value}`)
    }
  }
  return resolved
}

function hasProviderNativeTools(parent: Agent): boolean {
  const config = parent.model.getConfig()
  if ('builtInTools' in config && Array.isArray(config.builtInTools) && config.builtInTools.length > 0) {
    return true
  }
  if (!('params' in config) || !config.params || typeof config.params !== 'object') {
    return false
  }
  const params = config.params
  return 'tools' in params && Array.isArray(params.tools) && params.tools.length > 0
}

function resultBlock(toolUseId: string, result: UseAgentResult): ToolResultBlock {
  return new ToolResultBlock({
    toolUseId,
    status: result.status === 'completed' ? 'success' : 'error',
    content: [new JsonBlock({ json: result as unknown as JSONValue })],
  })
}

async function waitForCancellation<T>(operation: Promise<T>, cancelSignal: AbortSignal): Promise<T> {
  let onAbort!: () => void
  const cancellation = new Promise<never>((_, reject) => {
    onAbort = (): void => {
      reject(cancelSignal.reason ?? new DOMException('use_agent child was cancelled', 'AbortError'))
    }
    if (cancelSignal.aborted) {
      onAbort()
    } else {
      cancelSignal.addEventListener('abort', onAbort, { once: true })
    }
  })
  try {
    return await Promise.race([operation, cancellation])
  } finally {
    cancelSignal.removeEventListener('abort', onAbort)
  }
}
