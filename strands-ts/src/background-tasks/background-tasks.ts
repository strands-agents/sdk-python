import { z } from 'zod'
import type { Agent } from '../agent/agent.js'
import { InitializedEvent, type ToolUseData } from '../hooks/events.js'
import { HookOrder } from '../hooks/types.js'
import { logger } from '../logging/logger.js'
import type { Plugin } from '../plugins/plugin.js'
import { STRUCTURED_OUTPUT_TOOL_NAME } from '../tools/structured-output-tool.js'
import { tool } from '../tools/tool-factory.js'
import type { Tool } from '../tools/tool.js'
import { TextBlock, ToolResultBlock, ToolUseBlock } from '../types/messages.js'
import type { InvocationState, LocalAgent } from '../types/agent.js'
import { deepCopy, type JSONValue } from '../types/json.js'
import type { ToolSpec } from '../tools/types.js'
import type { SpanContext } from '@opentelemetry/api'
import { BACKGROUND_RESULT_TOOL_NAME } from './delivery.js'
import { BackgroundTaskNotFoundError } from './errors.js'
import { InProcessTaskManager } from './in-process-task-manager.js'
import { addBackgroundSelection, stripBackgroundSelection } from './schema.js'
import type { TaskManager, ToolCallSubmission } from './task-manager.js'
import type { BackgroundTasksConfig } from './types.js'

const MANAGE_TOOL_NAME = 'strands_manage_background_task'
type BackgroundMode = 'never' | 'agentic' | 'always'

export class BackgroundTasks implements Plugin {
  readonly name = 'strands:background-tasks'

  private readonly _config: BackgroundTasksConfig & {
    readonly policy: Readonly<Record<string, BackgroundMode>>
  }
  private readonly _managers = new WeakMap<Agent, TaskManager<ToolCallSubmission>>()
  private readonly _warnedWildcardTools = new Set<string>()
  private readonly _manageTool: Tool

  constructor(config: BackgroundTasksConfig = {}) {
    const resolved = resolvePolicy(config)
    this._config = {
      policy: resolved.policy,
      ...(config.waitForCompletion !== undefined && { waitForCompletion: config.waitForCompletion }),
      ...(config.maxConcurrency !== undefined && { maxConcurrency: config.maxConcurrency }),
      ...(config.timeout !== undefined && { timeout: config.timeout }),
    }
    this._manageTool = tool({
      name: MANAGE_TOOL_NAME,
      description:
        'Inspect or cancel a background task. Completed results are delivered automatically; do not poll with this tool.',
      inputSchema: z.object({
        mode: z.enum(['get', 'cancel']).describe('Whether to inspect or cancel the background task.'),
        taskId: z.string().min(1).describe('The background task ID returned when the task was admitted.'),
      }),
      callback: async (input, context): Promise<JSONValue> => {
        if (!context) throw new Error('Background task management requires tool context')
        const manager = this._requireManager(context.agent)
        try {
          if (input.mode === 'get') {
            const task = await manager.getTask(input.taskId)
            if (!task) throw new BackgroundTaskNotFoundError(input.taskId)
            return deepCopy(task)
          }
          const task = await manager.cancelTask(input.taskId)
          return { taskId: task.taskId, status: task.status }
        } catch (error) {
          throw new Error(modelSafeErrorMessage(error, 'Background task management failed'), { cause: error })
        }
      },
    })
  }

  getTools(): Tool[] {
    return [this._manageTool]
  }

  initAgent(localAgent: LocalAgent): void {
    const agent = localAgent as Agent
    const manager = new InProcessTaskManager(agent, this._config)
    this._managers.set(agent, manager)
    manager.registerHooks()
    agent.addHook(
      InitializedEvent,
      async () => {
        this._validateCurrentTools(agent)
        await manager.initialize()
      },
      { order: HookOrder.SDK_LAST }
    )
  }

  /** @internal */
  _appStateLoaded(agent: Agent): void {
    this._managers.get(agent)?.appStateLoaded()
  }

  /** @internal */
  _validateReservedToolNames(tools: readonly Tool[]): void {
    for (const toolInstance of tools) {
      this._validateReservedToolName(toolInstance)
    }
  }

  /** @internal */
  _transformToolSpecs(tools: readonly Tool[], toolSpecs: readonly ToolSpec[]): readonly ToolSpec[] {
    this._validateReservedToolNames(tools)
    const toolsByName = new Map(tools.map((toolInstance) => [toolInstance.name, toolInstance]))
    return toolSpecs.map((toolSpec) => {
      const toolInstance = toolsByName.get(toolSpec.name)
      const policy = this._resolvePolicy(toolSpec.name)
      if (!toolInstance || !policy) return toolSpec
      if (isFrameworkTool(toolInstance)) {
        if (policy.exact && policy.mode !== 'never') {
          throw new TypeError(`Tool '${toolSpec.name}' is framework-owned and cannot run in the background`)
        }
        return toolSpec
      }
      if (policy.mode !== 'agentic') return toolSpec

      const transformed = addBackgroundSelection(toolSpec)
      if (transformed.compatible) return transformed.toolSpec
      if (policy.exact) {
        throw new TypeError(`Tool '${toolSpec.name}' cannot use agentic background selection: ${transformed.reason}`)
      }
      this._warnWildcardSkip(toolSpec.name, transformed.reason)
      return toolSpec
    })
  }

  /** @internal */
  _routeToolCall(context: {
    readonly toolUseBlock: ToolUseBlock
    readonly tool: Tool | undefined
  }):
    | { readonly kind: 'execute'; readonly input: JSONValue }
    | { readonly kind: 'background'; readonly input: JSONValue }
    | { readonly kind: 'result'; readonly result: ToolResultBlock } {
    if (context.toolUseBlock.name === BACKGROUND_RESULT_TOOL_NAME) {
      return this._routingError(
        context.toolUseBlock,
        'This tool is reserved for Strands. Do not call it. Background task results are delivered automatically.'
      )
    }
    const policy = this._resolvePolicy(context.toolUseBlock.name)
    if (!policy || policy.mode === 'never') {
      return { kind: 'execute', input: context.toolUseBlock.input }
    }

    if (context.tool && isFrameworkTool(context.tool)) {
      if (policy.exact) {
        return this._routingError(
          context.toolUseBlock,
          `Tool '${context.toolUseBlock.name}' is framework-owned and cannot run in the background`
        )
      }
      return { kind: 'execute', input: context.toolUseBlock.input }
    }

    let routedInput = context.toolUseBlock.input
    let selected = policy.mode === 'always'
    if (policy.mode === 'agentic') {
      const compatibility = context.tool ? addBackgroundSelection(context.tool.toolSpec) : undefined
      if (compatibility && !compatibility.compatible) {
        if (policy.exact) {
          return this._routingError(
            context.toolUseBlock,
            `Tool '${context.toolUseBlock.name}' cannot run in the background: ${compatibility.reason}`
          )
        }
        this._warnWildcardSkip(context.toolUseBlock.name, compatibility.reason)
        return { kind: 'execute', input: context.toolUseBlock.input }
      }
      if (!compatibility?.compatible) {
        return { kind: 'execute', input: context.toolUseBlock.input }
      }

      try {
        const stripped = stripBackgroundSelection(context.toolUseBlock.input)
        selected = stripped.selected === true
        routedInput = stripped.input
      } catch (error) {
        return this._routingError(context.toolUseBlock, error instanceof Error ? error.message : String(error))
      }
    }

    if (!selected) {
      return { kind: 'execute', input: routedInput }
    }
    if (!context.tool) {
      return this._routingError(
        context.toolUseBlock,
        `Tool '${context.toolUseBlock.name}' is not registered and cannot be admitted`
      )
    }
    return { kind: 'background', input: routedInput }
  }

  /** @internal */
  async _submitApprovedToolCall(
    agent: Agent,
    context: {
      readonly originalToolUseBlock: ToolUseBlock
      readonly toolUse: ToolUseData
      readonly effectiveTool: Tool | undefined
      readonly invocationState: InvocationState
      readonly passId: string
      readonly originSpanContext?: SpanContext
    }
  ): Promise<ToolResultBlock> {
    try {
      this._validateApprovedToolCall(agent, {
        originalToolUseId: context.originalToolUseBlock.toolUseId,
        effectiveTool: context.effectiveTool,
        toolUse: context.toolUse,
      })
    } catch (error) {
      return this._routingError(context.originalToolUseBlock, error instanceof Error ? error.message : String(error))
        .result
    }

    try {
      const task = await this._requireManager(agent).submitTask({
        kind: 'toolCall',
        toolName: context.effectiveTool!.name,
        originalToolUseId: context.originalToolUseBlock.toolUseId,
        input: context.toolUse.input,
        invocationState: context.invocationState,
        passId: context.passId,
        ...(context.originSpanContext && { originSpanContext: context.originSpanContext }),
      })
      return new ToolResultBlock({
        toolUseId: context.originalToolUseBlock.toolUseId,
        status: 'success',
        content: [new TextBlock(renderDispatchAcknowledgement(task.taskId, task.toolName))],
      })
    } catch (error) {
      return this._routingError(
        context.originalToolUseBlock,
        modelSafeErrorMessage(error, 'Background task admission failed')
      ).result
    }
  }

  private _validateApprovedToolCall(
    agent: Agent,
    context: {
      readonly originalToolUseId: string
      readonly effectiveTool: Tool | undefined
      readonly toolUse: { readonly name: string; readonly toolUseId: string }
    }
  ): void {
    if (
      context.toolUse.name === BACKGROUND_RESULT_TOOL_NAME ||
      context.effectiveTool?.name === BACKGROUND_RESULT_TOOL_NAME
    ) {
      throw new Error('The Background Tasks delivery tool name is reserved')
    }
    if (context.toolUse.toolUseId !== context.originalToolUseId) {
      throw new Error('Background task hooks cannot change the original tool-use ID')
    }
    if (!context.effectiveTool) {
      throw new Error(`Background task tool '${context.toolUse.name}' is not registered`)
    }
    if (agent.toolRegistry.get(context.effectiveTool.name) !== context.effectiveTool) {
      throw new Error(`Background task tool '${context.effectiveTool.name}' must be registered on the Agent`)
    }
    if (isFrameworkTool(context.effectiveTool)) {
      throw new Error(`Framework-owned tool '${context.effectiveTool.name}' cannot run in the background`)
    }
    const policy = this._resolvePolicy(context.effectiveTool.name)
    if (!policy || policy.mode === 'never') {
      throw new Error(`Tool '${context.effectiveTool.name}' is forbidden by background task policy`)
    }
    if (policy.mode === 'agentic') {
      const compatibility = addBackgroundSelection(context.effectiveTool.toolSpec)
      if (!compatibility.compatible) {
        throw new Error(`Tool '${context.effectiveTool.name}' cannot run in the background: ${compatibility.reason}`)
      }
    }
  }

  private _validateCurrentTools(agent: Agent): void {
    this._validateReservedToolNames(agent.tools)
    for (const toolInstance of agent.tools) {
      const policy = this._resolvePolicy(toolInstance.name)
      if (!policy?.exact) continue
      if (isFrameworkTool(toolInstance) && policy.mode !== 'never') {
        throw new TypeError(`Tool '${toolInstance.name}' is framework-owned and cannot run in the background`)
      }
      if (policy.mode === 'agentic') {
        const compatibility = addBackgroundSelection(toolInstance.toolSpec)
        if (!compatibility.compatible) {
          throw new TypeError(
            `Tool '${toolInstance.name}' cannot use agentic background selection: ${compatibility.reason}`
          )
        }
      }
    }
  }

  private _validateReservedToolName(toolInstance: Tool): void {
    if (toolInstance.name === MANAGE_TOOL_NAME && toolInstance !== this._manageTool) {
      throw new TypeError(`Tool name '${MANAGE_TOOL_NAME}' is reserved for Background Tasks management`)
    }
    if (toolInstance.name === BACKGROUND_RESULT_TOOL_NAME) {
      throw new TypeError(`Tool name '${BACKGROUND_RESULT_TOOL_NAME}' is reserved for Background Tasks delivery`)
    }
  }

  private _resolvePolicy(toolName: string): { readonly mode: BackgroundMode; readonly exact: boolean } | undefined {
    if (Object.prototype.hasOwnProperty.call(this._config.policy, toolName)) {
      return { mode: this._config.policy[toolName]!, exact: true }
    }
    if (Object.prototype.hasOwnProperty.call(this._config.policy, '*')) {
      return { mode: this._config.policy['*']!, exact: false }
    }
    return undefined
  }

  private _warnWildcardSkip(toolName: string, reason: string): void {
    if (this._warnedWildcardTools.has(toolName)) return
    this._warnedWildcardTools.add(toolName)
    logger.warn(`tool_name=<${toolName}>, reason=<${reason}> | wildcard background policy skipped incompatible tool`)
  }

  private _routingError(
    toolUseBlock: ToolUseBlock,
    message: string
  ): { readonly kind: 'result'; readonly result: ToolResultBlock } {
    const error = new Error(message)
    return {
      kind: 'result',
      result: new ToolResultBlock({
        toolUseId: toolUseBlock.toolUseId,
        status: 'error',
        content: [new TextBlock(message)],
        error,
      }),
    }
  }

  /** @internal */
  _requireManager(agent: LocalAgent): TaskManager<ToolCallSubmission> {
    const manager = this._managers.get(agent as Agent)
    if (!manager) {
      throw new Error('BackgroundTasks is not initialized for this Agent')
    }
    return manager
  }
}

function resolvePolicy(config: BackgroundTasksConfig): { readonly policy: Readonly<Record<string, BackgroundMode>> } {
  if (!config || typeof config !== 'object') throw new TypeError('BackgroundTasks config is required')

  const policy: Record<string, BackgroundMode> = {}
  const configuredSelectors = new Map<string, Tool>()
  const hasExplicitWildcard = config.always?.includes('*') === true || config.never?.includes('*') === true
  const assignments = [
    ['agentic', config.agentic ?? (hasExplicitWildcard ? [] : ['*'])],
    ['always', config.always ?? []],
    ['never', config.never ?? []],
  ] as const

  for (const [mode, selectors] of assignments) {
    if (!Array.isArray(selectors)) {
      throw new TypeError(`BackgroundTasks ${mode} must be an array`)
    }
    for (const selector of selectors) {
      const toolName = selectorName(selector, mode)
      if (selector !== '*') {
        const existingTool = configuredSelectors.get(toolName)
        if (existingTool && existingTool !== selector) {
          throw new TypeError(`BackgroundTasks policy contains multiple Tool instances named '${toolName}'`)
        }
        configuredSelectors.set(toolName, selector)
      }
      const existingMode = policy[toolName]
      if (existingMode !== undefined && existingMode !== mode) {
        throw new TypeError(`Tool '${toolName}' cannot be configured as both '${existingMode}' and '${mode}'`)
      }
      policy[toolName] = mode
    }
  }

  return { policy: Object.freeze(policy) }
}

function selectorName(selector: Tool | '*', mode: BackgroundMode): string {
  if (selector === '*') return selector
  if (typeof selector === 'string' || !selector || typeof selector !== 'object') {
    throw new TypeError(`BackgroundTasks ${mode} entries must be Tool instances or '*'`)
  }
  if (typeof selector.name !== 'string' || selector.name.length === 0) {
    throw new TypeError(`BackgroundTasks ${mode} tool name must be a non-empty string`)
  }
  const toolName = selector.name
  if (toolName === BACKGROUND_RESULT_TOOL_NAME) {
    throw new TypeError(`Tool name '${BACKGROUND_RESULT_TOOL_NAME}' is reserved for Background Tasks delivery`)
  }
  return toolName
}

function isFrameworkTool(toolInstance: Tool): boolean {
  return toolInstance.name === MANAGE_TOOL_NAME || toolInstance.name === STRUCTURED_OUTPUT_TOOL_NAME
}

function renderDispatchAcknowledgement(taskId: string, toolName: string): string {
  return [
    'Background task dispatched.',
    '',
    `Task ID: ${taskId}`,
    `Tool: ${toolName}`,
    'Status: queued',
    '',
    'The task is running in the background. Continue without waiting or polling.',
    'The final result will be delivered automatically when the task completes.',
  ].join('\n')
}

function modelSafeErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof BackgroundTaskNotFoundError) {
    return error.message
  }
  logger.warn(`error=<${error}> | ${fallback.toLowerCase()}`)
  return fallback
}
