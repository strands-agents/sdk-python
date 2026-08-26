import type { Agent } from '../../agent/agent.js'
import type { ToolUseData } from '../../hooks/events.js'
import { InterruptError, InterruptState } from '../../interrupt.js'
import type { InvocationState } from '../../types/agent.js'
import { InterruptResponseContent, type InterruptParams, type InterruptResponse } from '../../types/interrupt.js'
import type { JSONValue } from '../../types/json.js'
import { TextBlock, type ToolResultBlock } from '../../types/messages.js'
import type { Tool, ToolContext } from '../../tools/tool.js'
import { BackgroundTaskNotFoundError } from '../errors.js'
import { InProcessTaskEngine } from './engine.js'
import {
  type InProcessTaskExecutionContext,
  type InProcessTaskExecutionOutcome,
  type InProcessTaskRecord,
} from './types.js'
import type { BackgroundTask, BackgroundTaskManager } from '../task-manager.js'
import type { BackgroundTaskStatus } from '../types.js'

const DEFAULT_MAX_CONCURRENCY = 4
const MAX_TIMER_DELAY_MS = 2 ** 31 - 1

interface ToolExecution {
  readonly toolUse: ToolUseData
  readonly invocationState: InvocationState
  readonly tool: Tool
}

type InProcessTaskWaiter = (task: BackgroundTask) => void
type ExecuteTool = (tool: Tool, context: ToolContext) => Promise<ToolResultBlock>

/** Configures in-process background task execution. @internal */
export interface InProcessTaskManagerConfig {
  /** Maximum number of physically executing background tasks. Defaults to `4`. */
  readonly maxConcurrency?: number
  /** Per-execution timeout in milliseconds. Defaults to `Infinity`. */
  readonly timeout?: number
}

/** Executes approved tool calls as in-process background tasks. @internal */
export class InProcessTaskManager implements BackgroundTaskManager {
  private readonly _agent: Agent
  private readonly _executeTool: ExecuteTool
  private readonly _engine: InProcessTaskEngine
  private readonly _executions = new Map<string, ToolExecution>()
  private readonly _taskWaiters = new Map<string, Set<InProcessTaskWaiter>>()

  /**
   * Creates an in-process background task manager.
   *
   * @param agent - Agent whose registered tools execute background work.
   * @param executeTool - Executes an approved tool through the Agent's tool pipeline.
   * @param config - Execution limits.
   */
  constructor(agent: Agent, executeTool: ExecuteTool, config: InProcessTaskManagerConfig = {}) {
    this._agent = agent
    this._executeTool = executeTool
    this._engine = new InProcessTaskEngine({
      maxConcurrency: config.maxConcurrency ?? DEFAULT_MAX_CONCURRENCY,
      timeout: config.timeout ?? Infinity,
      execute: (context): Promise<InProcessTaskExecutionOutcome> => this._executeToolTask(context),
      onTaskUpdated: (record): void => {
        if (isTerminalTaskStatus(record.status)) {
          this._executions.delete(record.invocationStateId)
        }
        if (isTaskWaitComplete(record.status)) this._notifyTaskWaiters(record)
      },
    })
  }

  /** {@inheritDoc BackgroundTaskManager.submitTask} */
  async submitTask(
    toolUse: Readonly<ToolUseData>,
    invocationState: InvocationState,
    tool: Tool
  ): Promise<BackgroundTask> {
    const invocationStateId = globalThis.crypto.randomUUID()
    this._executions.set(invocationStateId, {
      toolUse: globalThis.structuredClone(toolUse),
      invocationState,
      tool,
    })
    try {
      const record = this._engine.submit({
        toolName: tool.name,
        toolUseId: toolUse.toolUseId,
        invocationStateId,
      })
      return toBackgroundTask(record)
    } catch (error) {
      this._executions.delete(invocationStateId)
      throw error
    }
  }

  /** {@inheritDoc BackgroundTaskManager.getTask} */
  async getTask(taskId: string): Promise<BackgroundTask | undefined> {
    const record = this._engine.get(taskId)
    return record ? toBackgroundTask(record) : undefined
  }

  /** {@inheritDoc BackgroundTaskManager.listTasks} */
  async listTasks(): Promise<readonly BackgroundTask[]> {
    return this._engine.list().map(toBackgroundTask)
  }

  /** {@inheritDoc BackgroundTaskManager.cancelTask} */
  async cancelTask(taskId: string): Promise<BackgroundTask> {
    return toBackgroundTask(this._engine.cancel(taskId, { reason: 'Cancellation requested' }))
  }

  /** {@inheritDoc BackgroundTaskManager.waitForTask} */
  async waitForTask(taskId: string): Promise<BackgroundTask> {
    const current = this._engine.get(taskId)
    if (!current) throw new BackgroundTaskNotFoundError(taskId)
    if (isTaskWaitComplete(current.status)) return toBackgroundTask(current)

    return new Promise<BackgroundTask>((resolve) => {
      const waiters = this._taskWaiters.get(taskId) ?? new Set<InProcessTaskWaiter>()
      const onTaskUpdated = (task: BackgroundTask): void => {
        waiters.delete(onTaskUpdated)
        if (waiters.size === 0) this._taskWaiters.delete(taskId)
        resolve(task)
      }
      waiters.add(onTaskUpdated)
      this._taskWaiters.set(taskId, waiters)
    })
  }

  /** {@inheritDoc BackgroundTaskManager.waitForTasks} */
  async waitForTasks(options?: { readonly timeout?: number }): Promise<void> {
    const timeout = options?.timeout
    if (timeout !== undefined && (!Number.isSafeInteger(timeout) || timeout <= 0 || timeout > MAX_TIMER_DELAY_MS)) {
      throw new TypeError(
        `wait timeout must be a positive integer no greater than ${MAX_TIMER_DELAY_MS}, got ${timeout}`
      )
    }
    await this._engine.waitForIdle(timeout === undefined ? undefined : { cancelSignal: AbortSignal.timeout(timeout) })
  }

  /** {@inheritDoc BackgroundTaskManager.resumeTask} */
  async resumeTask(taskId: string, responses: readonly InterruptResponse[]): Promise<BackgroundTask> {
    return toBackgroundTask(
      this._engine.resume(taskId, (state) => {
        const taskState = InterruptState.fromJSON(state)
        taskState.resume(
          responses.map(
            (response) =>
              new InterruptResponseContent({
                interruptId: response.interruptId,
                response: response.response,
              })
          )
        )
        return {
          state: taskState.toJSON(),
          ready: taskState.getUnansweredInterrupts().length === 0,
        }
      })
    )
  }

  /** {@inheritDoc BackgroundTaskManager.consumeTasks} */
  async consumeTasks(taskIds: readonly string[]): Promise<void> {
    const uniqueTaskIds = new Set(taskIds)
    for (const taskId of uniqueTaskIds) {
      const task = this._engine.get(taskId)
      if (!task) throw new BackgroundTaskNotFoundError(taskId)
      if (!isTerminalTaskStatus(task.status)) {
        throw new Error(`Background task '${taskId}' cannot be consumed before reaching a terminal status`)
      }
    }
    for (const taskId of uniqueTaskIds) {
      this._engine.remove(taskId)
    }
  }

  private async _executeToolTask(context: InProcessTaskExecutionContext): Promise<InProcessTaskExecutionOutcome> {
    const execution = this._executions.get(context.invocationStateId)
    if (!execution) throw new Error('Background task live execution state is unavailable')
    const interruptState = context.state ? InterruptState.fromJSON(context.state) : new InterruptState()
    try {
      return toolTaskOutcome(
        await this._executeTool(execution.tool, {
          agent: this._agent,
          invocationState: execution.invocationState,
          cancelSignal: context.cancelSignal,
          toolUse: execution.toolUse,
          interrupt: <T = JSONValue>(params: InterruptParams): T =>
            interruptTool<T>(interruptState, context.taskId, params),
        })
      )
    } catch (error) {
      if (error instanceof InterruptError) {
        for (const interrupt of error.interrupts) interruptState.registerInterrupt(interrupt)
        interruptState.activate()
        return { status: 'input_required', state: interruptState.toJSON() }
      }
      throw error
    }
  }

  private _notifyTaskWaiters(record: InProcessTaskRecord): void {
    const waiters = this._taskWaiters.get(record.taskId)
    if (!waiters) return
    const task = toBackgroundTask(record)
    for (const waiter of waiters) waiter(task)
  }
}

function toolTaskOutcome(resultBlock: ToolResultBlock): InProcessTaskExecutionOutcome {
  const result = resultBlock.toJSON().toolResult
  if (resultBlock.status === 'success') return { status: 'completed', result }
  return {
    status: 'failed',
    failure: {
      type: 'toolError',
      message:
        resultBlock.error?.message ??
        resultBlock.content.find((content): content is TextBlock => content instanceof TextBlock)?.text ??
        'Tool returned an error without a message',
    },
    result,
  }
}

function interruptTool<T>(state: InterruptState, taskId: string, params: InterruptParams): T {
  const interrupt = state.getOrCreateInterrupt(
    `tool:${taskId}:${params.name}`,
    params.name,
    params.reason,
    params.response,
    'tool'
  )
  if (interrupt.response !== undefined) return interrupt.response as T
  throw new InterruptError(interrupt)
}

function toBackgroundTask(record: InProcessTaskRecord): BackgroundTask {
  const interrupts = record.state ? InterruptState.fromJSON(record.state).getUnansweredInterrupts() : []
  return {
    taskId: record.taskId,
    toolUseId: record.toolUseId,
    toolName: record.toolName,
    status: record.status,
    createdAt: record.createdAt,
    lastUpdatedAt: record.lastUpdatedAt,
    ...(record.result && {
      result: { content: record.result.content },
    }),
    ...(record.failure && {
      error: {
        type: record.failure.type,
        message: record.failure.message,
      },
    }),
    ...(interrupts.length > 0 && { interrupts }),
  }
}

function isTaskWaitComplete(status: BackgroundTaskStatus): boolean {
  return status === 'input_required' || isTerminalTaskStatus(status)
}

function isTerminalTaskStatus(status: BackgroundTaskStatus): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled'
}
