import type { ToolUseData } from '../hooks/events.js'
import type { Interrupt } from '../interrupt.js'
import type { Tool } from '../tools/tool.js'
import type { InvocationState } from '../types/agent.js'
import type { InterruptResponse } from '../types/interrupt.js'
import type { ToolResultContentData } from '../types/messages.js'
import type { BackgroundTaskFailureType, BackgroundTaskStatus } from './types.js'

/** Read-only snapshot of a background task. @internal */
export interface BackgroundTask {
  /** Stable identifier for task inspection and cancellation. */
  readonly taskId: string
  /** Tool-use identifier from the original model request. */
  readonly toolUseId: string
  /** Name of the executing tool. */
  readonly toolName: string
  /** Current task lifecycle status. */
  readonly status: BackgroundTaskStatus
  /** ISO timestamp recorded when the task was admitted. */
  readonly createdAt: string
  /** ISO timestamp recorded at the latest task state change. */
  readonly lastUpdatedAt: string
  /** Tool result when execution produced one. */
  readonly result?: {
    /** Tool-result content blocks. */
    readonly content: readonly ToolResultContentData[]
  }
  /** Task failure details. */
  readonly error?: {
    /** Failure category. */
    readonly type: BackgroundTaskFailureType
    /** Failure message. */
    readonly message: string
  }
  /** Unanswered interrupts when the task requires input. */
  readonly interrupts?: readonly Readonly<Interrupt>[]
}

/** Background task lifecycle operations used by an Agent integration. @internal */
export interface BackgroundTaskManager {
  /** Submits an approved tool call for background execution. */
  submitTask(toolUse: Readonly<ToolUseData>, invocationState: InvocationState, tool: Tool): Promise<BackgroundTask>
  /** Gets one task by its stable identifier. */
  getTask(taskId: string): Promise<BackgroundTask | undefined>
  /** Lists the tasks currently tracked by the manager. */
  listTasks(): Promise<readonly BackgroundTask[]>
  /** Requests cancellation of one task. */
  cancelTask(taskId: string): Promise<BackgroundTask>
  /** Waits until one task requires input or reaches a terminal state. */
  waitForTask(taskId: string): Promise<BackgroundTask>
  /** Waits until the manager has no queued or executing tasks. */
  waitForTasks(options?: { readonly timeout?: number }): Promise<void>
  /** Applies interrupt responses to one task that requires input. */
  resumeTask(taskId: string, responses: readonly InterruptResponse[]): Promise<BackgroundTask>
  /** Removes terminal tasks after their results have been consumed. */
  consumeTasks(taskIds: readonly string[]): Promise<void>
}
