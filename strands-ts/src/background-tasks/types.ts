import type { Interrupt } from '../interrupt.js'
import type { ToolResultContentData } from '../types/messages.js'

/** Background task lifecycle status. @internal */
export type BackgroundTaskStatus = 'queued' | 'working' | 'input_required' | 'completed' | 'failed' | 'cancelled'

/** Background task failure category. @internal */
export type BackgroundTaskFailureType = 'toolError' | 'executionError' | 'timeout'

/** Read-only snapshot of a background task. @internal */
export interface BackgroundTask {
  /** Stable identifier for task inspection and cancellation. */
  readonly taskId: string
  /** Tool-use identifier from the original model request. */
  readonly toolUseId: string
  /** Tool name from the original model request. */
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

/**
 * Checks whether a background task status is terminal.
 *
 * @param status - Status to check.
 * @returns Whether the status is terminal.
 * @internal
 */
export function isTaskStatusTerminal(status: BackgroundTaskStatus): boolean {
  return status === 'completed' || status === 'failed' || status === 'cancelled'
}
