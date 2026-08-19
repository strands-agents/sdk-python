import { normalizeError } from '../../errors.js'
import { BackgroundTaskNotFoundError } from '../errors.js'
import { isInProcessTaskTerminalStatus } from './record.js'
import type { InterruptStateData } from '../../interrupt.js'
import type {
  InProcessTaskDescriptor,
  InProcessTaskEngineOptions,
  InProcessTaskRecord,
  TaskExecutionOutcome,
} from './types.js'

// Node.js treats delays above the signed 32-bit limit as 1ms.
const MAX_TIMER_DELAY_MS = 2 ** 31 - 1
const DEFAULT_EXECUTION_FAILURE_MESSAGE = 'Background task execution failed'

interface ActiveExecution {
  readonly controller: AbortController
  timeout?: ReturnType<typeof setTimeout>
}

/** Runs and tracks bounded in-process tasks. @internal */
export class InProcessTaskEngine {
  private readonly _options: InProcessTaskEngineOptions
  private readonly _tasks = new Map<string, InProcessTaskRecord>()
  private readonly _queue = new Set<string>()
  private readonly _activeExecutions = new Map<string, ActiveExecution>()
  private readonly _idleWaiters = new Set<() => void>()

  /**
   * Creates an in-process task engine.
   *
   * @param options - Execution limits and callbacks supplied by the task manager.
   * @throws TypeError if the concurrency or timeout configuration is invalid.
   */
  constructor(options: InProcessTaskEngineOptions) {
    if (!Number.isSafeInteger(options.maxConcurrency) || options.maxConcurrency <= 0) {
      throw new TypeError(`maxConcurrency must be a positive finite integer, got ${options.maxConcurrency}`)
    }
    if (options.timeout !== Infinity) assertTimerDelay('timeout', options.timeout)
    this._options = options
  }

  /**
   * Submits a task for execution, returning an existing task when its idempotency key matches.
   *
   * @param admission - Task descriptor and optional idempotency key.
   * @returns The admitted or matching task.
   */
  submit(admission: {
    readonly descriptor: InProcessTaskDescriptor
    readonly idempotencyKey?: string
  }): InProcessTaskRecord {
    if (admission.idempotencyKey !== undefined) {
      const existing = [...this._tasks.values()].find((task) => task.idempotencyKey === admission.idempotencyKey)
      if (existing) return existing
    }
    const now = new Date().toISOString()
    const stored: InProcessTaskRecord = {
      taskId: globalThis.crypto.randomUUID(),
      ...(admission.idempotencyKey !== undefined && { idempotencyKey: admission.idempotencyKey }),
      descriptor: admission.descriptor,
      status: 'queued',
      createdAt: now,
      updatedAt: now,
    }
    this._tasks.set(stored.taskId, stored)
    this._notifyTaskUpdated(stored)
    this._enqueue(stored.taskId)
    return stored
  }

  /**
   * Gets a task by ID.
   *
   * @param taskId - ID of the task to retrieve.
   * @returns The task, or `undefined` when it does not exist.
   */
  get(taskId: string): InProcessTaskRecord | undefined {
    return this._tasks.get(taskId)
  }

  /**
   * Lists the tasks tracked by this engine.
   *
   * @returns The tracked tasks.
   */
  list(): readonly InProcessTaskRecord[] {
    return [...this._tasks.values()]
  }

  /**
   * Removes a terminal task.
   *
   * @param taskId - ID of the task to remove.
   * @throws Error if the task does not exist or has not reached a terminal state.
   */
  remove(taskId: string): void {
    const task = this._requireTask(taskId)
    if (!isInProcessTaskTerminalStatus(task.status)) {
      throw new Error(`Background task '${taskId}' cannot be removed before reaching a terminal status`)
    }
    this._tasks.delete(taskId)
  }

  /**
   * Cancels a non-terminal task and aborts its active execution.
   *
   * @param taskId - ID of the task to cancel.
   * @param options - Cancellation options passed to the active execution.
   * @returns The cancelled task, or the unchanged task if it was already terminal.
   * @throws Error if the task does not exist.
   */
  cancel(taskId: string, options: { readonly reason: string }): InProcessTaskRecord {
    const current = this._requireTask(taskId)
    if (isInProcessTaskTerminalStatus(current.status)) return current
    const task = this._updateTask(taskId, (record) => {
      record.status = 'cancelled'
      delete record.state
      return true
    })!
    this._queue.delete(taskId)
    const activeExecution = this._activeExecutions.get(taskId)
    if (activeExecution?.timeout) {
      clearTimeout(activeExecution.timeout)
    }
    activeExecution?.controller.abort(options.reason)
    this._wakeIdleWaiters()
    return task
  }

  /**
   * Waits until no tasks are queued or executing.
   *
   * @param options - Optional cancellation signal for the wait.
   * @returns A promise that resolves when the engine is idle.
   * @throws The cancellation signal's reason if the wait is aborted.
   */
  async waitForIdle(options?: { readonly cancelSignal?: AbortSignal }): Promise<void> {
    const cancelSignal = options?.cancelSignal
    while (this._queue.size > 0 || this._activeExecutions.size > 0) {
      if (cancelSignal?.aborted) throw cancelSignal.reason
      await new Promise<void>((resolve, reject) => {
        const onAbort = (): void => {
          this._idleWaiters.delete(onIdle)
          reject(cancelSignal!.reason)
        }
        const onIdle = (): void => {
          cancelSignal?.removeEventListener('abort', onAbort)
          resolve()
        }
        this._idleWaiters.add(onIdle)
        cancelSignal?.addEventListener('abort', onAbort, { once: true })
      })
    }
  }

  /**
   * Updates a paused task and queues it when the update marks it ready.
   *
   * @param taskId - ID of the paused task.
   * @param update - Applies input to the paused state and determines whether execution can resume.
   * @returns The updated task.
   * @throws Error if the task is not paused or has no state.
   */
  resume(
    taskId: string,
    update: (state: InterruptStateData) => {
      readonly state: InterruptStateData
      readonly ready: boolean
    }
  ): InProcessTaskRecord {
    const task = this._updateTask(taskId, (record) => {
      if (record.status !== 'paused') {
        throw new Error(`Background task '${taskId}' cannot be resumed: status is '${record.status}', not 'paused'`)
      }
      if (record.state === undefined) {
        throw new Error(`Background task '${taskId}' cannot be resumed: paused state is missing`)
      }
      const resumed = update(record.state)
      record.state = resumed.state
      if (resumed.ready) record.status = 'queued'
      return true
    })!
    if (task.status === 'queued') this._enqueue(taskId)
    return task
  }

  private _enqueue(taskId: string): void {
    if (this._activeExecutions.has(taskId)) return
    this._queue.add(taskId)
    this._startQueuedTasks()
  }

  private _startQueuedTasks(): void {
    while (this._activeExecutions.size < this._options.maxConcurrency && this._queue.size > 0) {
      const taskId = this._queue.values().next().value!
      this._queue.delete(taskId)
      const activeExecution: ActiveExecution = {
        controller: new AbortController(),
      }
      this._activeExecutions.set(taskId, activeExecution)
      void this._execute(taskId, activeExecution)
        .finally(() => {
          if (activeExecution.timeout) clearTimeout(activeExecution.timeout)
          this._activeExecutions.delete(taskId)
          if (this._tasks.get(taskId)?.status === 'queued') {
            this._queue.add(taskId)
          }
          this._wakeIdleWaiters()
          this._startQueuedTasks()
        })
        .catch(() => undefined)
    }
  }

  private async _execute(taskId: string, activeExecution: ActiveExecution): Promise<void> {
    const working = this._updateTask(taskId, (task) => {
      task.status = 'working'
      return true
    })!

    if (Number.isFinite(this._options.timeout)) {
      activeExecution.timeout = setTimeout(() => {
        delete activeExecution.timeout
        this._timeoutTask(taskId, activeExecution)
      }, this._options.timeout)
    }

    let outcome: TaskExecutionOutcome
    try {
      outcome = await this._options.execute({
        taskId,
        descriptor: working.descriptor,
        ...(working.state !== undefined && { state: working.state }),
        cancelSignal: activeExecution.controller.signal,
      })
    } catch (error) {
      outcome = {
        status: 'failed',
        failure: {
          type: 'executionError',
          message: normalizeError(error).message || DEFAULT_EXECUTION_FAILURE_MESSAGE,
        },
      }
    }
    this._finishOutcome(taskId, outcome)
  }

  private _finishOutcome(taskId: string, outcome: TaskExecutionOutcome): void {
    if (outcome.status === 'paused') {
      this._updateTask(taskId, (record) => {
        if (record.status !== 'working') return false
        record.status = 'paused'
        record.state = outcome.state
        return true
      })
      return
    }

    this._updateTask(taskId, (record) => {
      if (record.status !== 'working') return false
      if (outcome.status === 'failed') {
        record.status = 'failed'
        delete record.state
        record.failure = {
          type: outcome.failure.type,
          message: outcome.failure.message || DEFAULT_EXECUTION_FAILURE_MESSAGE,
        }
        if (outcome.result !== undefined) {
          record.result = outcome.result
        }
        return true
      }
      record.status = 'completed'
      delete record.state
      record.result = outcome.result
      return true
    })
  }

  private _timeoutTask(taskId: string, activeExecution: ActiveExecution): void {
    const reason = `Timed out after ${this._options.timeout}ms`
    const task = this._updateTask(taskId, (record) => {
      if (record.status !== 'working') return false
      record.status = 'failed'
      delete record.state
      record.failure = {
        type: 'timeout',
        message: reason,
      }
      return true
    })
    if (task) activeExecution.controller.abort(reason)
  }

  private _updateTask(taskId: string, update: (task: InProcessTaskRecord) => boolean): InProcessTaskRecord | undefined {
    const current = this._tasks.get(taskId)
    if (!current) throw new BackgroundTaskNotFoundError(taskId)

    const next = globalThis.structuredClone(current)
    if (!update(next)) return undefined

    next.updatedAt = new Date().toISOString()
    this._tasks.set(taskId, next)
    this._notifyTaskUpdated(next)

    return next
  }

  private _requireTask(taskId: string): InProcessTaskRecord {
    const task = this._tasks.get(taskId)
    if (!task) throw new BackgroundTaskNotFoundError(taskId)
    return task
  }

  private _notifyTaskUpdated(task: InProcessTaskRecord): void {
    const snapshot = globalThis.structuredClone(task)
    void Promise.resolve()
      .then(() => this._options.onTaskUpdated(snapshot))
      .catch(() => undefined)
  }

  private _wakeIdleWaiters(): void {
    for (const resolve of this._idleWaiters) resolve()
    this._idleWaiters.clear()
  }
}

function assertTimerDelay(name: string, value: number): void {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new TypeError(`${name} must be a positive finite integer, got ${value}`)
  }
  if (value > MAX_TIMER_DELAY_MS) {
    throw new TypeError(`${name} must be at most ${MAX_TIMER_DELAY_MS}ms, got ${value}`)
  }
}
