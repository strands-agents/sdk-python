import { normalizeError } from '../../errors.js'
import { BackgroundTaskNotFoundError } from '../errors.js'
import { isInProcessTaskTerminalStatus, validateStoredInProcessTask } from './record.js'
import type { InProcessTaskEngineConfig, StoredInProcessTask, TaskExecutionOutcome } from './types.js'

// Node.js treats delays above the signed 32-bit limit as 1ms.
const MAX_TIMER_DELAY_MS = 2 ** 31 - 1
const DEFAULT_EXECUTION_FAILURE_TYPE = 'executionError'
const DEFAULT_EXECUTION_FAILURE_MESSAGE = 'Background task execution failed'

interface ActiveExecution {
  readonly controller: AbortController
  timeout?: ReturnType<typeof setTimeout>
}

interface TaskWaiter<Descriptor, Result, State> {
  resolve(task: StoredInProcessTask<Descriptor, Result, State>): void
  reject(error: unknown): void
}

/** Bounded in-process task execution. @internal */
export class InProcessTaskEngine<Descriptor, Result, State> {
  private readonly _config: InProcessTaskEngineConfig<Descriptor, Result, State>
  private readonly _tasks = new Map<string, StoredInProcessTask<Descriptor, Result, State>>()
  private readonly _queue = new Set<string>()
  private readonly _activeExecutions = new Map<string, ActiveExecution>()
  private readonly _taskWaiters = new Map<string, Set<TaskWaiter<Descriptor, Result, State>>>()
  private readonly _idleWaiters = new Set<() => void>()
  private _initialized = false
  private _closed = false
  private _shutdown: Promise<void> | undefined
  private _failure: { readonly error: unknown } | undefined

  constructor(config: InProcessTaskEngineConfig<Descriptor, Result, State>) {
    if (!Number.isSafeInteger(config.maxConcurrency) || config.maxConcurrency <= 0) {
      throw new TypeError(`maxConcurrency must be a positive finite integer, got ${config.maxConcurrency}`)
    }
    if (config.timeout !== Infinity) assertTimerDelay('timeout', config.timeout)
    this._config = config
  }

  initialize(restoredTasks: readonly StoredInProcessTask<Descriptor, Result, State>[] = []): void {
    if (this._initialized) return
    this._throwIfFailed()
    if (this._closed) throw new Error('Background task execution is closed')
    try {
      for (const restoredTask of restoredTasks) {
        const task = snapshot(restoredTask)
        validateStoredInProcessTask(task)
        this._tasks.set(task.taskId, task)
      }
      this._initialized = true
      for (const task of [...this._tasks.values()]) {
        if (isInProcessTaskTerminalStatus(task.status)) continue
        this._updateTask(task.taskId, (record) => {
          delete record.state
          record.status = 'failed'
          record.failure = {
            type: 'recoveryError',
            message: 'Background task execution was interrupted while restoring persisted state',
          }
          return true
        })
      }
    } catch (error) {
      this._initialized = false
      this._tasks.clear()
      throw error
    }
  }

  submit(admission: {
    readonly descriptor: Descriptor
    readonly idempotencyKey?: string
  }): StoredInProcessTask<Descriptor, Result, State> {
    this._assertInitialized()
    this._throwIfFailed()
    if (this._closed) throw new Error('Background task admission is closed')
    if (admission.idempotencyKey !== undefined) {
      const existing = [...this._tasks.values()].find((task) => task.idempotencyKey === admission.idempotencyKey)
      if (existing) return snapshot(existing)
    }
    const now = new Date().toISOString()
    const stored: StoredInProcessTask<Descriptor, Result, State> = {
      taskId: globalThis.crypto.randomUUID(),
      ...(admission.idempotencyKey !== undefined && { idempotencyKey: admission.idempotencyKey }),
      descriptor: globalThis.structuredClone(admission.descriptor),
      status: 'queued',
      createdAt: now,
      updatedAt: now,
    }
    this._persistTask(stored)
    this._tasks.set(stored.taskId, stored)
    this._enqueue(stored.taskId)
    return snapshot(stored)
  }

  get(taskId: string): StoredInProcessTask<Descriptor, Result, State> | undefined {
    this._assertInitialized()
    const task = this._tasks.get(taskId)
    return task ? snapshot(task) : undefined
  }

  list(): readonly StoredInProcessTask<Descriptor, Result, State>[] {
    this._assertInitialized()
    return [...this._tasks.values()].map(snapshot)
  }

  remove(taskId: string): void {
    this._assertInitialized()
    const task = this._requireTask(taskId)
    if (!isInProcessTaskTerminalStatus(task.status)) {
      throw new Error(`Background task '${taskId}' cannot be removed before reaching a terminal status`)
    }
    this._tasks.delete(taskId)
  }

  cancel(taskId: string, options: { readonly reason: string }): StoredInProcessTask<Descriptor, Result, State> {
    this._assertInitialized()
    this._throwIfFailed()
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
      delete activeExecution.timeout
    }
    activeExecution?.controller.abort(options.reason)
    this._wakeIdleWaiters()
    return task
  }

  async wait(
    taskId: string,
    options?: { readonly cancelSignal?: AbortSignal }
  ): Promise<StoredInProcessTask<Descriptor, Result, State>> {
    this._assertInitialized()
    const current = this._requireTask(taskId)
    if (isWaitComplete(current)) return current
    this._throwIfFailed()
    const signal = options?.cancelSignal
    if (signal?.aborted) throw getAbortReason(signal)

    return new Promise((resolve, reject) => {
      const waiters = this._taskWaiters.get(taskId) ?? new Set()
      const onAbort = (): void => {
        waiters.delete(waiter)
        if (waiters.size === 0) this._taskWaiters.delete(taskId)
        waiter.reject(getAbortReason(signal!))
      }
      const waiter: TaskWaiter<Descriptor, Result, State> = {
        resolve: (task) => {
          signal?.removeEventListener('abort', onAbort)
          resolve(task)
        },
        reject: (error) => {
          signal?.removeEventListener('abort', onAbort)
          reject(error)
        },
      }
      waiters.add(waiter)
      this._taskWaiters.set(taskId, waiters)
      signal?.addEventListener('abort', onAbort, { once: true })
    })
  }

  async waitForIdle(options?: { readonly cancelSignal?: AbortSignal }): Promise<void> {
    this._assertInitialized()
    await this._waitForIdle(options?.cancelSignal)
  }

  resume(
    taskId: string,
    update: (state: Exclude<State, undefined>) => { readonly state: Exclude<State, undefined>; readonly ready: boolean }
  ): StoredInProcessTask<Descriptor, Result, State> {
    this._assertInitialized()
    const task = this._updateTask(taskId, (record) => {
      if (this._closed) {
        throw new Error('Background task execution is closed')
      }
      if (record.status !== 'paused') {
        throw new Error(`Background task '${taskId}' cannot transition: status is '${record.status}', not 'paused'`)
      }
      if (record.state === undefined) {
        throw new Error(`Background task '${taskId}' cannot transition: paused state is missing`)
      }
      const resumed = update(record.state)
      record.state = resumed.state
      if (resumed.ready) record.status = 'queued'
      return true
    })!
    if (task.status === 'queued') this._enqueue(taskId)
    return task
  }

  async shutdown(options: { readonly timeout: number }): Promise<void> {
    assertTimerDelay('shutdown timeout', options.timeout)
    if (this._shutdown) return this._shutdown
    this._shutdown = this._shutdownEngine(options).catch((error: unknown) => {
      this._shutdown = undefined
      throw error
    })
    return this._shutdown
  }

  private _enqueue(taskId: string): void {
    if (this._closed || this._activeExecutions.has(taskId)) return
    this._queue.add(taskId)
    this._startQueuedTasks()
  }

  private _startQueuedTasks(): void {
    if (this._closed) return
    while (this._activeExecutions.size < this._config.maxConcurrency && this._queue.size > 0) {
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
          if (!this._closed && this.get(taskId)?.status === 'queued') {
            this._queue.add(taskId)
          }
          this._wakeIdleWaiters()
          this._startQueuedTasks()
        })
        .catch((error: unknown) => this._rejectWaiters(taskId, error))
    }
  }

  private async _execute(taskId: string, activeExecution: ActiveExecution): Promise<void> {
    const working = this._updateTask(taskId, (task) => {
      task.status = 'working'
      return true
    })!

    if (Number.isFinite(this._config.timeout)) {
      activeExecution.timeout = setTimeout(() => {
        delete activeExecution.timeout
        try {
          this._timeoutTask(taskId, activeExecution)
        } catch (error) {
          this._rejectWaiters(taskId, error)
        }
      }, this._config.timeout)
    }

    let outcome: TaskExecutionOutcome<Result, State>
    try {
      outcome = await this._config.execute({
        taskId,
        descriptor: working.descriptor,
        ...(working.state !== undefined && { state: working.state }),
        cancelSignal: activeExecution.controller.signal,
      })
    } catch (error) {
      outcome = {
        status: 'failed',
        failure: {
          type: DEFAULT_EXECUTION_FAILURE_TYPE,
          message: getExecutionFailureMessage(error),
        },
      }
    }
    try {
      this._finishOutcome(taskId, outcome)
    } catch (error) {
      if (this._failure) throw error
      this._finishOutcome(taskId, {
        status: 'failed',
        failure: {
          type: DEFAULT_EXECUTION_FAILURE_TYPE,
          message: getExecutionFailureMessage(error),
        },
      })
    }
  }

  private _finishOutcome(taskId: string, outcome: TaskExecutionOutcome<Result, State>): void {
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
    const reason = `Timed out after ${this._config.timeout}ms`
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

  private async _shutdownEngine(options: { readonly timeout: number }): Promise<void> {
    this._closed = true
    if (!this._initialized) return

    const failed = this._failure !== undefined
    const idleController = new AbortController()
    const timer = setTimeout(
      () => idleController.abort(new Error(`In-process task engine shutdown timed out after ${options.timeout}ms`)),
      options.timeout
    )
    try {
      if (!failed) {
        this.list()
          .filter((task) => !isInProcessTaskTerminalStatus(task.status))
          .forEach((task) => this.cancel(task.taskId, { reason: 'In-process task engine shutdown' }))
      }
      await this._waitForIdle(idleController.signal, failed)
    } finally {
      clearTimeout(timer)
    }
  }

  private _updateTask(
    taskId: string,
    update: (task: StoredInProcessTask<Descriptor, Result, State>) => boolean
  ): StoredInProcessTask<Descriptor, Result, State> | undefined {
    this._throwIfFailed()
    const current = this._tasks.get(taskId)
    if (!current) throw new BackgroundTaskNotFoundError(taskId)

    const next = snapshot(current)
    if (!update(next)) return undefined

    next.updatedAt = new Date().toISOString()
    const stored = snapshot(next)
    this._persistTask(stored)
    this._tasks.set(taskId, stored)
    this._notify(stored)

    return snapshot(stored)
  }

  private _requireTask(taskId: string): StoredInProcessTask<Descriptor, Result, State> {
    const task = this._tasks.get(taskId)
    if (!task) throw new BackgroundTaskNotFoundError(taskId)
    return snapshot(task)
  }

  private _notify(task: StoredInProcessTask<Descriptor, Result, State>): void {
    if (!isWaitComplete(task)) return
    const waiters = this._taskWaiters.get(task.taskId)
    if (!waiters) return
    this._taskWaiters.delete(task.taskId)
    for (const waiter of waiters) waiter.resolve(snapshot(task))
  }

  private _rejectWaiters(taskId: string, error: unknown): void {
    const waiters = this._taskWaiters.get(taskId)
    if (!waiters) return
    this._taskWaiters.delete(taskId)
    for (const waiter of waiters) waiter.reject(error)
  }

  private _persistTask(task: StoredInProcessTask<Descriptor, Result, State>): void {
    validateStoredInProcessTask(task)
    try {
      this._config.onTaskUpdated(snapshot(task))
    } catch (error) {
      this._failEngine(error)
      throw error
    }
  }

  private _failEngine(error: unknown): void {
    if (this._failure) return
    this._failure = { error }
    this._closed = true
    this._queue.clear()
    for (const activeExecution of this._activeExecutions.values()) {
      if (activeExecution.timeout) {
        clearTimeout(activeExecution.timeout)
        delete activeExecution.timeout
      }
      activeExecution.controller.abort(error)
    }
    for (const taskId of [...this._taskWaiters.keys()]) this._rejectWaiters(taskId, error)
    this._wakeIdleWaiters()
  }

  private _wakeIdleWaiters(): void {
    for (const resolve of this._idleWaiters) resolve()
    this._idleWaiters.clear()
  }

  private async _waitForIdle(cancelSignal?: AbortSignal, ignoreFailure = false): Promise<void> {
    while (this._queue.size > 0 || this._activeExecutions.size > 0) {
      if (!ignoreFailure) this._throwIfFailed()
      if (cancelSignal?.aborted) throw getAbortReason(cancelSignal)
      await new Promise<void>((resolve, reject) => {
        const onAbort = (): void => {
          this._idleWaiters.delete(onIdle)
          reject(getAbortReason(cancelSignal!))
        }
        const onIdle = (): void => {
          cancelSignal?.removeEventListener('abort', onAbort)
          resolve()
        }
        this._idleWaiters.add(onIdle)
        cancelSignal?.addEventListener('abort', onAbort, { once: true })
      })
    }
    if (!ignoreFailure) this._throwIfFailed()
  }

  private _assertInitialized(): void {
    if (!this._initialized) throw new Error('In-process task engine is not initialized')
  }

  private _throwIfFailed(): void {
    if (this._failure) throw this._failure.error
  }
}

function snapshot<Descriptor, Result, State>(
  task: StoredInProcessTask<Descriptor, Result, State>
): StoredInProcessTask<Descriptor, Result, State> {
  return globalThis.structuredClone(task)
}

function isWaitComplete(task: Pick<StoredInProcessTask<unknown, unknown, unknown>, 'status'>): boolean {
  return task.status === 'paused' || isInProcessTaskTerminalStatus(task.status)
}

function getAbortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException('Observation aborted', 'AbortError')
}

function getExecutionFailureMessage(error: unknown): string {
  try {
    return normalizeError(error).message || DEFAULT_EXECUTION_FAILURE_MESSAGE
  } catch {
    return DEFAULT_EXECUTION_FAILURE_MESSAGE
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
