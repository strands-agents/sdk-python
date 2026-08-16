import { normalizeError } from '../../errors.js'
import { BackgroundTaskNotFoundError } from '../errors.js'
import { isEngineTerminalStatus, validateStoredEngineTask } from './record.js'
import type {
  BackgroundTaskEngineConfig,
  BackgroundTaskExecutionOutcome,
  DefinedValue,
  StoredEngineTask,
} from './types.js'

// Node.js treats delays above the signed 32-bit limit as 1ms.
const MAX_TIMER_DELAY_MS = 2 ** 31 - 1
const DEFAULT_EXECUTION_FAILURE_TYPE = 'executionError'
const DEFAULT_EXECUTION_FAILURE_MESSAGE = 'Background task execution failed'

interface ActiveExecution {
  readonly controller: AbortController
  timeout?: ReturnType<typeof setTimeout>
}

interface TaskWaiter<Descriptor, Result extends DefinedValue, State extends DefinedValue> {
  resolve(task: StoredEngineTask<Descriptor, Result, State>): void
  reject(error: unknown): void
}

/** Bounded background task execution. @internal */
export class BackgroundTaskEngine<Descriptor, Result extends DefinedValue, State extends DefinedValue> {
  private readonly _config: BackgroundTaskEngineConfig<Descriptor, Result, State>
  private readonly _tasks = new Map<string, StoredEngineTask<Descriptor, Result, State>>()
  private readonly _queue = new Set<string>()
  private readonly _activeExecutions = new Map<string, ActiveExecution>()
  private readonly _pendingRemoval = new Set<string>()
  private readonly _waiters = new Map<string, Set<TaskWaiter<Descriptor, Result, State>>>()
  private readonly _idleWaiters = new Set<() => void>()
  private _initialized = false
  private _accepting = true
  private _stopping = false
  private _shutdown: Promise<void> | undefined
  private _failure: { readonly error: unknown } | undefined

  constructor(config: BackgroundTaskEngineConfig<Descriptor, Result, State>) {
    if (!Number.isSafeInteger(config.maxConcurrency) || config.maxConcurrency <= 0) {
      throw new TypeError(`maxConcurrency must be a positive finite integer, got ${config.maxConcurrency}`)
    }
    if (config.timeout !== Infinity) assertTimerDelay('timeout', config.timeout)
    this._config = config
  }

  initialize(restoredTasks: readonly StoredEngineTask<Descriptor, Result, State>[] = []): void {
    if (this._initialized) return
    this._throwIfFailed()
    if (!this._accepting) throw new Error('Background task execution is closed')
    try {
      const idempotencyKeys = new Set<string>()
      for (const restoredTask of restoredTasks) {
        const task = snapshot(restoredTask)
        validateStoredEngineTask(task)
        if (this._tasks.has(task.taskId)) throw new Error(`Duplicate background task ID '${task.taskId}'`)
        if (task.idempotencyKey !== undefined) {
          if (idempotencyKeys.has(task.idempotencyKey)) {
            throw new Error(`Duplicate background task idempotency key '${task.idempotencyKey}'`)
          }
          idempotencyKeys.add(task.idempotencyKey)
        }
        this._tasks.set(task.taskId, task)
      }
      this._initialized = true
      for (const task of [...this._tasks.values()]) {
        if (task.status !== 'working') continue
        this._updateTask(task.taskId, (record) => {
          delete record.attemptId
          record.status = 'failed'
          record.failure = {
            type: 'recoveryError',
            message: 'Background task execution was interrupted while restoring persisted state',
          }
          return true
        })
      }
      for (const task of this._tasks.values()) {
        if (task.status === 'queued') this._enqueue(task.taskId)
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
  }): StoredEngineTask<Descriptor, Result, State> {
    this._assertInitialized()
    this._throwIfFailed()
    if (!this._accepting) throw new Error('Background task admission is closed')
    if (admission.idempotencyKey !== undefined) {
      const existing = [...this._tasks.values()].find(
        (task) => !this._pendingRemoval.has(task.taskId) && task.idempotencyKey === admission.idempotencyKey
      )
      if (existing) return snapshot(existing)
    }
    const now = new Date().toISOString()
    const stored: StoredEngineTask<Descriptor, Result, State> = {
      taskId: globalThis.crypto.randomUUID(),
      ...(admission.idempotencyKey !== undefined && { idempotencyKey: admission.idempotencyKey }),
      descriptor: globalThis.structuredClone(admission.descriptor),
      status: 'queued',
      attemptCount: 0,
      createdAt: now,
      updatedAt: now,
    }
    this._persistTask(stored)
    this._tasks.set(stored.taskId, stored)
    this._emit({ type: 'admitted', task: stored })
    this._enqueue(stored.taskId)
    return snapshot(stored)
  }

  get(taskId: string): StoredEngineTask<Descriptor, Result, State> | undefined {
    this._assertInitialized()
    if (this._pendingRemoval.has(taskId)) return undefined
    const task = this._tasks.get(taskId)
    return task ? snapshot(task) : undefined
  }

  list(): readonly StoredEngineTask<Descriptor, Result, State>[] {
    this._assertInitialized()
    return [...this._tasks.values()].filter((task) => !this._pendingRemoval.has(task.taskId)).map(snapshot)
  }

  remove(taskId: string): void {
    this._assertInitialized()
    const task = this._requireVisibleTask(taskId)
    if (!isEngineTerminalStatus(task.status)) {
      throw new Error(`Background task '${taskId}' cannot be removed before reaching a terminal status`)
    }
    if (this._activeExecutions.has(taskId)) {
      // Cancellation and timeout can become terminal before a non-cooperative callback returns.
      this._pendingRemoval.add(taskId)
      return
    }
    this._tasks.delete(taskId)
  }

  cancel(taskId: string, options: { readonly reason: string }): StoredEngineTask<Descriptor, Result, State> {
    this._assertInitialized()
    this._throwIfFailed()
    this._requireVisibleTask(taskId)
    const updated = this._updateTask(taskId, (task) => {
      if (isEngineTerminalStatus(task.status)) return false
      task.cancellationReason = options.reason
      task.status = 'cancelled'
      delete task.attemptId
      delete task.failure
      delete task.result
      return true
    })
    const task = updated ?? this._requireTask(taskId)
    if (updated) {
      this._queue.delete(taskId)
      const activeExecution = this._activeExecutions.get(taskId)
      if (activeExecution?.timeout) {
        clearTimeout(activeExecution.timeout)
        delete activeExecution.timeout
      }
      activeExecution?.controller.abort(options.reason)
      this._emit({ type: 'cancelled', task })
    }
    this._signalIdle()
    this._pump()
    return task
  }

  async wait(
    taskId: string,
    options?: { readonly cancelSignal?: AbortSignal }
  ): Promise<StoredEngineTask<Descriptor, Result, State>> {
    this._assertInitialized()
    const current = this._requireVisibleTask(taskId)
    if (isWaitComplete(current)) return current
    this._throwIfFailed()
    const signal = options?.cancelSignal
    if (signal?.aborted) throw getAbortReason(signal)

    return new Promise((resolve, reject) => {
      const waiters = this._waiters.get(taskId) ?? new Set()
      const waiter: TaskWaiter<Descriptor, Result, State> = { resolve, reject }
      const onAbort = (): void => {
        waiters.delete(waiter)
        if (waiters.size === 0) this._waiters.delete(taskId)
        waiter.reject(getAbortReason(signal!))
      }
      const finish = (task: StoredEngineTask<Descriptor, Result, State>): void => {
        signal?.removeEventListener('abort', onAbort)
        resolve(task)
      }
      const fail = (error: unknown): void => {
        signal?.removeEventListener('abort', onAbort)
        reject(error)
      }
      waiter.resolve = finish
      waiter.reject = fail
      waiters.add(waiter)
      this._waiters.set(taskId, waiters)
      signal?.addEventListener('abort', onAbort, { once: true })
    })
  }

  async waitForIdle(options?: { readonly cancelSignal?: AbortSignal }): Promise<void> {
    this._assertInitialized()
    await this._waitForIdle(options?.cancelSignal)
  }

  resume(
    taskId: string,
    update: (state: State) => { readonly state: State; readonly ready: boolean }
  ): StoredEngineTask<Descriptor, Result, State> {
    this._assertInitialized()
    this._throwIfFailed()
    this._requireVisibleTask(taskId)
    const task = this._updateTask(taskId, (record) => {
      if (!this._accepting) {
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

  async shutdown(options: { readonly mode: 'drain' | 'cancel'; readonly timeout: number }): Promise<void> {
    if (this._shutdown) return this._shutdown
    assertTimerDelay('shutdown timeout', options.timeout)
    this._shutdown = this._shutdownEngine(options).catch((error: unknown) => {
      this._shutdown = undefined
      throw error
    })
    return this._shutdown
  }

  private _enqueue(taskId: string): void {
    if (this._stopping || this._queue.has(taskId) || this._activeExecutions.has(taskId)) return
    this._queue.add(taskId)
    this._pump()
  }

  private _pump(): void {
    if (this._stopping || !this._initialized) return
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
          if (this._pendingRemoval.delete(taskId)) {
            this._tasks.delete(taskId)
          } else {
            const current = this.get(taskId)
            if (current?.status === 'queued') this._enqueue(taskId)
          }
          this._signalIdle()
          this._pump()
        })
        .catch((error: unknown) => this._rejectWaiters(taskId, error))
    }
  }

  private async _execute(taskId: string, activeExecution: ActiveExecution): Promise<void> {
    const before = this._requireTask(taskId)
    if (before.status !== 'queued') return
    const resumed = before.attemptId !== undefined
    const attempt = before.attemptCount + (resumed ? 0 : 1)
    const attemptId = before.attemptId ?? globalThis.crypto.randomUUID()
    const executionId = globalThis.crypto.randomUUID()
    const startedAt = Date.now()
    const working = this._updateTask(taskId, (task) => {
      if (task.status !== 'queued') return false
      task.status = 'working'
      if (!resumed) task.attemptCount = attempt
      task.attemptId = attemptId
      delete task.failure
      return true
    })
    if (!working) return

    this._emit({
      type: 'executionStarted',
      task: working,
      resumed,
      queueDuration: startedAt - Date.parse(before.updatedAt),
    })
    if (Number.isFinite(this._config.timeout)) {
      activeExecution.timeout = setTimeout(() => {
        delete activeExecution.timeout
        try {
          this._timeoutTask(taskId, attemptId, activeExecution)
        } catch (error) {
          this._rejectWaiters(taskId, error)
        }
      }, this._config.timeout)
    }

    try {
      let outcome: BackgroundTaskExecutionOutcome<Result, State>
      try {
        outcome = await this._config.execute({
          taskId,
          descriptor: working.descriptor,
          ...(working.state !== undefined && { state: working.state }),
          attempt,
          attemptId,
          executionId,
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
        this._finishOutcome(taskId, attemptId, outcome)
      } catch (error) {
        if (this._failure) throw error
        this._finishOutcome(taskId, attemptId, {
          status: 'failed',
          failure: {
            type: DEFAULT_EXECUTION_FAILURE_TYPE,
            message: getExecutionFailureMessage(error),
          },
        })
      }
    } finally {
      const latest = this._requireTask(taskId)
      this._emit({
        type: 'executionFinished',
        task: latest,
        duration: Date.now() - startedAt,
      })
    }
  }

  private _finishOutcome(
    taskId: string,
    attemptId: string,
    outcome: BackgroundTaskExecutionOutcome<Result, State>
  ): void {
    if (outcome.status === 'paused') {
      this._updateTask(taskId, (record) => {
        if (record.status !== 'working' || record.attemptId !== attemptId) return false
        record.status = 'paused'
        record.state = outcome.state
        return true
      })
      return
    }

    this._updateTask(taskId, (record) => {
      if (record.status !== 'working' || record.attemptId !== attemptId) return false
      if (outcome.status === 'failed') {
        if (outcome.state !== undefined) {
          record.state = outcome.state
        }
        record.status = 'failed'
        delete record.attemptId
        record.failure = {
          type: outcome.failure.type || DEFAULT_EXECUTION_FAILURE_TYPE,
          message: outcome.failure.message || DEFAULT_EXECUTION_FAILURE_MESSAGE,
        }
        if (outcome.result === undefined) {
          delete record.result
        } else {
          record.result = outcome.result
        }
        return true
      }
      record.status = 'completed'
      delete record.attemptId
      record.result = outcome.result
      if (outcome.state === undefined) {
        delete record.state
      } else {
        record.state = outcome.state
      }
      delete record.failure
      return true
    })
  }

  private _timeoutTask(taskId: string, attemptId: string, activeExecution: ActiveExecution): void {
    const reason = `Timed out after ${this._config.timeout}ms`
    const task = this._updateTask(taskId, (record) => {
      if (record.status !== 'working' || record.attemptId !== attemptId) return false
      record.status = 'failed'
      delete record.attemptId
      record.failure = {
        type: 'timeout',
        message: reason,
      }
      return true
    })
    if (task) activeExecution.controller.abort(reason)
  }

  private async _shutdownEngine(options: {
    readonly mode: 'drain' | 'cancel'
    readonly timeout: number
  }): Promise<void> {
    this._accepting = false
    if (!this._initialized) return

    const failed = this._failure !== undefined
    const idleController = new AbortController()
    let timer: ReturnType<typeof setTimeout> | undefined
    const completed = await Promise.race([
      (async (): Promise<true> => {
        if (options.mode === 'cancel' && !failed) {
          this._stopping = true
          this.list()
            .filter((task) => !isEngineTerminalStatus(task.status))
            .forEach((task) => this.cancel(task.taskId, { reason: 'Coordinator shutdown' }))
        }
        await this._waitForIdle(idleController.signal, failed)
        return true
      })(),
      new Promise<false>((resolve) => {
        timer = setTimeout(() => resolve(false), options.timeout)
      }),
    ]).finally(() => {
      if (timer) clearTimeout(timer)
      idleController.abort()
    })
    if (!completed) throw new Error(`Background Task Engine shutdown timed out after ${options.timeout}ms`)
  }

  private _updateTask(
    taskId: string,
    update: (task: StoredEngineTask<Descriptor, Result, State>) => boolean
  ): StoredEngineTask<Descriptor, Result, State> | undefined {
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

  private _requireTask(taskId: string): StoredEngineTask<Descriptor, Result, State> {
    const task = this._tasks.get(taskId)
    if (!task) throw new BackgroundTaskNotFoundError(taskId)
    return snapshot(task)
  }

  private _requireVisibleTask(taskId: string): StoredEngineTask<Descriptor, Result, State> {
    if (this._pendingRemoval.has(taskId)) throw new BackgroundTaskNotFoundError(taskId)
    return this._requireTask(taskId)
  }

  private _notify(task: StoredEngineTask<Descriptor, Result, State>): void {
    if (!isWaitComplete(task)) return
    const waiters = this._waiters.get(task.taskId)
    if (!waiters) return
    this._waiters.delete(task.taskId)
    for (const waiter of waiters) waiter.resolve(snapshot(task))
  }

  private _rejectWaiters(taskId: string, error: unknown): void {
    const waiters = this._waiters.get(taskId)
    if (!waiters) return
    this._waiters.delete(taskId)
    for (const waiter of waiters) waiter.reject(error)
  }

  private _persistTask(task: StoredEngineTask<Descriptor, Result, State>): void {
    validateStoredEngineTask(task)
    try {
      this._config.onTaskUpdated?.(snapshot(task))
    } catch (error) {
      this._failEngine(error)
      throw error
    }
  }

  private _failEngine(error: unknown): void {
    if (this._failure) return
    this._failure = { error }
    this._accepting = false
    this._stopping = true
    this._queue.clear()
    for (const activeExecution of this._activeExecutions.values()) {
      if (activeExecution.timeout) {
        clearTimeout(activeExecution.timeout)
        delete activeExecution.timeout
      }
      activeExecution.controller.abort(error)
    }
    for (const taskId of [...this._waiters.keys()]) this._rejectWaiters(taskId, error)
    this._signalIdle()
  }

  private _emit(event: Parameters<NonNullable<typeof this._config.onEvent>>[0]): void {
    try {
      this._config.onEvent?.(globalThis.structuredClone(event))
    } catch {
      // Observers cannot affect task execution.
    }
  }

  private _signalIdle(): void {
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
    if (!this._initialized) throw new Error('Background Task Engine is not initialized')
  }

  private _throwIfFailed(): void {
    if (this._failure) throw this._failure.error
  }
}

function snapshot<Descriptor, Result extends DefinedValue, State extends DefinedValue>(
  task: StoredEngineTask<Descriptor, Result, State>
): StoredEngineTask<Descriptor, Result, State> {
  return globalThis.structuredClone(task)
}

function isWaitComplete(task: Pick<StoredEngineTask, 'status'>): boolean {
  return task.status === 'paused' || isEngineTerminalStatus(task.status)
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
