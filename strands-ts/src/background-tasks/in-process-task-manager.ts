import { isSpanContextValid } from '@opentelemetry/api'

import { normalizeError } from '../errors.js'
import { InterruptError, InterruptState, type Interrupt, type InterruptStateData } from '../interrupt.js'
import { InterruptResponseContent, type InterruptResponse } from '../types/interrupt.js'
import { deepCopyWithValidation } from '../types/json.js'
import { TextBlock, ToolUseBlock, type ToolResultBlock, type ToolResultBlockData } from '../types/messages.js'
import type { Agent } from '../agent/agent.js'
import { BackgroundTaskEngine } from './engine/engine.js'
import { isEngineTerminalStatus } from './engine/record.js'
import type {
  BackgroundTaskEngineEvent,
  BackgroundTaskExecutionContext,
  BackgroundTaskExecutionOutcome,
} from './engine/types.js'
import { BackgroundTaskNotFoundError, BackgroundTasksTimeoutError } from './errors.js'
import { AfterInvocationEvent, AfterModelCallEvent, BeforeModelCallEvent } from '../hooks/events.js'
import { HookOrder } from '../hooks/types.js'
import { AgentResult, type InvocationState } from '../types/agent.js'
import type { JSONValue } from '../types/json.js'
import {
  assertDeliveryConsumed,
  historyContainsBackgroundDelivery,
  renderBackgroundDelivery,
  stableStringify,
  unpinBackgroundDeliveries,
} from './delivery.js'
import { toBackgroundTask, validateStoredTask, type StoredBackgroundTask, type ToolTaskDescriptor } from './record.js'
import { BackgroundTaskTelemetry } from './telemetry.js'
import type { BackgroundTask, BackgroundTasksConfig } from './types.js'
import type { TaskManager, ToolCallSubmission } from './task-manager.js'

const DEFAULT_MAX_CONCURRENCY = 4
const STATE_RELOAD_TIMEOUT = 30_000

export const BACKGROUND_TASKS_STATE_KEY = 'strands.backgroundTasks'

export class InProcessTaskManager implements TaskManager<ToolCallSubmission> {
  private readonly _agent: Agent
  private readonly _config: {
    readonly maxConcurrency: number
    readonly timeout: number
    readonly waitForCompletion: boolean
  }
  private _engine: BackgroundTaskEngine<ToolTaskDescriptor, ToolResultBlockData, InterruptStateData>
  private readonly _records = new Map<string, StoredBackgroundTask>()
  private readonly _deliveryStates = new Map<string, 'pending' | 'ready' | 'delivered'>()
  private readonly _telemetry = new BackgroundTaskTelemetry()
  private readonly _delivering = new Set<string>()
  private readonly _deliveryWaiters = new Set<() => void>()
  private _reload: Promise<void> | undefined

  constructor(agent: Agent, config: BackgroundTasksConfig) {
    this._agent = agent
    this._config = {
      maxConcurrency: config.maxConcurrency ?? DEFAULT_MAX_CONCURRENCY,
      timeout: config.timeout ?? Infinity,
      waitForCompletion: config.waitForCompletion !== false,
    }
    this._engine = this._createEngine()
  }

  private _createEngine(): BackgroundTaskEngine<ToolTaskDescriptor, ToolResultBlockData, InterruptStateData> {
    return new BackgroundTaskEngine({
      maxConcurrency: this._config.maxConcurrency,
      timeout: this._config.timeout,
      execute: (context) => this._executeToolTask(context),
      onTaskUpdated: (record): void => {
        this._records.set(record.taskId, record)
        this._deliveryStates.set(record.taskId, deliveryStateFor(record, this._deliveryStates.get(record.taskId)))
        this._persistAppState()
      },
      onEvent: (event): void => this._recordEngineEvent(event),
    })
  }

  async initialize(): Promise<void> {
    this._initializeEngine(this._engine)
  }

  registerHooks(): void {
    this._agent.addHook(BeforeModelCallEvent, (event) => this._onBeforeModelCall(event))
    this._agent.addHook(AfterModelCallEvent, (event) => this._onAfterModelCall(event))
    this._agent.addHook(AfterInvocationEvent, (event) => this._onAfterInvocation(event), {
      order: HookOrder.SDK_FIRST,
    })
  }

  private _initializeEngine(
    engine: BackgroundTaskEngine<ToolTaskDescriptor, ToolResultBlockData, InterruptStateData>
  ): void {
    this._loadAppState()
    engine.initialize([...this._records.values()])
    const records = engine.list()
    const taskIdsToUnpin = new Set(
      records.filter((record) => this._deliveryStates.get(record.taskId) === 'delivered').map((record) => record.taskId)
    )
    this._pruneDelivered(engine, [...taskIdsToUnpin])
    for (const taskId of this._reconcileReadyDeliveries(engine)) taskIdsToUnpin.add(taskId)
    unpinBackgroundDeliveries(this._agent.messages, taskIdsToUnpin)
  }

  private _loadAppState(): void {
    this._records.clear()
    this._deliveryStates.clear()
    const value = this._agent.appState.get(BACKGROUND_TASKS_STATE_KEY)
    if (value === undefined) return
    if (!isObject(value)) throw new Error(`${BACKGROUND_TASKS_STATE_KEY} must be an object`)

    for (const [taskId, storedValue] of Object.entries(value)) {
      if (!isObject(storedValue) || !('record' in storedValue) || !('deliveryState' in storedValue)) {
        throw new Error(`${BACKGROUND_TASKS_STATE_KEY}.${taskId} is invalid`)
      }
      validateStoredTask(storedValue.record)
      if (storedValue.record.taskId !== taskId) {
        throw new Error(`${BACKGROUND_TASKS_STATE_KEY}.${taskId}.record.taskId must match its map key`)
      }
      if (
        storedValue.deliveryState !== 'pending' &&
        storedValue.deliveryState !== 'ready' &&
        storedValue.deliveryState !== 'delivered'
      ) {
        throw new Error(`${BACKGROUND_TASKS_STATE_KEY}.${taskId}.deliveryState is invalid`)
      }
      this._records.set(taskId, storedValue.record)
      this._deliveryStates.set(taskId, storedValue.deliveryState)
    }
  }

  private _persistAppState(): void {
    if (this._records.size === 0) {
      this._agent.appState.delete(BACKGROUND_TASKS_STATE_KEY)
      return
    }
    const tasks: Record<
      string,
      {
        record: StoredBackgroundTask
        deliveryState: 'pending' | 'ready' | 'delivered'
      }
    > = {}
    for (const record of this._records.values()) {
      tasks[record.taskId] = {
        record,
        deliveryState: this._deliveryStates.get(record.taskId) ?? deliveryStateFor(record),
      }
    }
    this._agent.appState.set(BACKGROUND_TASKS_STATE_KEY, tasks)
  }

  private _pruneDelivered(
    engine: BackgroundTaskEngine<ToolTaskDescriptor, ToolResultBlockData, InterruptStateData>,
    taskIds: readonly string[]
  ): void {
    if (taskIds.length === 0) return
    for (const taskId of taskIds) {
      const record = this._records.get(taskId)
      if (!record) throw new Error(`Background task '${taskId}' was not found`)
      const state = this._deliveryStates.get(taskId)
      if ((state !== 'ready' && state !== 'delivered') || !isEngineTerminalStatus(record.status)) {
        throw new Error(`Background task '${taskId}' does not have a ready result`)
      }
      engine.remove(taskId)
      this._records.delete(taskId)
      this._deliveryStates.delete(taskId)
    }
    this._persistAppState()
  }

  private _reconcileReadyDeliveries(
    engine: BackgroundTaskEngine<ToolTaskDescriptor, ToolResultBlockData, InterruptStateData>
  ): readonly string[] {
    const delivered = engine
      .list()
      .filter(
        (record) =>
          isEngineTerminalStatus(record.status) &&
          this._deliveryStates.get(record.taskId) === 'ready' &&
          !this._delivering.has(record.taskId) &&
          historyContainsBackgroundDelivery(this._agent.messages, record)
      )
      .map((record) => record.taskId)
    this._pruneDelivered(engine, delivered)
    return delivered
  }

  appStateLoaded(): void {
    const restoredTasks = this._agent.appState.get(BACKGROUND_TASKS_STATE_KEY)
    const previousReload = this._reload
    const reload = previousReload
      ? previousReload.then(
          () => this._reloadFromAppState(restoredTasks),
          () => this._reloadFromAppState(restoredTasks)
        )
      : this._reloadFromAppState(restoredTasks)
    this._reload = reload
    void reload.then(
      () => {
        if (this._reload === reload) this._reload = undefined
      },
      () => undefined
    )
  }

  private async _reloadFromAppState(restoredTasks: JSONValue | undefined): Promise<void> {
    const deadline = Date.now() + STATE_RELOAD_TIMEOUT
    await this._waitForDeliveries(STATE_RELOAD_TIMEOUT)
    await this._engine.shutdown({
      mode: 'cancel',
      timeout: Math.max(1, deadline - Date.now()),
    })
    this._delivering.clear()

    if (restoredTasks === undefined) {
      this._agent.appState.delete(BACKGROUND_TASKS_STATE_KEY)
    } else {
      this._agent.appState.set(BACKGROUND_TASKS_STATE_KEY, restoredTasks)
    }

    const engine = this._createEngine()
    try {
      this._initializeEngine(engine)
      this._engine = engine
    } catch (error) {
      await engine.shutdown({ mode: 'cancel', timeout: 1_000 }).catch(() => undefined)
      throw error
    }
  }

  private async _waitForReload(cancelSignal?: AbortSignal): Promise<void> {
    let reload = this._reload
    while (reload) {
      try {
        await waitWithSignal(reload, cancelSignal)
      } catch (error) {
        if (cancelSignal?.aborted || reload === this._reload) throw error
      }
      if (reload === this._reload || this._reload === undefined) return
      reload = this._reload
    }
  }

  async submitTask(submission: ToolCallSubmission): Promise<BackgroundTask> {
    if (this._reload) await this._waitForReload()
    const originTraceContext =
      submission.originSpanContext && isSpanContextValid(submission.originSpanContext)
        ? {
            traceId: submission.originSpanContext.traceId,
            spanId: submission.originSpanContext.spanId,
            traceFlags: submission.originSpanContext.traceFlags,
            ...(submission.originSpanContext.isRemote !== undefined && {
              isRemote: submission.originSpanContext.isRemote,
            }),
          }
        : undefined
    const input = deepCopyWithValidation(submission.input, 'background task input')
    const invocationState = deepCopyWithValidation(
      submission.invocationState,
      'background task invocationState'
    ) as InvocationState
    const descriptor: ToolTaskDescriptor = {
      originalToolUseId: submission.originalToolUseId,
      toolName: submission.toolName,
      input,
      invocationState,
      ...(originTraceContext && { originTraceContext }),
    }
    const stored = this._engine.submit({
      descriptor,
      idempotencyKey: JSON.stringify([submission.passId, descriptor.originalToolUseId]),
    })
    return toBackgroundTask(stored)
  }

  async getTask(taskId: string): Promise<BackgroundTask | undefined> {
    if (this._reload) await this._waitForReload()
    const record = this._engine.get(taskId)
    return record ? toBackgroundTask(record) : undefined
  }

  async listTasks(): Promise<readonly BackgroundTask[]> {
    if (this._reload) await this._waitForReload()
    return this._engine.list().map(toBackgroundTask)
  }

  async cancelTask(taskId: string): Promise<BackgroundTask> {
    if (this._reload) await this._waitForReload()
    const previousStatus = this._engine.get(taskId)?.status
    const task = this._engine.cancel(taskId, { reason: 'Cancellation requested' })
    if (task.status === 'cancelled' && previousStatus !== 'cancelled') {
      this._telemetry.recordCancellation(task.descriptor.toolName)
    }
    return toBackgroundTask(task)
  }

  async waitForTasks(options?: { readonly timeout?: number }): Promise<void> {
    const timeout = options?.timeout
    if (timeout !== undefined && (!Number.isSafeInteger(timeout) || timeout <= 0)) {
      throw new TypeError(`wait timeout must be a positive finite integer, got ${timeout}`)
    }
    const timeoutController = timeout === undefined ? undefined : new AbortController()
    const timeoutTimer =
      timeoutController && timeout !== undefined
        ? setTimeout(() => timeoutController.abort(new BackgroundTasksTimeoutError(timeout)), timeout)
        : undefined
    try {
      if (this._reload) await this._waitForReload(timeoutController?.signal)
      await this._engine.waitForIdle(timeoutController ? { cancelSignal: timeoutController.signal } : undefined)
    } finally {
      if (timeoutTimer) clearTimeout(timeoutTimer)
    }
  }

  private _resumeTask(taskId: string, responses: readonly InterruptResponse[]): BackgroundTask {
    const current = this._engine.get(taskId)
    if (!current) throw new BackgroundTaskNotFoundError(taskId)
    if (current.status !== 'paused') {
      if (current.state && responsesAlreadyApplied(current.state, responses)) {
        return toBackgroundTask(current)
      }
      throw new Error(`Background task '${taskId}' cannot transition: status is '${current.status}', not 'paused'`)
    }
    return toBackgroundTask(
      this._engine.resume(taskId, (state) => {
        const interruptState = InterruptState.fromJSON(state)
        const knownIds = new Set(Object.keys(interruptState.interrupts))
        for (const response of responses) {
          if (!knownIds.has(response.interruptId)) {
            throw new Error(
              `Background task '${taskId}' cannot transition: unknown interrupt '${response.interruptId}'`
            )
          }
        }
        interruptState.resume(
          responses.map(
            (response) =>
              new InterruptResponseContent({
                interruptId: response.interruptId,
                response: response.response,
              })
          )
        )
        return {
          state: interruptState.toJSON(),
          ready: interruptState.getUnansweredInterrupts().length === 0,
        }
      })
    )
  }

  private async _onAfterModelCall(event: AfterModelCallEvent): Promise<void> {
    if (this._reload) await this._waitForReload()
    if (event.error || !event.stopData) return
    this._throwPausedInterrupts()
  }

  private async _onAfterInvocation(event: AfterInvocationEvent): Promise<void> {
    if (this._reload) await this._waitForReload()
    const stopReason = event._getResult()?.stopReason
    if (!this._config.waitForCompletion || !stopReason || stopReason === 'cancelled' || stopReason === 'interrupt') {
      return
    }

    const cannotContinue =
      stopReason === 'checkpoint' ||
      stopReason === 'limitTurns' ||
      stopReason === 'limitOutputTokens' ||
      stopReason === 'limitTotalTokens'
    await this._waitForTaskResult(cannotContinue)
    if (this._agent.cancelSignal.aborted) return
    if (this._surfacePausedInterrupt(event)) return
    if (cannotContinue) return
    this._deliverReady(event)
  }

  private _surfacePausedInterrupt(event: AfterInvocationEvent): boolean {
    const interrupts = this._pausedInterrupts()
    if (interrupts.length === 0) return false

    const interruptState = event._getInterruptState()
    const result = event._getResult()
    if (!interruptState || !result) {
      throw new Error('Background interrupt cannot be surfaced without an active invocation result')
    }
    for (const interrupt of interrupts) {
      interruptState.registerInterrupt(interrupt)
    }
    interruptState.activate()
    event._setResult(
      new AgentResult({
        stopReason: 'interrupt',
        lastMessage: result.lastMessage,
        invocationState: event.invocationState,
        ...(result.traces !== undefined && { traces: result.traces }),
        ...(result.metrics !== undefined && { metrics: result.metrics }),
        interrupts: interruptState.getUnansweredInterrupts(),
      })
    )
    return true
  }

  private async _onBeforeModelCall(event: BeforeModelCallEvent): Promise<void> {
    if (this._reload) await this._waitForReload()
    this._routeInterruptResponses(event)
    this._throwPausedInterrupts()
    this._deliverReady(event)
  }

  private _deliverReady(event: BeforeModelCallEvent | AfterInvocationEvent): void {
    let continuationRegistered = false
    let taskIds: string[] = []
    try {
      const engine = this._engine
      const alreadyDelivered = this._reconcileReadyDeliveries(engine)
      unpinBackgroundDeliveries(this._agent.messages, new Set(alreadyDelivered))

      const records = engine
        .list()
        .filter(
          (record) =>
            isEngineTerminalStatus(record.status) &&
            this._deliveryStates.get(record.taskId) === 'ready' &&
            !this._delivering.has(record.taskId)
        )
      if (records.length === 0) return
      taskIds = records.map((record) => record.taskId)
      for (const taskId of taskIds) this._delivering.add(taskId)

      const deliveries = records.map((record) => renderBackgroundDelivery(record))

      event._continueWith({
        phase: 'deferredResult',
        args: deliveries.flat(),
        onSelected: (modelRequestMessages) => {
          records.forEach((record, index) => {
            assertDeliveryConsumed(record.taskId, deliveries[index]!, modelRequestMessages)
          })
        },
        onCommitted: () => {
          try {
            if (this._engine === engine) {
              this._pruneDelivered(engine, taskIds)
              unpinBackgroundDeliveries(this._agent.messages, new Set(taskIds))
            }
          } finally {
            this._finishDelivery(taskIds)
          }
        },
        onRejected: () => {
          this._finishDelivery(taskIds)
        },
      })
      continuationRegistered = true
    } finally {
      if (!continuationRegistered) {
        this._finishDelivery(taskIds)
      }
    }
  }

  private _finishDelivery(taskIds: readonly string[]): void {
    for (const taskId of taskIds) this._delivering.delete(taskId)
    for (const resolve of this._deliveryWaiters) resolve()
    this._deliveryWaiters.clear()
  }

  private async _executeToolTask(
    context: BackgroundTaskExecutionContext<ToolTaskDescriptor, InterruptStateData>
  ): Promise<BackgroundTaskExecutionOutcome<ToolResultBlockData, InterruptStateData>> {
    const descriptor = context.descriptor
    if (!this._agent.toolRegistry.get(descriptor.toolName)) {
      return {
        status: 'failed',
        failure: {
          type: 'recoveryError',
          message: `Tool '${descriptor.toolName}' is not registered on Agent '${this._agent.id}'`,
        },
        ...(context.state !== undefined && { state: context.state }),
      }
    }
    const originSpanContext = descriptor.originTraceContext
    let outcome: ToolResultBlock | { readonly interruptState: InterruptStateData }
    try {
      outcome = await this._agent.executeDetachedTool({
        toolUseBlock: new ToolUseBlock({
          name: descriptor.toolName,
          toolUseId: descriptor.originalToolUseId,
          input: descriptor.input,
        }),
        invocationState: descriptor.invocationState,
        cancelSignal: context.cancelSignal,
        beforeToolCallCompleted: true,
        ...(context.state && { interruptState: context.state }),
        background: {
          taskId: context.taskId,
          attempt: context.attempt,
          attemptId: context.attemptId,
          executionId: context.executionId,
        },
        ...(originSpanContext && { originSpanContext }),
      })
    } catch (error) {
      return {
        status: 'failed',
        failure: {
          type: 'executionError',
          message: normalizeError(error).message,
        },
        ...(context.state !== undefined && { state: context.state }),
      }
    }
    if ('interruptState' in outcome) {
      return {
        status: 'paused',
        state: outcome.interruptState,
      }
    }
    const serialized = outcome.toJSON().toolResult
    if (outcome.status === 'error') {
      return {
        status: 'failed',
        failure: {
          type: 'toolError',
          message:
            outcome.error?.message ??
            outcome.content.find((content): content is TextBlock => content instanceof TextBlock)?.text ??
            'Tool returned an error without a message',
        },
        result: serialized,
        ...(context.state && { state: context.state }),
      }
    }
    return {
      status: 'completed',
      result: serialized,
      ...(context.state && { state: context.state }),
    }
  }

  private _recordEngineEvent(
    event: BackgroundTaskEngineEvent<ToolTaskDescriptor, ToolResultBlockData, InterruptStateData>
  ): void {
    const toolName = event.task.descriptor.toolName
    if (event.type === 'admitted') {
      this._telemetry.recordAdmission(toolName)
      return
    }
    if (event.type === 'executionStarted') {
      this._telemetry.recordExecutionStarted({
        toolName,
        attempt: event.task.attemptCount,
        resumed: event.resumed,
        queueDuration: event.queueDuration,
      })
      return
    }
    if (event.type === 'executionFinished') {
      this._telemetry.recordExecutionFinished({
        toolName,
        outcome:
          event.task.status === 'failed' && event.task.failure?.type === 'executionError'
            ? 'executionError'
            : event.task.status === 'queued' || event.task.status === 'working'
              ? 'executionError'
              : event.task.status,
        duration: event.duration,
      })
      if (event.task.failure) this._telemetry.recordFailure(toolName, event.task.failure.type)
      if (event.task.status === 'completed' || event.task.status === 'failed') {
        this._telemetry.recordTerminal(toolName, event.task.status)
      }
      return
    }
    if (event.type === 'cancelled') {
      this._telemetry.recordTerminal(toolName, 'cancelled')
    }
  }

  private async _waitForDeliveries(timeout: number): Promise<void> {
    const deadline = Date.now() + timeout
    while (this._delivering.size > 0) {
      const remaining = deadline - Date.now()
      if (remaining <= 0) throw new Error(`Background Tasks state reload timed out after ${timeout}ms`)
      let timer: ReturnType<typeof setTimeout> | undefined
      await Promise.race([
        new Promise<void>((resolve) => {
          this._deliveryWaiters.add(resolve)
        }),
        new Promise<never>((_, reject) => {
          timer = setTimeout(
            () => reject(new Error(`Background Tasks state reload timed out after ${timeout}ms`)),
            remaining
          )
        }),
      ]).finally(() => {
        if (timer) clearTimeout(timer)
      })
    }
  }

  private async _waitForTaskResult(waitForAll: boolean): Promise<void> {
    const cancelSignal = this._agent.cancelSignal
    while (!cancelSignal.aborted) {
      const tasks = this._engine.list()
      if (tasks.some((task) => task.status === 'paused')) return
      if (
        !waitForAll &&
        tasks.some(
          (task) =>
            isEngineTerminalStatus(task.status) &&
            this._deliveryStates.get(task.taskId) === 'ready' &&
            !this._delivering.has(task.taskId)
        )
      ) {
        return
      }

      const pending = tasks.filter((task) => task.status === 'queued' || task.status === 'working')
      if (pending.length === 0) return

      const observationController = new AbortController()
      const observationSignal = AbortSignal.any([cancelSignal, observationController.signal])
      try {
        await Promise.race(pending.map((task) => this._engine.wait(task.taskId, { cancelSignal: observationSignal })))
      } catch (error) {
        if (!cancelSignal.aborted) throw error
      } finally {
        observationController.abort()
      }
    }
  }

  private _routeInterruptResponses(event: BeforeModelCallEvent): void {
    const interruptState = event._getInterruptState()
    const responseContents = interruptState?.resumeResponses
    if (!interruptState || !responseContents || responseContents.length === 0) return

    const paused = this._engine.list().filter((task) => task.status === 'paused')
    const taskByInterruptId = new Map<string, string>()
    for (const task of paused) {
      for (const interrupt of toBackgroundTask(task).interrupts ?? []) {
        const owner = taskByInterruptId.get(interrupt.id)
        if (owner && owner !== task.taskId) {
          throw new Error(`Background interrupt '${interrupt.id}' is ambiguous across paused tasks`)
        }
        taskByInterruptId.set(interrupt.id, task.taskId)
      }
    }

    const responsesByTask = new Map<string, InterruptResponse[]>()
    for (const content of responseContents) {
      const response = content.interruptResponse
      const taskId = taskByInterruptId.get(response.interruptId)
      if (!taskId) continue
      const responses = responsesByTask.get(taskId) ?? []
      responses.push(response)
      responsesByTask.set(taskId, responses)
    }
    if (responsesByTask.size === 0) return

    for (const [taskId, responses] of responsesByTask) {
      this._resumeTask(taskId, responses)
    }

    const foregroundInterruptIds = Object.keys(interruptState.interrupts)
    if (
      foregroundInterruptIds.length > 0 &&
      foregroundInterruptIds.every((interruptId) => taskByInterruptId.has(interruptId))
    ) {
      interruptState.deactivate()
    }
  }

  private _throwPausedInterrupts(): void {
    const interrupts = this._pausedInterrupts()
    if (interrupts.length > 0) throw new InterruptError(interrupts)
  }

  private _pausedInterrupts(): Interrupt[] {
    return this._engine
      .list()
      .filter((task) => task.status === 'paused')
      .flatMap((task) => toBackgroundTask(task).interrupts ?? [])
  }
}

async function waitWithSignal<Value>(promise: Promise<Value>, cancelSignal?: AbortSignal): Promise<Value> {
  if (!cancelSignal) return promise
  if (cancelSignal.aborted) {
    throw cancelSignal.reason ?? new DOMException('Observation aborted', 'AbortError')
  }
  return new Promise<Value>((resolve, reject) => {
    const onAbort = (): void => {
      reject(cancelSignal.reason ?? new DOMException('Observation aborted', 'AbortError'))
    }
    cancelSignal.addEventListener('abort', onAbort, { once: true })
    void promise.then(
      (value) => {
        cancelSignal.removeEventListener('abort', onAbort)
        resolve(value)
      },
      (error: unknown) => {
        cancelSignal.removeEventListener('abort', onAbort)
        reject(error)
      }
    )
  })
}

function deliveryStateFor(
  record: StoredBackgroundTask,
  current?: 'pending' | 'ready' | 'delivered'
): 'pending' | 'ready' | 'delivered' {
  if (!isEngineTerminalStatus(record.status)) return 'pending'
  return current === 'delivered' ? 'delivered' : 'ready'
}

function isObject(value: unknown): value is { [key: string]: JSONValue } {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function responsesAlreadyApplied(state: InterruptStateData, responses: readonly InterruptResponse[]): boolean {
  return responses.every(
    (response) =>
      stableStringify(state.interrupts[response.interruptId]?.response) === stableStringify(response.response)
  )
}
