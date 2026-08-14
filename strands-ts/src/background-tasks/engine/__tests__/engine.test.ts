import { afterEach, describe, expect, it, vi } from 'vitest'
import { BackgroundTaskEngine } from '../engine.js'
import type {
  BackgroundTaskEngineConfig,
  BackgroundTaskExecutionContext,
  BackgroundTaskExecutionOutcome,
  StoredEngineTask,
} from '../types.js'

interface TestDescriptor {
  readonly value: string
}

interface TestResult {
  readonly value: string
}

interface TestState {
  readonly phase: string
}

type TestContext = BackgroundTaskExecutionContext<TestDescriptor, TestState>
type TestOutcome = BackgroundTaskExecutionOutcome<TestResult, TestState>
type TestEngine = BackgroundTaskEngine<TestDescriptor, TestResult, TestState>

const engines = new Set<TestEngine>()

afterEach(async () => {
  await Promise.allSettled([...engines].map((engine) => engine.shutdown({ mode: 'cancel', timeout: 1_000 })))
  engines.clear()
})

function createEngine(
  execute: (context: TestContext) => Promise<TestOutcome>,
  options: Partial<BackgroundTaskEngineConfig<TestDescriptor, TestResult, TestState>> = {}
): TestEngine {
  const engine = new BackgroundTaskEngine<TestDescriptor, TestResult, TestState>({
    maxConcurrency: 2,
    timeout: 1_000,
    execute,
    ...options,
  })
  engines.add(engine)
  return engine
}

function initialize(
  engine: TestEngine,
  restoredTasks: readonly StoredEngineTask<TestDescriptor, TestResult, TestState>[] = []
): TestEngine {
  engine.initialize(restoredTasks)
  return engine
}

async function waitForStatus(
  engine: TestEngine,
  taskId: string,
  status: StoredEngineTask['status']
): Promise<StoredEngineTask<TestDescriptor, TestResult, TestState>> {
  for (let count = 0; count < 100; count++) {
    const task = engine.get(taskId)
    if (task?.status === status) return task
    await Promise.resolve()
  }
  throw new Error(`Task '${taskId}' did not reach '${status}'`)
}

function deferred<Value>(): { promise: Promise<Value>; resolve(value: Value): void } {
  let resolve!: (value: Value) => void
  const promise = new Promise<Value>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

function abortable(context: TestContext): Promise<TestOutcome> {
  return new Promise((_resolve, reject) => {
    context.cancelSignal.addEventListener('abort', () => reject(context.cancelSignal.reason), { once: true })
  })
}

describe('BackgroundTaskEngine', () => {
  it('executes, reports updates, lists, and deduplicates work', async () => {
    const statuses: StoredEngineTask['status'][] = []
    const engine = initialize(
      createEngine(
        async (context) => ({
          status: 'completed',
          result: { value: context.descriptor.value.toUpperCase() },
        }),
        {
          onTaskUpdated: (task) => statuses.push(task.status),
        }
      )
    )
    const admitted = engine.submit({
      descriptor: { value: 'hello' },
      idempotencyKey: 'work-1',
    })
    const duplicate = engine.submit({
      descriptor: { value: 'hello' },
      idempotencyKey: 'work-1',
    })

    expect(duplicate.taskId).toBe(admitted.taskId)
    expect(await engine.wait(admitted.taskId)).toEqual(
      expect.objectContaining({ status: 'completed', result: { value: 'HELLO' }, attemptCount: 1 })
    )
    expect(engine.list()).toHaveLength(1)
    expect(statuses).toEqual(['queued', 'working', 'completed'])

    const external = engine.get(admitted.taskId)
    ;(external!.result as { value: string }).value = 'changed'
    expect(engine.get(admitted.taskId)?.result).toEqual({ value: 'HELLO' })
  })

  it('bounds execution concurrency', async () => {
    const releases = [deferred<TestOutcome>(), deferred<TestOutcome>(), deferred<TestOutcome>()]
    let active = 0
    let maximum = 0
    const engine = initialize(
      createEngine(
        async (context) => {
          active++
          maximum = Math.max(maximum, active)
          const outcome = await releases[Number(context.descriptor.value)]!.promise
          active--
          return outcome
        },
        { maxConcurrency: 2 }
      )
    )
    const tasks = ['0', '1', '2'].map((value) => engine.submit({ descriptor: { value } }))
    await waitForStatus(engine, tasks[0]!.taskId, 'working')
    await waitForStatus(engine, tasks[1]!.taskId, 'working')
    expect(engine.get(tasks[2]!.taskId)?.status).toBe('queued')

    releases[0]!.resolve({ status: 'completed', result: { value: '0' } })
    await waitForStatus(engine, tasks[2]!.taskId, 'working')
    releases[1]!.resolve({ status: 'completed', result: { value: '1' } })
    releases[2]!.resolve({ status: 'completed', result: { value: '2' } })
    await Promise.all(tasks.map((task) => engine.wait(task.taskId)))
    expect(maximum).toBe(2)
  })

  it('allows idle observation to stop without cancelling work', async () => {
    const finish = deferred<TestOutcome>()
    const engine = initialize(createEngine(async () => finish.promise))
    const admitted = engine.submit({ descriptor: { value: 'work' } })
    const observationController = new AbortController()
    const waiting = engine.waitForIdle({ cancelSignal: observationController.signal })

    observationController.abort(new Error('stop observing'))

    await expect(waiting).rejects.toThrow('stop observing')
    expect(engine.get(admitted.taskId)?.status).toBe('working')

    finish.resolve({ status: 'completed', result: { value: 'done' } })
    await expect(engine.waitForIdle()).resolves.toBeUndefined()
  })

  it('pauses, persists state, and resumes the same logical attempt', async () => {
    const contexts: TestContext[] = []
    const engine = initialize(
      createEngine(async (context) => {
        contexts.push(context)
        return context.state
          ? { status: 'completed', result: { value: 'resumed' } }
          : { status: 'paused', state: { phase: 'waiting' } }
      })
    )
    const admitted = engine.submit({ descriptor: { value: 'work' } })
    const paused = await engine.wait(admitted.taskId)
    expect(paused).toEqual(expect.objectContaining({ status: 'paused', state: { phase: 'waiting' } }))

    engine.resume(admitted.taskId, (state) => ({ state, ready: true }))
    expect(await engine.wait(admitted.taskId)).toEqual(
      expect.objectContaining({ status: 'completed', result: { value: 'resumed' }, attemptCount: 1 })
    )
    expect(contexts[0]!.attemptId).toBe(contexts[1]!.attemptId)
    expect(contexts[0]!.executionId).not.toBe(contexts[1]!.executionId)
  })

  it('cancels running work and wakes waiters', async () => {
    const engine = initialize(createEngine(abortable))
    const admitted = engine.submit({ descriptor: { value: 'work' } })
    await waitForStatus(engine, admitted.taskId, 'working')
    const waiting = engine.wait(admitted.taskId)

    engine.cancel(admitted.taskId, { reason: 'Stop work' })

    expect(await waiting).toEqual(
      expect.objectContaining({
        status: 'cancelled',
        cancellationReason: 'Stop work',
      })
    )
  })

  it('removes a delivered cancellation after its active execution settles', async () => {
    const finish = deferred<TestOutcome>()
    const events: string[] = []
    const engine = initialize(
      createEngine(async () => finish.promise, {
        onEvent: (event) => events.push(event.type),
      })
    )
    const admitted = engine.submit({
      descriptor: { value: 'work' },
      idempotencyKey: 'delivered-cancellation',
    })
    await waitForStatus(engine, admitted.taskId, 'working')

    engine.cancel(admitted.taskId, { reason: 'Stop work' })
    engine.remove(admitted.taskId)

    expect(engine.get(admitted.taskId)).toBeUndefined()
    expect(engine.list()).toEqual([])
    expect(() => engine.cancel(admitted.taskId, { reason: 'Again' })).toThrow(
      `Background task '${admitted.taskId}' was not found`
    )

    finish.resolve({ status: 'completed', result: { value: 'late' } })
    await expect(engine.shutdown({ mode: 'drain', timeout: 1_000 })).resolves.toBeUndefined()
    expect(events).toContain('executionFinished')
    expect(engine.get(admitted.taskId)).toBeUndefined()
  })

  it('times out work and records classified execution failures', async () => {
    const timeoutEngine = initialize(createEngine(abortable, { timeout: 10 }))
    const timed = timeoutEngine.submit({ descriptor: { value: 'timeout' } })
    expect(await timeoutEngine.wait(timed.taskId)).toEqual(
      expect.objectContaining({
        status: 'failed',
        failure: { type: 'timeout', message: 'Timed out after 10ms' },
      })
    )

    const executionErrorEngine = initialize(
      createEngine(async () => {
        throw new TypeError('Execution exploded')
      })
    )
    const executionError = executionErrorEngine.submit({ descriptor: { value: 'throw' } })
    expect(await executionErrorEngine.wait(executionError.taskId)).toEqual(
      expect.objectContaining({
        status: 'failed',
        failure: { type: 'executionError', message: 'Execution exploded' },
      })
    )

    const failureEngine = initialize(
      createEngine(async () => ({
        status: 'failed',
        failure: { type: 'toolError', message: 'Tool failed' },
        result: { value: 'tool detail' },
      }))
    )
    const failed = failureEngine.submit({ descriptor: { value: 'work' } })
    expect(await failureEngine.wait(failed.taskId)).toEqual(
      expect.objectContaining({
        status: 'failed',
        failure: { type: 'toolError', message: 'Tool failed' },
        result: { value: 'tool detail' },
      })
    )

    const releaseHung = deferred<TestOutcome>()
    let activeExecutions = 0
    let maximumExecutions = 0
    let hungSignal: AbortSignal | undefined
    const nonCooperative = initialize(
      createEngine(
        async (context) => {
          activeExecutions += 1
          maximumExecutions = Math.max(maximumExecutions, activeExecutions)
          try {
            if (context.descriptor.value === 'hang') {
              hungSignal = context.cancelSignal
              return await releaseHung.promise
            }
            return { status: 'completed', result: { value: 'next' } }
          } finally {
            activeExecutions -= 1
          }
        },
        {
          maxConcurrency: 1,
          timeout: 10,
        }
      )
    )
    const hung = nonCooperative.submit({ descriptor: { value: 'hang' } })
    const next = nonCooperative.submit({ descriptor: { value: 'next' } })
    expect(await nonCooperative.wait(hung.taskId)).toEqual(
      expect.objectContaining({ status: 'failed', failure: expect.objectContaining({ type: 'timeout' }) })
    )
    try {
      await vi.waitFor(() => expect(hungSignal?.aborted).toBe(true))
      expect(nonCooperative.get(next.taskId)).toEqual(expect.objectContaining({ status: 'queued' }))
      expect(maximumExecutions).toBe(1)
    } finally {
      releaseHung.resolve({ status: 'completed', result: { value: 'late' } })
    }

    expect(await nonCooperative.wait(next.taskId)).toEqual(
      expect.objectContaining({ status: 'completed', result: { value: 'next' } })
    )
    expect(nonCooperative.get(hung.taskId)).toEqual(
      expect.objectContaining({ status: 'failed', failure: expect.objectContaining({ type: 'timeout' }) })
    )
  })

  it('allows Infinity to disable the execution timeout', async () => {
    const finish = deferred<TestOutcome>()
    const timeoutSpy = vi.spyOn(globalThis, 'setTimeout')
    const engine = initialize(createEngine(async () => finish.promise, { timeout: Infinity }))
    const task = engine.submit({ descriptor: { value: 'work' } })

    await waitForStatus(engine, task.taskId, 'working')
    expect(timeoutSpy).not.toHaveBeenCalled()

    finish.resolve({ status: 'completed', result: { value: 'done' } })
    expect(await engine.wait(task.taskId)).toEqual(
      expect.objectContaining({ status: 'completed', result: { value: 'done' } })
    )
    timeoutSpy.mockRestore()
  })

  it('recovers terminal outcomes without executing them again', async () => {
    const records = new Map<string, StoredEngineTask<TestDescriptor, TestResult, TestState>>()
    let executions = 0
    const first = initialize(
      createEngine(
        async () => {
          executions++
          return { status: 'completed', result: { value: 'done' } }
        },
        {
          onTaskUpdated: (task) => records.set(task.taskId, task),
        }
      )
    )
    const admitted = first.submit({ descriptor: { value: 'work' } })
    await first.wait(admitted.taskId)
    await first.shutdown({ mode: 'drain', timeout: 1_000 })
    engines.delete(first)

    const second = initialize(
      createEngine(async () => {
        executions++
        return { status: 'completed', result: { value: 'duplicate' } }
      }),
      [...records.values()]
    )
    expect(second.get(admitted.taskId)).toEqual(
      expect.objectContaining({ status: 'completed', result: { value: 'done' } })
    )
    expect(executions).toBe(1)
  })

  it('drains or cancels cleanly during shutdown', async () => {
    const finish = deferred<TestOutcome>()
    const draining = initialize(createEngine(async () => finish.promise))
    const task = draining.submit({ descriptor: { value: 'drain' } })
    const shutdown = draining.shutdown({ mode: 'drain', timeout: 1_000 })
    expect(() => draining.submit({ descriptor: { value: 'rejected' } })).toThrow(/admission is closed/)
    finish.resolve({ status: 'completed', result: { value: 'done' } })
    await expect(shutdown).resolves.toBeUndefined()
    expect(draining.get(task.taskId)?.status).toBe('completed')

    const cancelling = initialize(createEngine(abortable))
    const cancelled = cancelling.submit({ descriptor: { value: 'cancel' } })
    await waitForStatus(cancelling, cancelled.taskId, 'working')
    await expect(cancelling.shutdown({ mode: 'cancel', timeout: 1_000 })).resolves.toBeUndefined()
    expect(cancelling.get(cancelled.taskId)?.status).toBe('cancelled')
  })

  it('keeps paused work stopped after drain shutdown', async () => {
    let executions = 0
    const engine = initialize(
      createEngine(async (context) => {
        executions++
        return context.state
          ? { status: 'completed', result: { value: 'resumed' } }
          : { status: 'paused', state: { phase: 'waiting' } }
      })
    )
    const admitted = engine.submit({ descriptor: { value: 'work' } })
    await engine.wait(admitted.taskId)
    await engine.shutdown({ mode: 'drain', timeout: 1_000 })

    expect(() =>
      engine.resume(admitted.taskId, (state) => ({
        state,
        ready: true,
      }))
    ).toThrow('Background task execution is closed')

    expect(engine.get(admitted.taskId)).toEqual(
      expect.objectContaining({ status: 'paused', state: { phase: 'waiting' } })
    )
    expect(executions).toBe(1)
  })
})
