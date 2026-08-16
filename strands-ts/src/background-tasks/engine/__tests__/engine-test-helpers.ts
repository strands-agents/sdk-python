import { expect } from 'vitest'
import { BackgroundTaskEngine } from '../engine.js'
import type {
  BackgroundTaskEngineConfig,
  BackgroundTaskExecutionContext,
  BackgroundTaskExecutionOutcome,
  StoredEngineTask,
} from '../types.js'

export interface TestDescriptor {
  readonly value: string
}

export interface TestResult {
  readonly value: string
}

export interface TestState {
  readonly phase: string
}

export type TestContext = BackgroundTaskExecutionContext<TestDescriptor, TestState>
export type TestOutcome = BackgroundTaskExecutionOutcome<TestResult, TestState>
export type TestEngine = BackgroundTaskEngine<TestDescriptor, TestResult, TestState>
export type TestTask = StoredEngineTask<TestDescriptor, TestResult, TestState>

type ExpectedTaskFields = Pick<TestTask, 'status'> &
  Partial<Record<'attemptCount' | 'attemptId' | 'cancellationReason' | 'state' | 'result' | 'failure', unknown>>

const engines = new Set<TestEngine>()

export function createEngine(
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

export function initialize(engine: TestEngine, restoredTasks: readonly TestTask[] = []): TestEngine {
  engine.initialize(restoredTasks)
  return engine
}

export async function shutdownEngines(): Promise<void> {
  await Promise.allSettled([...engines].map((engine) => engine.shutdown({ mode: 'cancel', timeout: 1_000 })))
  engines.clear()
}

export async function waitForStatus(engine: TestEngine, taskId: string, status: TestTask['status']): Promise<TestTask> {
  for (let count = 0; count < 100; count++) {
    const task = engine.get(taskId)
    if (task?.status === status) return task
    await Promise.resolve()
  }
  throw new Error(`Task '${taskId}' did not reach '${status}'`)
}

export function deferred<Value>(): { promise: Promise<Value>; resolve(value: Value): void } {
  let resolve!: (value: Value) => void
  const promise = new Promise<Value>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

export function createStoredTask(overrides: Partial<TestTask> & Pick<TestTask, 'status'>): TestTask {
  const now = new Date().toISOString()
  return {
    taskId: globalThis.crypto.randomUUID(),
    descriptor: { value: 'restored' },
    attemptCount: 0,
    createdAt: now,
    updatedAt: now,
    ...overrides,
  }
}

export function expectTask(actual: TestTask | undefined, task: TestTask, fields: ExpectedTaskFields): void {
  const { status, attemptCount = status === 'queued' ? 0 : 1, ...rest } = fields
  expect(actual).toEqual({
    taskId: task.taskId,
    ...(task.idempotencyKey !== undefined && { idempotencyKey: task.idempotencyKey }),
    descriptor: task.descriptor,
    status,
    attemptCount,
    ...((status === 'working' || status === 'paused') && { attemptId: expect.any(String) }),
    ...rest,
    createdAt: task.createdAt,
    updatedAt: expect.any(String),
  })
}

export function abortable(context: TestContext): Promise<TestOutcome> {
  return new Promise((_resolve, reject) => {
    context.cancelSignal.addEventListener('abort', () => reject(context.cancelSignal.reason), { once: true })
  })
}
