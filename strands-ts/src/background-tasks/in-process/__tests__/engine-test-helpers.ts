import { expect } from 'vitest'
import { InProcessTaskEngine } from '../engine.js'
import type {
  InProcessTaskEngineConfig,
  InProcessTaskExecutionContext,
  StoredInProcessTask,
  TaskExecutionOutcome,
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

export type TestContext = InProcessTaskExecutionContext<TestDescriptor, TestState>
export type TestOutcome = TaskExecutionOutcome<TestResult, TestState>
export type TestEngine = InProcessTaskEngine<TestDescriptor, TestResult, TestState>
export type TestTask = StoredInProcessTask<TestDescriptor, TestResult, TestState>

type ExpectedTaskFields = Pick<TestTask, 'status'> & Partial<Pick<TestTask, 'state' | 'result' | 'failure'>>

const engines = new Set<TestEngine>()

export function createEngine(
  execute: (context: TestContext) => Promise<TestOutcome>,
  options: Partial<InProcessTaskEngineConfig<TestDescriptor, TestResult, TestState>> = {}
): TestEngine {
  const engine = new InProcessTaskEngine<TestDescriptor, TestResult, TestState>({
    maxConcurrency: 2,
    timeout: 1_000,
    execute,
    onTaskUpdated: () => undefined,
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
  await Promise.allSettled([...engines].map((engine) => engine.shutdown({ timeout: 1_000 })))
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
    createdAt: now,
    updatedAt: now,
    ...overrides,
  }
}

export function expectTask(actual: TestTask | undefined, task: TestTask, fields: ExpectedTaskFields): void {
  expect(actual).toEqual({
    taskId: task.taskId,
    ...(task.idempotencyKey !== undefined && { idempotencyKey: task.idempotencyKey }),
    descriptor: task.descriptor,
    ...fields,
    createdAt: task.createdAt,
    updatedAt: expect.any(String),
  })
}
