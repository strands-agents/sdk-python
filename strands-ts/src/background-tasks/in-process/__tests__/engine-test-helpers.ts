import { expect } from 'vitest'
import { InProcessTaskEngine } from '../engine.js'
import type { InterruptStateData } from '../../../interrupt.js'
import type { ToolResultBlockData } from '../../../types/messages.js'
import type {
  InProcessTaskDescriptor,
  InProcessTaskEngineOptions,
  InProcessTaskRecord,
  TaskExecutionOutcome,
} from '../types.js'

export type TestOutcome = TaskExecutionOutcome
export type TestTask = InProcessTaskRecord

type ExpectedTaskFields = Pick<TestTask, 'status'> & Partial<Pick<TestTask, 'state' | 'result' | 'failure'>>

export function createEngine(
  execute: InProcessTaskEngineOptions['execute'],
  options: Partial<InProcessTaskEngineOptions> = {}
): InProcessTaskEngine {
  return new InProcessTaskEngine({
    maxConcurrency: 2,
    timeout: 1_000,
    execute,
    onTaskUpdated: () => undefined,
    ...options,
  })
}

export function createDescriptor(value: string): InProcessTaskDescriptor {
  return {
    toolUseId: value,
    toolName: value,
    invocationStateId: value,
  }
}

export function createResult(value: string): ToolResultBlockData {
  return {
    toolUseId: value,
    status: 'success',
    content: [{ text: value }],
  }
}

export function createState(value: string): InterruptStateData {
  return {
    interrupts: {
      [value]: {
        id: value,
        name: value,
      },
    },
    activated: true,
  }
}

export function getStateValue(state: InterruptStateData): string {
  return Object.keys(state.interrupts)[0]!
}

export function deferred<Value>(): { promise: Promise<Value>; resolve(value: Value): void } {
  let resolve!: (value: Value) => void
  const promise = new Promise<Value>((done) => {
    resolve = done
  })
  return { promise, resolve }
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
