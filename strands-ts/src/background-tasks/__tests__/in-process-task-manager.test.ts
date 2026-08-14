import { afterEach, describe, expect, it, vi } from 'vitest'
import { metrics as otelMetrics, type Meter as OtelMeter } from '@opentelemetry/api'
import { z } from 'zod'

import { MockMeter } from '../../__fixtures__/mock-meter.js'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { Agent } from '../../agent/agent.js'
import { tool } from '../../tools/tool-factory.js'
import type { ToolContext } from '../../tools/tool.js'
import { InProcessTaskManager } from '../in-process-task-manager.js'
import type { BackgroundTask, BackgroundTasksConfig } from '../types.js'

function deferred<Value>(): { readonly promise: Promise<Value>; resolve(value: Value): void } {
  let resolve!: (value: Value) => void
  const promise = new Promise<Value>((promiseResolve) => {
    resolve = promiseResolve
  })
  return { promise, resolve }
}

const managers = new Set<InProcessTaskManager>()

afterEach(async () => {
  await Promise.allSettled([...managers].map(cancelManagerTasks))
  managers.clear()
  vi.restoreAllMocks()
})

async function cancelManagerTasks(manager: InProcessTaskManager): Promise<void> {
  const tasks = await manager.listTasks()
  await Promise.allSettled(
    tasks
      .filter((task) => task.status === 'queued' || task.status === 'working' || task.status === 'paused')
      .map((task) => manager.cancelTask(task.taskId))
  )
  await manager.waitForTasks({ timeout: 1_000 })
}

async function createManager(
  callback: (input: { value: string }, context?: ToolContext) => string | Promise<string>,
  options: BackgroundTasksConfig = {}
): Promise<InProcessTaskManager> {
  const agent = new Agent({
    id: 'manager-test-agent',
    model: new MockMessageModel().addTurn({ type: 'textBlock', text: 'unused' }),
    tools: [
      tool({
        name: 'work',
        description: 'Perform controlled test work.',
        inputSchema: z.object({ value: z.string() }),
        callback,
      }),
    ],
    printer: false,
  })
  const manager = new InProcessTaskManager(agent, options)
  managers.add(manager)
  await manager.initialize()
  return manager
}

function admit(manager: InProcessTaskManager, value: string, passId?: string) {
  return manager.submitTask({
    kind: 'toolCall',
    toolName: 'work',
    originalToolUseId: `tool-use-${value}`,
    input: { value },
    invocationState: { request: value },
    passId: passId ?? globalThis.crypto.randomUUID(),
  })
}

async function waitForTask(manager: InProcessTaskManager, taskId: string): Promise<BackgroundTask> {
  let task: BackgroundTask | undefined
  await vi.waitFor(async () => {
    task = await manager.getTask(taskId)
    expect(task?.status).toMatch(/^(paused|completed|failed|cancelled)$/)
  })
  return task!
}

describe('InProcessTaskManager', () => {
  describe('execution', () => {
    it('executes admitted tool work and exposes the minimal public snapshot', async () => {
      const manager = await createManager(({ value }) => value.toUpperCase())

      const admitted = await admit(manager, 'hello')
      const completed = await waitForTask(manager, admitted.taskId)

      expect(completed).toEqual({
        taskId: admitted.taskId,
        toolUseId: 'tool-use-hello',
        toolName: 'work',
        status: 'completed',
        createdAt: expect.any(String),
        updatedAt: expect.any(String),
        result: { content: [{ text: 'HELLO' }] },
      })
    })

    it('passes JSON invocation state to the tool', async () => {
      const invocationStates: ToolContext['invocationState'][] = []
      const manager = await createManager((_input, context) => {
        invocationStates.push(context!.invocationState)
        return 'done'
      })

      const admitted = await manager.submitTask({
        kind: 'toolCall',
        toolName: 'work',
        originalToolUseId: 'tool-use-json-state',
        input: { value: 'json-state' },
        invocationState: { requestId: 'request-1', tenant: { id: 'tenant-1' } },
        passId: globalThis.crypto.randomUUID(),
      })
      await waitForTask(manager, admitted.taskId)
      expect(invocationStates).toEqual([{ requestId: 'request-1', tenant: { id: 'tenant-1' } }])
    })

    it('records queued and running cancellations as terminal exactly once', async () => {
      const mockMeter = new MockMeter()
      vi.spyOn(otelMetrics, 'getMeter').mockReturnValue(mockMeter as unknown as OtelMeter)
      const firstStarted = deferred<void>()
      const firstFinished = deferred<string>()
      const calls: string[] = []
      const manager = await createManager(
        async ({ value }) => {
          calls.push(value)
          if (value === 'first') {
            firstStarted.resolve()
            return firstFinished.promise
          }
          return value
        },
        { maxConcurrency: 1 }
      )

      const first = await admit(manager, 'first')
      await firstStarted.promise
      const second = await admit(manager, 'second')

      await manager.cancelTask(second.taskId)
      await manager.cancelTask(first.taskId)

      const cancellations = mockMeter.getCounter('gen_ai.agent.background_task.cancellation.count')
      expect(cancellations?.dataPoints).toEqual([
        {
          value: 1,
          attributes: {
            'gen_ai.tool.name': 'work',
          },
        },
        {
          value: 1,
          attributes: {
            'gen_ai.tool.name': 'work',
          },
        },
      ])

      const terminal = mockMeter.getCounter('gen_ai.agent.background_task.terminal.count')
      expect(terminal?.dataPoints).toEqual([
        {
          value: 1,
          attributes: {
            'gen_ai.tool.name': 'work',
            'background_task.status': 'cancelled',
          },
        },
        {
          value: 1,
          attributes: {
            'gen_ai.tool.name': 'work',
            'background_task.status': 'cancelled',
          },
        },
      ])

      firstFinished.resolve('late')
      await manager.waitForTasks({ timeout: 1_000 })
      expect(terminal?.dataPoints).toHaveLength(2)
      expect(calls).toEqual(['first'])
    })

    it('does not record cancellation when a terminal task is unchanged', async () => {
      const mockMeter = new MockMeter()
      vi.spyOn(otelMetrics, 'getMeter').mockReturnValue(mockMeter as unknown as OtelMeter)
      const manager = await createManager(() => 'done')
      const admitted = await admit(manager, 'completed')
      const completed = await waitForTask(manager, admitted.taskId)

      await expect(manager.cancelTask(completed.taskId)).resolves.toEqual(completed)
      await expect(manager.cancelTask(completed.taskId)).resolves.toEqual(completed)

      expect(mockMeter.getCounter('gen_ai.agent.background_task.cancellation.count')?.dataPoints).toEqual([])
    })

    it('deduplicates the same tool call admission', async () => {
      const finished = deferred<string>()
      const manager = await createManager(() => finished.promise)
      const first = await admit(manager, 'same', 'same-pass')
      const duplicate = await admit(manager, 'same', 'same-pass')

      expect(duplicate.taskId).toBe(first.taskId)
      finished.resolve('done')
      await waitForTask(manager, first.taskId)
    })
  })
})
