import { describe, expect, it } from 'vitest'
import { z } from 'zod'

import { createMockAgent } from '../../../__fixtures__/agent-helpers.js'
import { tool } from '../../../tools/tool-factory.js'
import type { Tool, ToolContext } from '../../../tools/tool.js'
import { InProcessTaskManager } from '../manager.js'

function createFixture(
  callback: (
    input: { value: string },
    context?: ToolContext
  ) => AsyncGenerator<unknown, string, never> | string | Promise<string>
): { readonly manager: InProcessTaskManager; readonly work: Tool } {
  const work = tool({
    name: 'work',
    description: 'Perform controlled test work.',
    inputSchema: z.object({ value: z.string() }),
    callback,
  })
  const manager = new InProcessTaskManager(createMockAgent(), async (selectedTool, context) => {
    const stream = selectedTool.stream(context)
    let next = await stream.next()
    while (!next.done) next = await stream.next()
    return next.value
  })
  return { manager, work }
}

function submit(manager: InProcessTaskManager, work: Tool, value: string, toolUseId = `tool-use-${value}`) {
  return manager.submitTask({ name: 'work', toolUseId, input: { value } }, {}, work)
}

describe('InProcessTaskManager', () => {
  it('executes the selected tool with transient invocation state', async () => {
    const invocationState: ToolContext['invocationState'] = {
      requestId: 'request-1',
      callback: (): string => 'not serializable',
    }
    const { manager, work } = createFixture(async function* ({ value }, context) {
      expect(context?.invocationState).toBe(invocationState)
      expect(context?.toolUse.name).toBe('hook-alias')
      yield 'progress'
      return value.toUpperCase()
    })

    const admitted = await manager.submitTask(
      { name: 'hook-alias', toolUseId: 'tool-use-1', input: { value: 'hello' } },
      invocationState,
      work
    )

    await manager.waitForTasks()
    const completed = {
      taskId: admitted.taskId,
      toolUseId: 'tool-use-1',
      toolName: 'work',
      status: 'completed',
      createdAt: expect.any(String),
      lastUpdatedAt: expect.any(String),
      result: { content: [{ text: 'HELLO' }] },
    }
    expect(await manager.getTask(admitted.taskId)).toEqual(completed)
    expect(await manager.listTasks()).toEqual([completed])
    await manager.consumeTasks([admitted.taskId])
    await expect(manager.getTask(admitted.taskId)).resolves.toBeUndefined()
  })

  it('projects tool errors', async () => {
    const { manager, work } = createFixture(() => {
      throw new Error('tool failed')
    })
    const failed = await submit(manager, work, 'failed')

    await expect(manager.waitForTask(failed.taskId)).resolves.toEqual({
      ...failed,
      status: 'failed',
      lastUpdatedAt: expect.any(String),
      result: { content: [{ text: 'Error: tool failed' }] },
      error: { type: 'toolError', message: 'tool failed' },
    })
  })

  it('cancels tool execution', async () => {
    let started!: () => void
    const toolStarted = new Promise<void>((resolve) => {
      started = resolve
    })
    let cancelSignal: AbortSignal | undefined
    const { manager, work } = createFixture(async (_input, context) => {
      cancelSignal = context!.cancelSignal
      started()
      await new Promise<void>((resolve) => cancelSignal!.addEventListener('abort', () => resolve(), { once: true }))
      return 'late'
    })
    const admitted = await submit(manager, work, 'cancel')
    await toolStarted

    await expect(manager.cancelTask(admitted.taskId)).resolves.toEqual({
      ...admitted,
      status: 'cancelled',
      lastUpdatedAt: expect.any(String),
    })
    expect(cancelSignal).toMatchObject({ aborted: true, reason: 'Cancellation requested' })
    await manager.waitForTasks()
  })

  it('requests input and resumes tool interrupts', async () => {
    const { manager, work } = createFixture(({ value }, context) => {
      if (value === 'complete') return 'complete'
      const response = context!.interrupt<string>({ name: 'approve_work', reason: 'Approve work?' })
      return `approved:${response}`
    })
    const completed = await submit(manager, work, 'complete')
    await manager.waitForTask(completed.taskId)
    const admitted = await submit(manager, work, 'interrupt', 'tool-use-1')
    const inputRequired = await manager.waitForTask(admitted.taskId)
    expect(inputRequired).toEqual({
      ...admitted,
      status: 'input_required',
      lastUpdatedAt: expect.any(String),
      interrupts: [
        {
          id: expect.any(String),
          name: 'approve_work',
          reason: 'Approve work?',
          source: 'tool',
        },
      ],
    })
    const interrupt = inputRequired.interrupts?.[0]
    if (!interrupt) throw new Error('Expected task interrupt')

    await expect(manager.consumeTasks([completed.taskId, admitted.taskId])).rejects.toThrow(
      `Background task '${admitted.taskId}' cannot be consumed before reaching a terminal status`
    )
    await expect(manager.getTask(completed.taskId)).resolves.toBeDefined()

    await expect(
      manager.resumeTask(admitted.taskId, [{ interruptId: interrupt.id, response: 'yes' }])
    ).resolves.toEqual({
      ...admitted,
      status: 'queued',
      lastUpdatedAt: expect.any(String),
    })
    await expect(manager.waitForTask(admitted.taskId)).resolves.toEqual({
      ...admitted,
      status: 'completed',
      lastUpdatedAt: expect.any(String),
      result: { content: [{ text: 'approved:yes' }] },
    })
  })
})
