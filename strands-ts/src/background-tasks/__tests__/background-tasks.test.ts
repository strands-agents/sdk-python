import { afterEach, describe, expect, it, vi } from 'vitest'
import { z } from 'zod'
import { MockMessageModel } from '../../__fixtures__/mock-message-model.js'
import { Agent } from '../../agent/agent.js'
import {
  AfterInvocationEvent,
  AfterToolCallEvent,
  AfterToolsEvent,
  BeforeInvocationEvent,
  BeforeToolCallEvent,
} from '../../hooks/events.js'
import { HookOrder } from '../../hooks/types.js'
import { logger } from '../../logging/logger.js'
import { Checkpoint } from '../../experimental/checkpoint.js'
import { McpClient } from '../../mcp/client.js'
import type { StreamOptions } from '../../models/model.js'
import { FunctionTool } from '../../tools/function-tool.js'
import { McpTool } from '../../tools/mcp-tool.js'
import { tool } from '../../tools/tool-factory.js'
import type { Tool } from '../../tools/tool.js'
import type { JSONValue } from '../../types/json.js'
import { InterruptResponseContent } from '../../types/interrupt.js'
import type { Message, ToolResultBlock, ToolUseBlock } from '../../types/messages.js'
import { HumanInTheLoop } from '../../vended-interventions/hitl/hitl.js'
import { BackgroundTaskNotFoundError, BackgroundTasksTimeoutError } from '../errors.js'
import { BACKGROUND_TASKS_STATE_KEY } from '../in-process-task-manager.js'
import { toBackgroundTask, type StoredBackgroundTask } from '../record.js'
import type { BackgroundTask, BackgroundTasksConfig } from '../types.js'

class RecordingModel extends MockMessageModel {
  readonly requests: { readonly messages: Message[]; readonly options?: StreamOptions }[] = []

  override async *stream(
    messages: Message[],
    options?: StreamOptions
  ): AsyncGenerator<import('../../models/streaming.js').ModelStreamEvent> {
    this.requests.push({
      messages: messages.map((message) => message.clone()),
      ...(options && { options }),
    })
    yield* super.stream(messages, options)
  }
}

class StatefulGenericHistoryModel extends RecordingModel {
  override get stateful(): boolean {
    return true
  }
}

afterEach(() => {
  vi.restoreAllMocks()
})

function createBackgroundTasks(mode: 'agentic' | 'always'): BackgroundTasksConfig {
  return { timeout: 5_000, [mode]: ['*'] }
}

function deferred<Value>(): { readonly promise: Promise<Value>; resolve(value: Value): void } {
  let resolve!: (value: Value) => void
  const promise = new Promise<Value>((promiseResolve) => {
    resolve = promiseResolve
  })
  return { promise, resolve }
}

function appStateTasks(agent: Agent): Record<string, BackgroundTask> {
  const entries = agent.appState.get(BACKGROUND_TASKS_STATE_KEY) as
    | Record<
        string,
        {
          readonly record: StoredBackgroundTask
          readonly deliveryState: 'pending' | 'ready' | 'delivered'
        }
      >
    | undefined
  return Object.fromEntries(
    Object.entries(entries ?? {}).map(([taskId, entry]) => [
      taskId,
      {
        ...toBackgroundTask(entry.record),
        delivery: { state: entry.deliveryState },
      },
    ])
  )
}

function taskRecords(agent: Agent): BackgroundTask[] {
  return Object.values(appStateTasks(agent))
}

interface BackgroundDelivery {
  readonly toolUse: ToolUseBlock
  readonly toolResult: ToolResultBlock
}

function backgroundDeliveries(agent: Agent): readonly BackgroundDelivery[] {
  const toolUses: ToolUseBlock[] = []
  const toolResults = new Map<string, ToolResultBlock>()
  for (const message of agent.messages) {
    for (const block of message.content) {
      if (block.type === 'toolUseBlock' && block.name === 'strands_background_task_result') {
        toolUses.push(block)
      } else if (block.type === 'toolResultBlock') {
        toolResults.set(block.toolUseId, block)
      }
    }
  }
  return toolUses.map((toolUse) => {
    const toolResult = toolResults.get(toolUse.toolUseId)
    if (!toolResult) throw new Error(`delivery result '${toolUse.toolUseId}' is missing`)
    return { toolUse, toolResult }
  })
}

async function waitForTaskRecord(agent: Agent, taskId: string): Promise<BackgroundTask> {
  let task: BackgroundTask | undefined
  await vi.waitFor(() => {
    task = appStateTasks(agent)[taskId]
    expect(task?.status).toMatch(/^(paused|completed|failed|cancelled)$/)
  })
  return task!
}

function createGatedWork(): {
  readonly work: Tool
  readonly started: Promise<void>
  readonly release: () => void
} {
  let release!: () => void
  const released = new Promise<void>((resolve) => {
    release = resolve
  })
  let markStarted!: () => void
  const started = new Promise<void>((resolve) => {
    markStarted = resolve
  })
  const work = tool({
    name: 'work',
    description: 'Do delayed work.',
    inputSchema: z.object({}),
    callback: async () => {
      markStarted()
      await released
      return 'complete'
    },
  })
  return { work, started, release }
}

async function observeAfterInvocationWait(agent: Agent): Promise<{
  readonly reachedWait: Promise<void>
}> {
  await agent.initialize()

  let markWaitReached!: () => void
  const reachedWait = new Promise<void>((resolve) => {
    markWaitReached = resolve
  })
  agent.addHook(AfterInvocationEvent, markWaitReached, { order: HookOrder.SDK_FIRST - 1 })
  return { reachedWait }
}

describe('BackgroundTasks', () => {
  describe('configuration', () => {
    it('is disabled when omitted or false and enables model-selected backgrounding with true', async () => {
      const work = tool({
        name: 'work',
        description: 'Do work.',
        inputSchema: z.object({ value: z.string() }),
        callback: () => 'done',
      })
      const disabledModel = new RecordingModel().addTurn({ type: 'textBlock', text: 'Done.' })
      const disabledAgent = new Agent({
        model: disabledModel,
        tools: [work],
        printer: false,
      })

      expect(disabledAgent.backgroundTasks).toBeUndefined()
      await disabledAgent.invoke('work')

      const disabledSpec = disabledModel.requests[0]!.options!.toolSpecs!.find((spec) => spec.name === 'work')!
      expect(disabledSpec.inputSchema?.properties).not.toHaveProperty('_background')

      const backgroundTasksEnabled: boolean = false
      const explicitlyDisabledAgent = new Agent({
        model: new RecordingModel(),
        tools: [work],
        backgroundTasks: backgroundTasksEnabled,
        printer: false,
      })
      expect(explicitlyDisabledAgent.backgroundTasks).toBeUndefined()

      const enabledModel = new RecordingModel().addTurn({ type: 'textBlock', text: 'Done.' })
      const enabledAgent = new Agent({
        model: enabledModel,
        tools: [work],
        backgroundTasks: true,
        printer: false,
      })

      expect(enabledAgent.backgroundTasks).toBeDefined()
      await expect(enabledAgent.backgroundTasks!.list()).resolves.toEqual([])
      await enabledAgent.invoke('work')

      const enabledSpec = enabledModel.requests[0]!.options!.toolSpecs!.find((spec) => spec.name === 'work')!
      expect(enabledSpec.inputSchema?.properties).toHaveProperty('_background')
    })

    it('rejects invalid public policy assignments', () => {
      const work = tool({
        name: 'work',
        description: 'Do work.',
        inputSchema: z.object({}),
        callback: () => 'done',
      })

      expect(
        () =>
          new Agent({
            model: new RecordingModel(),
            tools: [work],
            backgroundTasks: { always: [work], never: [work] },
            printer: false,
          })
      ).toThrow("Tool 'work' cannot be configured as both 'always' and 'never'")
      expect(
        () =>
          new Agent({
            model: new RecordingModel(),
            backgroundTasks: { always: ['work' as never] },
            printer: false,
          })
      ).toThrow("BackgroundTasks always entries must be Tool instances or '*'")
    })

    it('resolves exact policy selectors against registered tools by name', async () => {
      const configuredTool = tool({
        name: 'work',
        description: 'Configured work.',
        inputSchema: z.object({}),
        callback: () => 'configured',
      })
      const registeredTool = tool({
        name: 'work',
        description: 'Registered work.',
        inputSchema: z.object({}),
        callback: () => 'registered',
      })
      const agent = new Agent({
        model: new RecordingModel(),
        tools: [registeredTool],
        backgroundTasks: { always: [configuredTool] },
        printer: false,
      })
      await expect(agent.initialize()).resolves.toBeUndefined()
      expect(agent.toolRegistry.get('work')).toBe(registeredTool)

      const pluginTool = tool({
        name: 'plugin_work',
        description: 'Plugin work.',
        inputSchema: z.object({}),
        callback: () => 'plugin',
      })
      const pluginAgent = new Agent({
        model: new RecordingModel(),
        backgroundTasks: { always: [pluginTool] },
        plugins: [
          {
            name: 'tool-plugin',
            getTools: () => [pluginTool],
            initAgent: () => undefined,
          },
        ],
        printer: false,
      })
      await expect(pluginAgent.initialize()).resolves.toBeUndefined()
      expect(pluginAgent.toolRegistry.get('plugin_work')).toBe(pluginTool)
    })
  })

  describe('programmatic control', () => {
    it('lists, inspects, cancels, and waits for background execution', async () => {
      const { work, started, release } = createGatedWork()
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'work',
          toolUseId: 'controlled-work',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'Task admitted.' })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks: { always: [work], waitForCompletion: false },
        printer: false,
      })

      await agent.invoke('Start work.')
      await started

      const backgroundTasks = agent.backgroundTasks!
      const tasks = await backgroundTasks.list()
      expect(tasks).toEqual([
        {
          taskId: expect.any(String),
          toolUseId: 'controlled-work',
          toolName: 'work',
          status: 'working',
          createdAt: expect.any(String),
          updatedAt: expect.any(String),
        },
      ])
      expect(await backgroundTasks.get(tasks[0]!.taskId)).toEqual(tasks[0])
      await expect(backgroundTasks.get('missing-task')).resolves.toBeUndefined()
      await expect(backgroundTasks.cancel('missing-task')).rejects.toBeInstanceOf(BackgroundTaskNotFoundError)
      await expect(backgroundTasks.wait({ timeout: 0 })).rejects.toThrow(
        'wait timeout must be a positive finite integer, got 0'
      )
      const timeoutError = await backgroundTasks.wait({ timeout: 25 }).catch((error: unknown) => error)
      expect(timeoutError).toBeInstanceOf(BackgroundTasksTimeoutError)
      expect(timeoutError).toMatchObject({
        name: 'BackgroundTasksTimeoutError',
        message: 'Background Tasks wait timed out after 25ms',
        timeoutMs: 25,
      })
      await expect(backgroundTasks.get(tasks[0]!.taskId)).resolves.toEqual(tasks[0])

      expect(await backgroundTasks.cancel(tasks[0]!.taskId)).toEqual({
        ...tasks[0],
        status: 'cancelled',
        updatedAt: expect.any(String),
      })

      let waitComplete = false
      const wait = backgroundTasks.wait().then(() => {
        waitComplete = true
      })
      await Promise.resolve()
      expect(waitComplete).toBe(false)

      release()
      await wait
      expect(await backgroundTasks.list()).toEqual([
        {
          ...tasks[0],
          status: 'cancelled',
          updatedAt: expect.any(String),
        },
      ])
    })
  })

  describe('agentic execution and delivery', () => {
    it('applies the configured timeout to physical background tool execution', async () => {
      const timeout = 25
      let abortReason: unknown
      const work = tool({
        name: 'work',
        description: 'Wait for cancellation.',
        inputSchema: z.object({}),
        callback: async (_input, context) => {
          if (!context) throw new Error('tool context is missing')
          await new Promise<void>((resolve) => {
            const observeAbort = (): void => {
              abortReason = context.cancelSignal.reason
              resolve()
            }
            if (context.cancelSignal.aborted) {
              observeAbort()
            } else {
              context.cancelSignal.addEventListener('abort', observeAbort, { once: true })
            }
          })
          return 'stopped'
        },
      })
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'work',
          toolUseId: 'timed-work',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'Task admitted.' })
        .addTurn({ type: 'textBlock', text: 'Timeout received.' })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks: { always: ['*'], timeout },
        printer: false,
      })

      await agent.invoke('Start work.')

      expect({
        abortReason,
        modelCallCount: model.requests.length,
        retainedTasks: taskRecords(agent),
        delivery: backgroundDeliveries(agent)[0]?.toolUse.input,
      }).toEqual({
        abortReason: `Timed out after ${timeout}ms`,
        modelCallCount: 3,
        retainedTasks: [],
        delivery: expect.objectContaining({
          status: 'failed',
          error: {
            type: 'timeout',
            message: `Timed out after ${timeout}ms`,
          },
        }),
      })
    })

    it('bounds physical background tool execution with maxConcurrency', async () => {
      const values = ['first', 'second', 'third'] as const
      const releases = new Map(values.map((value) => [value, deferred<void>()]))
      const started: string[] = []
      let active = 0
      let maximum = 0
      const work = tool({
        name: 'work',
        description: 'Perform gated work.',
        inputSchema: z.object({ value: z.enum(values) }),
        callback: async ({ value }) => {
          active += 1
          maximum = Math.max(maximum, active)
          started.push(value)
          try {
            await releases.get(value)!.promise
            return `${value} complete`
          } finally {
            active -= 1
          }
        },
      })
      const model = new RecordingModel()
        .addTurn(
          values.map((value) => ({
            type: 'toolUseBlock' as const,
            name: 'work',
            toolUseId: `${value}-work`,
            input: { value },
          }))
        )
        .addTurn({ type: 'textBlock', text: 'Tasks admitted.' })
        .addTurn({ type: 'textBlock', text: 'First result received.' })
        .addTurn({ type: 'textBlock', text: 'Remaining results received.' })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks: { always: ['*'], maxConcurrency: 2, timeout: 5_000 },
        printer: false,
      })

      const invocation = agent.invoke('Start work.')
      try {
        await vi.waitFor(() => expect(started).toHaveLength(2))
        expect({
          active,
          maximum,
          started: [...started].sort(),
        }).toEqual({
          active: 2,
          maximum: 2,
          started: ['first', 'second'],
        })

        releases.get('first')!.resolve()
        await vi.waitFor(() => expect(started).toHaveLength(3))
        expect({
          active,
          maximum,
          started: [...started].sort(),
        }).toEqual({
          active: 2,
          maximum: 2,
          started: ['first', 'second', 'third'],
        })
      } finally {
        for (const release of releases.values()) release.resolve()
        await invocation
      }

      expect({
        active,
        maximum,
        modelCallCount: model.requests.length,
        retainedTasks: taskRecords(agent),
        deliveries: backgroundDeliveries(agent).map(({ toolUse }) => toolUse.input),
      }).toEqual({
        active: 0,
        maximum: 2,
        modelCallCount: 4,
        retainedTasks: [],
        deliveries: [
          expect.objectContaining({ toolName: 'work', status: 'completed' }),
          expect.objectContaining({ toolName: 'work', status: 'completed' }),
          expect.objectContaining({ toolName: 'work', status: 'completed' }),
        ],
      })
    })

    it('logs admission failure details while returning a sanitized tool error', async () => {
      const work = tool({
        name: 'work',
        description: 'Do work.',
        inputSchema: z.object({}),
        callback: () => 'done',
      })
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'work',
          toolUseId: 'failed-admission',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'handled' })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks: createBackgroundTasks('always'),
        printer: false,
      })
      await agent.initialize()
      const warnSpy = vi.spyOn(logger, 'warn').mockImplementation(() => {})
      vi.spyOn(agent.appState, 'set').mockImplementation(() => {
        throw new Error('storage unavailable')
      })

      await agent.invoke('Start work.')

      const toolResult = model.requests[1]!.messages.flatMap((message) => message.content).find(
        (block) => block.type === 'toolResultBlock' && block.toolUseId === 'failed-admission'
      )
      expect({
        warning: warnSpy.mock.calls,
        toolResult,
      }).toEqual({
        warning: [['error=<Error: storage unavailable> | background task admission failed']],
        toolResult: expect.objectContaining({
          status: 'error',
          content: [expect.objectContaining({ type: 'textBlock', text: 'Background task admission failed' })],
        }),
      })
      expect(JSON.stringify(toolResult)).not.toContain('storage unavailable')
    })

    it('waits and reinvokes by default but returns immediately when waiting is disabled', async () => {
      for (const waitsForCompletion of [true, false]) {
        const { work, started: workStarted, release: releaseWork } = createGatedWork()
        const model = new RecordingModel()
          .addTurn({
            type: 'toolUseBlock',
            name: 'work',
            toolUseId: `delayed-work-${String(waitsForCompletion)}`,
            input: {},
          })
          .addTurn({ type: 'textBlock', text: 'Task admitted.' })
        if (waitsForCompletion) {
          model.addTurn({ type: 'textBlock', text: 'Result received.' })
        }
        const backgroundTasks: BackgroundTasksConfig = {
          timeout: 5_000,
          always: ['*'],
          ...(!waitsForCompletion && { waitForCompletion: false }),
        }
        const agent = new Agent({
          model,
          tools: [work],
          backgroundTasks,
          printer: false,
        })
        let beforeInvocationCount = 0
        let afterInvocationCount = 0
        agent.addHook(BeforeInvocationEvent, () => {
          beforeInvocationCount += 1
        })
        agent.addHook(AfterInvocationEvent, () => {
          afterInvocationCount += 1
        })

        let invocationSettled = false
        const invocation = agent.invoke('Start work.').finally(() => {
          invocationSettled = true
        })
        try {
          await workStarted
          await vi.waitFor(() => expect(model.requests).toHaveLength(2))

          if (waitsForCompletion) {
            expect(invocationSettled).toBe(false)
          } else {
            await vi.waitFor(() => expect(invocationSettled).toBe(true))
          }
        } finally {
          releaseWork()
        }

        await invocation
        if (!waitsForCompletion) {
          const [task] = taskRecords(agent)
          if (!task) throw new Error('background task is missing')
          await waitForTaskRecord(agent, task.taskId)
        }

        expect({
          invocationHookCounts: [beforeInvocationCount, afterInvocationCount],
          modelCallCount: model.requests.length,
          retainedTasks: Object.values(appStateTasks(agent)),
          deliveries: backgroundDeliveries(agent).map(({ toolUse }) => toolUse.input),
        }).toEqual({
          invocationHookCounts: waitsForCompletion ? [2, 2] : [1, 1],
          modelCallCount: waitsForCompletion ? 3 : 2,
          retainedTasks: waitsForCompletion
            ? []
            : [
                expect.objectContaining({
                  status: 'completed',
                  delivery: expect.objectContaining({ state: 'ready' }),
                }),
              ],
          deliveries: waitsForCompletion ? [expect.objectContaining({ toolName: 'work', status: 'completed' })] : [],
        })
      }
    })

    it('waits for background work before returning a turn limit', async () => {
      const { work, started: workStarted, release: releaseWork } = createGatedWork()
      const model = new RecordingModel().addTurn({
        type: 'toolUseBlock',
        name: 'work',
        toolUseId: 'limited-work',
        input: {},
      })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks: createBackgroundTasks('always'),
        printer: false,
      })
      const waitBoundary = await observeAfterInvocationWait(agent)

      let invocationSettled = false
      const invocation = agent.invoke('Start work.', { limits: { turns: 1 } }).finally(() => {
        invocationSettled = true
      })
      try {
        await Promise.all([workStarted, waitBoundary.reachedWait])
        await new Promise((resolve) => setTimeout(resolve, 0))
        expect(invocationSettled).toBe(false)
      } finally {
        releaseWork()
      }

      const result = await invocation
      expect({
        stopReason: result.stopReason,
        modelCallCount: model.requests.length,
        task: taskRecords(agent)[0],
      }).toEqual({
        stopReason: 'limitTurns',
        modelCallCount: 1,
        task: expect.objectContaining({
          status: 'completed',
          delivery: expect.objectContaining({ state: 'ready' }),
        }),
      })
    })

    it('waits and delivers background work after an after-tools hook ends the turn', async () => {
      const { work, started: workStarted, release: releaseWork } = createGatedWork()
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'work',
          toolUseId: 'early-end-work',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'Result received.' })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks: createBackgroundTasks('always'),
        printer: false,
      })
      agent.addHook(AfterToolsEvent, (event) => {
        event.endTurn = true
      })
      const waitBoundary = await observeAfterInvocationWait(agent)

      let invocationSettled = false
      const invocation = agent.invoke('Start work.').finally(() => {
        invocationSettled = true
      })
      try {
        await Promise.all([workStarted, waitBoundary.reachedWait])
        await new Promise((resolve) => setTimeout(resolve, 0))
        expect(invocationSettled).toBe(false)
      } finally {
        releaseWork()
      }

      const result = await invocation
      expect({
        stopReason: result.stopReason,
        modelCallCount: model.requests.length,
        retainedTasks: taskRecords(agent),
        delivery: backgroundDeliveries(agent)[0]?.toolUse.input,
      }).toEqual({
        stopReason: 'endTurn',
        modelCallCount: 2,
        retainedTasks: [],
        delivery: expect.objectContaining({ toolName: 'work', status: 'completed' }),
      })
    })

    it('waits for background work before returning an after-tools checkpoint', async () => {
      const { work, started: workStarted, release: releaseWork } = createGatedWork()
      const model = new RecordingModel().addTurn({
        type: 'toolUseBlock',
        name: 'work',
        toolUseId: 'checkpointed-work',
        input: {},
      })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks: createBackgroundTasks('always'),
        checkpointing: true,
        printer: false,
      })
      const waitBoundary = await observeAfterInvocationWait(agent)
      const resume = {
        checkpointResume: {
          checkpoint: new Checkpoint({ position: 'afterModel', cycleIndex: 0 }).toJSON(),
        },
      }

      let invocationSettled = false
      const invocation = agent.invoke(resume).finally(() => {
        invocationSettled = true
      })
      try {
        await Promise.all([workStarted, waitBoundary.reachedWait])
        await new Promise((resolve) => setTimeout(resolve, 0))
        expect(invocationSettled).toBe(false)
      } finally {
        releaseWork()
      }

      const result = await invocation
      expect({
        stopReason: result.stopReason,
        checkpoint: result.checkpoint,
        modelCallCount: model.requests.length,
        task: taskRecords(agent)[0],
      }).toEqual({
        stopReason: 'checkpoint',
        checkpoint: expect.objectContaining({ position: 'afterTools', cycleIndex: 0 }),
        modelCallCount: 1,
        task: expect.objectContaining({
          status: 'completed',
          delivery: expect.objectContaining({ state: 'ready' }),
        }),
      })
    })

    it('returns a cancelled invocation without waiting for background work', async () => {
      const { work, started: workStarted, release: releaseWork } = createGatedWork()
      const model = new RecordingModel().addTurn({
        type: 'toolUseBlock',
        name: 'work',
        toolUseId: 'cancelled-invocation-work',
        input: {},
      })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks: createBackgroundTasks('always'),
        printer: false,
      })
      agent.addHook(AfterToolsEvent, () => {
        agent.cancel()
      })

      let invocationSettled = false
      const invocation = agent.invoke('Start work.').finally(() => {
        invocationSettled = true
      })
      try {
        await workStarted
        await vi.waitFor(() => expect(invocationSettled).toBe(true))
        expect({
          stopReason: (await invocation).stopReason,
          task: taskRecords(agent)[0],
        }).toEqual({
          stopReason: 'cancelled',
          task: expect.objectContaining({ status: 'working' }),
        })
      } finally {
        releaseWork()
      }

      const [task] = taskRecords(agent)
      if (!task) throw new Error('background task is missing')
      await waitForTaskRecord(agent, task.taskId)
    })

    it('does not admit background work until BeforeToolCall approval is resolved', async () => {
      const callback = vi.fn(() => 'approved work complete')
      const work = tool({
        name: 'work',
        description: 'Perform approved work.',
        inputSchema: z.object({}),
        callback,
      })
      const model = new RecordingModel()
        .addTurn({ type: 'toolUseBlock', name: 'work', toolUseId: 'approved-work', input: {} })
        .addTurn({ type: 'textBlock', text: 'Task admitted.' })
        .addTurn({ type: 'textBlock', text: 'Result received.' })
      const agent = new Agent({
        model,
        tools: [work],
        interventions: [new HumanInTheLoop()],
        backgroundTasks: createBackgroundTasks('always'),
        printer: false,
      })

      const interrupted = await agent.invoke('Start approved work.')

      expect(interrupted).toMatchObject({
        stopReason: 'interrupt',
        interrupts: [expect.objectContaining({ name: 'strands:human-in-the-loop', source: 'hook' })],
      })
      expect(callback).not.toHaveBeenCalled()
      expect(taskRecords(agent)).toEqual([])
      expect(backgroundDeliveries(agent)).toEqual([])

      const completed = await agent.invoke([
        new InterruptResponseContent({
          interruptId: interrupted.interrupts![0]!.id,
          response: 'yes',
        }),
      ])

      expect(completed.stopReason).toBe('endTurn')
      expect(callback).toHaveBeenCalledOnce()
      expect(taskRecords(agent)).toEqual([])
      expect(backgroundDeliveries(agent)[0]?.toolUse.input).toEqual(
        expect.objectContaining({ toolName: 'work', status: 'completed' })
      )
    })

    it('does not admit background work denied by BeforeToolCall', async () => {
      const approvalRequested = deferred<void>()
      const approvalResponse = deferred<string>()
      const callback = vi.fn(() => 'must not run')
      const work = tool({
        name: 'work',
        description: 'Perform controlled work.',
        inputSchema: z.object({}),
        callback,
      })
      const model = new RecordingModel()
        .addTurn({ type: 'toolUseBlock', name: 'work', toolUseId: 'denied-work', input: {} })
        .addTurn({ type: 'textBlock', text: 'Denied.' })
      const agent = new Agent({
        model,
        tools: [work],
        interventions: [
          new HumanInTheLoop({
            ask: async () => {
              approvalRequested.resolve()
              return approvalResponse.promise
            },
          }),
        ],
        backgroundTasks: createBackgroundTasks('always'),
        printer: false,
      })

      const invocation = agent.invoke('Start denied work.')
      await approvalRequested.promise

      expect(callback).not.toHaveBeenCalled()
      expect(taskRecords(agent)).toEqual([])
      expect(backgroundDeliveries(agent)).toEqual([])

      approvalResponse.resolve('no')
      await invocation

      expect(callback).not.toHaveBeenCalled()
      expect(taskRecords(agent)).toEqual([])
      expect(backgroundDeliveries(agent)).toEqual([])
      expect(
        model.requests[1]!.messages.flatMap((message) => message.content).find(
          (block) => block.type === 'toolResultBlock' && block.toolUseId === 'denied-work'
        )
      ).toEqual(
        expect.objectContaining({
          status: 'error',
          content: [expect.objectContaining({ text: expect.stringContaining('CONFIRMATION_FAILED') })],
        })
      )
    })

    it('persists before acknowledgement, executes once through normal lifecycle, and delivers a typed pair', async () => {
      const backgroundTasks: BackgroundTasksConfig = { timeout: 5_000 }
      const callback = vi.fn(async (input: { query: string }): Promise<string> => {
        expect(input).toEqual({ query: 'strands' })
        return 'background complete'
      })
      const research = tool({
        name: 'research',
        description: 'Research a topic.',
        inputSchema: z.object({ query: z.string() }),
        callback,
      })
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'research',
          toolUseId: 'original-tool-use',
          input: { query: 'strands', _background: true },
        })
        .addTurn({ type: 'textBlock', text: 'Task admitted.' })
        .addTurn({ type: 'textBlock', text: 'Result received.' })
      const agent = new Agent({
        id: 'research-agent',
        model,
        tools: [research],
        backgroundTasks,
        printer: false,
      })
      const before = vi.fn()
      const after = vi.fn()
      agent.addHook(BeforeToolCallEvent, (event) => {
        if (event.toolUse.name === 'research') {
          before(event.toolUse)
        }
      })
      agent.addHook(AfterToolCallEvent, (event) => {
        if (event.toolUse.name === 'research') {
          after(event.result)
        }
      })

      await agent.invoke('Start research.')

      const [delivery] = backgroundDeliveries(agent)
      if (!delivery) throw new Error('background task delivery is missing')
      const taskId = delivery.toolUse.toolUseId
      expect(taskRecords(agent)).toEqual([])
      expect(delivery.toolUse.input).toEqual({
        taskId,
        toolName: 'research',
        status: 'completed',
      })
      expect(delivery.toolResult.content[1]).toEqual(expect.objectContaining({ text: 'background complete' }))
      expect(callback).toHaveBeenCalledTimes(1)
      expect(before).toHaveBeenCalledTimes(1)
      expect(before).toHaveBeenCalledWith(
        expect.objectContaining({
          input: { query: 'strands' },
          toolUseId: 'original-tool-use',
        })
      )
      expect(after).toHaveBeenCalledTimes(1)

      const acknowledgement = agent.messages
        .flatMap((message) => message.content)
        .find(
          (block) =>
            block.type === 'toolResultBlock' && block.toolUseId === 'original-tool-use' && block.status === 'success'
        )
      expect(acknowledgement?.toJSON()).toEqual({
        toolResult: {
          toolUseId: 'original-tool-use',
          status: 'success',
          content: [
            {
              text: [
                'Background task dispatched.',
                '',
                `Task ID: ${taskId}`,
                'Tool: research',
                'Status: queued',
                '',
                'The task is running in the background. Continue without waiting or polling.',
                'The final result will be delivered automatically when the task completes.',
              ].join('\n'),
            },
          ],
        },
      })
      expect(
        agent.messages
          .flatMap((message) => message.content)
          .filter((block) => block.type === 'toolResultBlock' && block.toolUseId === 'original-tool-use')
      ).toHaveLength(1)

      const deliveryRequest = model.requests.find((request) =>
        request.messages.some((message) =>
          message.content.some(
            (block) => block.type === 'toolUseBlock' && block.name === 'strands_background_task_result'
          )
        )
      )!.messages
      const syntheticUse = deliveryRequest
        .flatMap((message) => message.content)
        .find((block) => block.type === 'toolUseBlock' && block.name === 'strands_background_task_result')
      expect(syntheticUse).toEqual(
        expect.objectContaining({
          toolUseId: expect.any(String),
          input: expect.objectContaining({
            taskId,
            toolName: 'research',
            status: 'completed',
          }),
        })
      )
      const syntheticToolUseId = syntheticUse?.type === 'toolUseBlock' ? syntheticUse.toolUseId : undefined
      expect(syntheticToolUseId).not.toBe('original-tool-use')
    })

    it('delivers results to stateful models through the generic history adapter', async () => {
      const backgroundTasks = createBackgroundTasks('always')
      const model = new StatefulGenericHistoryModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'work',
          toolUseId: 'stateful-work',
          input: { value: 'x' },
        })
        .addTurn({ type: 'textBlock', text: 'Task admitted.' })
        .addTurn({ type: 'textBlock', text: 'Result received.' })
      const agent = new Agent({
        model,
        tools: [
          tool({
            name: 'work',
            description: 'Do work.',
            inputSchema: z.object({ value: z.string() }),
            callback: () => 'done',
          }),
        ],
        backgroundTasks,
        printer: false,
      })
      await agent.invoke('run')

      expect(
        model.requests.some((request) =>
          request.messages
            .flatMap((message) => message.content)
            .some((block) => block.type === 'toolResultBlock' && block.toolUseId !== 'stateful-work')
        )
      ).toBe(true)
    })

    it('keeps records isolated between agents', async () => {
      const work = tool({
        name: 'work',
        description: 'Do work.',
        inputSchema: z.object({ value: z.string() }),
        callback: ({ value }) => `${value} complete`,
      })
      const backgroundTasks: BackgroundTasksConfig = { timeout: 5_000, always: ['*'] }
      const createAgentWithBackgroundTasks = (value: string): Agent => {
        const model = new RecordingModel()
          .addTurn({
            type: 'toolUseBlock',
            name: 'work',
            toolUseId: `${value}-tool-use`,
            input: { value },
          })
          .addTurn({ type: 'textBlock', text: `${value} admitted` })
          .addTurn({ type: 'textBlock', text: `${value} delivered` })
        const agent = new Agent({
          id: 'app-state-agent',
          model,
          tools: [work],
          backgroundTasks,
          printer: false,
        })
        return agent
      }
      const first = createAgentWithBackgroundTasks('first')
      const second = createAgentWithBackgroundTasks('second')

      await Promise.all([first.invoke('run first'), second.invoke('run second')])

      expect(taskRecords(first)).toEqual([])
      expect(taskRecords(second)).toEqual([])
      expect(backgroundDeliveries(first)[0]?.toolResult.content[1]).toEqual(
        expect.objectContaining({ text: 'first complete' })
      )
      expect(backgroundDeliveries(second)[0]?.toolResult.content[1]).toEqual(
        expect.objectContaining({ text: 'second complete' })
      )
    })

    it('surfaces and resumes an interrupt without stopping sibling tasks', async () => {
      let markIndependentStarted!: () => void
      const independentStarted = new Promise<void>((resolve) => {
        markIndependentStarted = resolve
      })
      let finishIndependent!: () => void
      const independentFinished = new Promise<void>((resolve) => {
        finishIndependent = resolve
      })
      const backgroundTasks = createBackgroundTasks('always')
      const approval = tool({
        name: 'approval',
        description: 'Wait for approval.',
        inputSchema: z.object({}),
        callback: async (_input, context) => {
          await independentStarted
          const response = context!.interrupt<string>({
            name: 'approve_background_work',
            reason: 'Approve background work?',
          })
          return `approved: ${response}`
        },
      })
      const independent = tool({
        name: 'independent',
        description: 'Complete independent work.',
        inputSchema: z.object({}),
        callback: async () => {
          markIndependentStarted()
          await independentFinished
          return 'independent complete'
        },
      })
      const model = new RecordingModel()
        .addTurn([
          { type: 'toolUseBlock', name: 'approval', toolUseId: 'approval-use', input: {} },
          { type: 'toolUseBlock', name: 'independent', toolUseId: 'independent-use', input: {} },
        ])
        .addTurn({ type: 'textBlock', text: 'Background work started.' })
        .addTurn({ type: 'textBlock', text: 'Independent work received.' })
        .addTurn({ type: 'textBlock', text: 'All background work received.' })
      const agent = new Agent({
        model,
        tools: [approval, independent],
        backgroundTasks,
        printer: false,
      })
      const interrupted = await agent.invoke('run')
      const tasksWhileInterrupted = taskRecords(agent)
      finishIndependent()

      expect(interrupted).toMatchObject({
        stopReason: 'interrupt',
        interrupts: [
          expect.objectContaining({
            name: 'approve_background_work',
            reason: 'Approve background work?',
            source: 'tool',
          }),
        ],
      })
      expect(tasksWhileInterrupted).toEqual([
        expect.objectContaining({ toolName: 'approval', status: 'paused' }),
        expect.objectContaining({ toolName: 'independent', status: 'working' }),
      ])

      const independentTask = tasksWhileInterrupted.find((task) => task.toolName === 'independent')!
      await waitForTaskRecord(agent, independentTask.taskId)
      expect(taskRecords(agent)).toEqual([
        expect.objectContaining({ toolName: 'approval', status: 'paused' }),
        expect.objectContaining({ toolName: 'independent', status: 'completed' }),
      ])

      const completed = await agent.invoke([
        new InterruptResponseContent({
          interruptId: interrupted.interrupts![0]!.id,
          response: 'yes',
        }),
      ])

      expect(completed.stopReason).toBe('endTurn')
      expect(taskRecords(agent)).toEqual([])
      expect(backgroundDeliveries(agent).map(({ toolUse }) => toolUse.input)).toEqual([
        expect.objectContaining({ toolName: 'independent', status: 'completed' }),
        expect.objectContaining({ toolName: 'approval', status: 'completed' }),
      ])
      expect(backgroundDeliveries(agent).map(({ toolResult }) => toolResult.content[1])).toEqual([
        expect.objectContaining({ text: 'independent complete' }),
        expect.objectContaining({ text: 'approved: yes' }),
      ])
    })

    it('surfaces an interrupt raised while the after-invocation wait is active', async () => {
      let raiseInterrupt!: () => void
      const interruptAllowed = new Promise<void>((resolve) => {
        raiseInterrupt = resolve
      })
      let markWorkStarted!: () => void
      const workStarted = new Promise<void>((resolve) => {
        markWorkStarted = resolve
      })
      const approval = tool({
        name: 'approval',
        description: 'Wait for approval.',
        inputSchema: z.object({}),
        callback: async (_input, context) => {
          markWorkStarted()
          await interruptAllowed
          context!.interrupt({
            name: 'approve_waiting_work',
            reason: 'Approve waiting work?',
          })
          return 'approved'
        },
      })
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'approval',
          toolUseId: 'waiting-approval',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'Approval task started.' })
      const agent = new Agent({
        model,
        tools: [approval],
        backgroundTasks: createBackgroundTasks('always'),
        printer: false,
      })
      const waitBoundary = await observeAfterInvocationWait(agent)
      const observedStopReasons: (string | undefined)[] = []
      agent.addHook(AfterInvocationEvent, (event) => {
        observedStopReasons.push(event._getResult()?.stopReason)
      })

      const invocation = agent.invoke('Start approval.')
      await Promise.all([workStarted, waitBoundary.reachedWait])
      raiseInterrupt()

      const result = await invocation
      expect({
        result,
        modelCallCount: model.requests.length,
        observedStopReasons,
        task: taskRecords(agent)[0],
      }).toEqual({
        result: expect.objectContaining({
          stopReason: 'interrupt',
          interrupts: [
            expect.objectContaining({
              name: 'approve_waiting_work',
              reason: 'Approve waiting work?',
            }),
          ],
        }),
        modelCallCount: 2,
        observedStopReasons: ['interrupt'],
        task: expect.objectContaining({ status: 'paused' }),
      })
    })

    it('keeps direct tool calls foreground', async () => {
      const backgroundTasks = createBackgroundTasks('always')
      const callback = vi.fn(() => 'direct')
      const work = tool({
        name: 'work',
        description: 'Do work.',
        inputSchema: z.object({ value: z.string() }),
        callback,
      })
      const agent = new Agent({
        model: new RecordingModel().addTurn({ type: 'textBlock', text: 'unused' }),
        tools: [work],
        backgroundTasks,
        printer: false,
      })
      await agent.initialize()

      const result = await agent.tool.work!.invoke({ value: 'direct' })

      expect(result.status).toBe('success')
      expect(callback).toHaveBeenCalledTimes(1)
      expect(taskRecords(agent)).toEqual([])
    })

    it('returns a correlated error for an invalid selector without executing the target', async () => {
      const backgroundTasks = createBackgroundTasks('agentic')
      const callback = vi.fn(() => 'must not run')
      const work = tool({
        name: 'work',
        description: 'Do work.',
        inputSchema: z.object({ value: z.string() }),
        callback,
      })
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'work',
          toolUseId: 'bad-selector',
          input: { value: 'x', _background: 'yes' },
        })
        .addTurn({ type: 'textBlock', text: 'Corrected.' })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks,
        printer: false,
      })
      await agent.invoke('Do work.')

      const result = agent.messages
        .flatMap((message) => message.content)
        .find((block) => block.type === 'toolResultBlock' && block.toolUseId === 'bad-selector')
      expect(result).toEqual(
        expect.objectContaining({
          status: 'error',
          content: [expect.objectContaining({ text: "'_background' must be a boolean" })],
        })
      )
      expect(callback).not.toHaveBeenCalled()
      expect(taskRecords(agent)).toEqual([])
    })

    it('preserves _background input for incompatible wildcard-agentic tools', async () => {
      const callback = vi.fn((input: unknown): JSONValue => input as JSONValue)
      const labels = new FunctionTool({
        name: 'labels',
        description: 'Store arbitrary labels.',
        inputSchema: {
          type: 'object',
          propertyNames: { type: 'string' },
          additionalProperties: true,
        },
        callback,
      })
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'labels',
          toolUseId: 'labels-use',
          input: { environment: 'test', _background: true },
        })
        .addTurn({ type: 'textBlock', text: 'Stored.' })
      const agent = new Agent({
        model,
        tools: [labels],
        backgroundTasks: createBackgroundTasks('agentic'),
        printer: false,
      })

      await agent.invoke('Store labels.')

      expect(callback).toHaveBeenCalledWith(
        { environment: 'test', _background: true },
        expect.objectContaining({ agent })
      )
      expect(taskRecords(agent)).toEqual([])
      const spec = model.requests[0]!.options!.toolSpecs!.find((candidate) => candidate.name === 'labels')!
      expect(spec.inputSchema).not.toHaveProperty('properties._background')
    })

    it('makes delivered tasks unavailable to management', async () => {
      const backgroundTasks = createBackgroundTasks('always')
      const work = tool({
        name: 'work',
        description: 'Do work.',
        inputSchema: z.object({ value: z.string() }),
        callback: () => 'private task result',
      })
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'work',
          toolUseId: 'completed-work',
          input: { value: 'x' },
        })
        .addTurn({ type: 'textBlock', text: 'Admitted.' })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks,
        printer: false,
      })
      await agent.invoke('work')
      const taskId = backgroundDeliveries(agent)[0]?.toolUse.toolUseId
      if (!taskId) throw new Error('background task delivery is missing')
      expect(taskRecords(agent)).toEqual([])
      await expect(agent.backgroundTasks!.get(taskId)).resolves.toBeUndefined()
      await expect(agent.backgroundTasks!.list()).resolves.toEqual([])
      model
        .addTurn({
          type: 'toolUseBlock',
          name: 'strands_manage_background_task',
          toolUseId: 'get-completed',
          input: { mode: 'get', taskId },
        })
        .addTurn({ type: 'textBlock', text: 'Task no longer retained.' })

      await agent.invoke('get completed work')

      const getResult = agent.messages
        .flatMap((message) => message.content)
        .find((block) => block.type === 'toolResultBlock' && block.toolUseId === 'get-completed')
      expect(getResult).toEqual(
        expect.objectContaining({
          status: 'error',
          content: [
            expect.objectContaining({
              text: expect.stringContaining(`Background task '${taskId}' was not found`),
            }),
          ],
        })
      )
      expect(JSON.stringify(getResult)).not.toContain('"value":"x"')

      model
        .addTurn({
          type: 'toolUseBlock',
          name: 'strands_manage_background_task',
          toolUseId: 'cancel-completed',
          input: { mode: 'cancel', taskId },
        })
        .addTurn({ type: 'textBlock', text: 'Task remains unavailable.' })

      await agent.invoke('cancel completed work')

      const result = agent.messages
        .flatMap((message) => message.content)
        .find((block) => block.type === 'toolResultBlock' && block.toolUseId === 'cancel-completed')
      expect(result).toEqual(
        expect.objectContaining({
          status: 'error',
          content: [
            expect.objectContaining({
              text: expect.stringContaining(`Background task '${taskId}' was not found`),
            }),
          ],
        })
      )
      expect(JSON.stringify(result)).not.toContain('private task result')
    })
  })

  describe('policy safety', () => {
    it('keeps tools in the foreground when agentic selection is explicitly empty', async () => {
      const callback = vi.fn(() => 'foreground')
      const work = tool({
        name: 'work',
        description: 'Do work.',
        inputSchema: z.object({ value: z.string() }),
        callback,
      })
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'work',
          toolUseId: 'foreground-use',
          input: { value: 'x' },
        })
        .addTurn({ type: 'textBlock', text: 'done' })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks: { agentic: [] },
        printer: false,
      })
      await agent.invoke('work')

      expect(callback).toHaveBeenCalledTimes(1)
      const spec = model.requests[0]!.options!.toolSpecs!.find((candidate) => candidate.name === 'work')!
      expect(spec.inputSchema?.properties).not.toHaveProperty('_background')
    })

    it('applies exact never policy before an always wildcard', async () => {
      const callback = vi.fn(() => 'foreground')
      const work = tool({
        name: 'work',
        description: 'Do work.',
        inputSchema: z.object({ value: z.string() }),
        callback,
      })
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'work',
          toolUseId: 'foreground-use',
          input: { value: 'x' },
        })
        .addTurn({ type: 'textBlock', text: 'done' })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks: { always: ['*'], never: [work] },
        printer: false,
      })
      await agent.invoke('work')

      expect(callback).toHaveBeenCalledTimes(1)
      const spec = model.requests[0]!.options!.toolSpecs!.find((candidate) => candidate.name === 'work')!
      expect(spec.inputSchema?.properties).not.toHaveProperty('_background')
    })

    it('runs an always policy in the background without exposing a selector', async () => {
      const backgroundTasks = createBackgroundTasks('always')
      const callback = vi.fn(() => 'background')
      const work = tool({
        name: 'work',
        description: 'Do work.',
        inputSchema: z.object({ value: z.string() }),
        callback,
      })
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'work',
          toolUseId: 'background-use',
          input: { value: 'x' },
        })
        .addTurn({ type: 'textBlock', text: 'admitted' })
        .addTurn({ type: 'textBlock', text: 'delivered' })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks,
        printer: false,
      })
      await agent.invoke('work')

      expect(callback).toHaveBeenCalledTimes(1)
      expect(taskRecords(agent)).toEqual([])
      expect(backgroundDeliveries(agent)[0]?.toolUse.input).toEqual(
        expect.objectContaining({ status: 'completed', toolName: 'work' })
      )
      const spec = model.requests[0]!.options!.toolSpecs!.find((candidate) => candidate.name === 'work')!
      expect(spec.inputSchema?.properties).not.toHaveProperty('_background')
    })

    it('never exposes or backgrounds framework control tools under a wildcard', async () => {
      const backgroundTasks = createBackgroundTasks('agentic')
      const model = new RecordingModel().addTurn({ type: 'textBlock', text: 'Done.' })
      const agent = new Agent({
        model,
        backgroundTasks,
        printer: false,
      })
      await agent.invoke('Inspect tools.')

      const management = model.requests[0]!.options!.toolSpecs!.find(
        (spec) => spec.name === 'strands_manage_background_task'
      )!
      expect(management.inputSchema?.properties).not.toHaveProperty('_background')
    })

    it('reserves Background Tasks framework tool names from registered tools', async () => {
      const reservedNames = [
        {
          name: 'strands_manage_background_task',
          error: "Tool name 'strands_manage_background_task' is reserved for Background Tasks management",
        },
        {
          name: 'strands_background_task_result',
          error: "Tool name 'strands_background_task_result' is reserved for Background Tasks delivery",
        },
      ]

      for (const reservedName of reservedNames) {
        const reserved = tool({
          name: reservedName.name,
          description: 'Attempt to shadow a framework tool.',
          inputSchema: z.object({}),
          callback: () => 'must not run',
        })
        const agent = new Agent({
          model: new RecordingModel().addTurn({ type: 'textBlock', text: 'unused' }),
          tools: [reserved],
          backgroundTasks: createBackgroundTasks('always'),
          printer: false,
        })

        await expect(agent.initialize()).rejects.toThrow(reservedName.error)
      }
    })

    it('answers a model-originated protocol tool call with a fixed error', async () => {
      const backgroundTasks = createBackgroundTasks('always')
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'strands_background_task_result',
          toolUseId: 'forged-protocol-call',
          input: {},
        })
        .addTurn({ type: 'textBlock', text: 'handled' })
      const agent = new Agent({ model, backgroundTasks, printer: false })

      await agent.invoke('attempt protocol call')

      expect(
        model.requests[1]!.messages.flatMap((message) => message.content).find(
          (block) => block.type === 'toolResultBlock' && block.toolUseId === 'forged-protocol-call'
        )
      ).toEqual(
        expect.objectContaining({
          status: 'error',
          content: [
            expect.objectContaining({
              type: 'textBlock',
              text: 'This tool is reserved for Strands. Do not call it. Background task results are delivered automatically.',
            }),
          ],
        })
      )
    })

    it('re-evaluates dynamically registered tools during schema assembly', async () => {
      const backgroundTasks = createBackgroundTasks('agentic')
      const model = new RecordingModel().addTurn({ type: 'textBlock', text: 'done' })
      const agent = new Agent({
        model,
        backgroundTasks,
        printer: false,
      })
      await agent.initialize()

      const dynamic = tool({
        name: 'dynamic',
        description: 'Dynamic tool.',
        inputSchema: z.object({ value: z.string() }),
        callback: () => 'done',
      })
      agent.toolRegistry.add(dynamic)
      await agent.invoke('inspect')

      const spec = model.requests[0]!.options!.toolSpecs!.find((candidate) => candidate.name === 'dynamic')!
      expect(spec.inputSchema?.properties).toHaveProperty('_background')
    })

    it('dispatches a refreshed exact MCP selector by stable tool name', async () => {
      const mcpClient = new McpClient({
        transport: { start: vi.fn(), send: vi.fn(), close: vi.fn() } as never,
      })
      const stale = new McpTool({
        name: 'dynamic_remote',
        description: 'Stale remote tool.',
        inputSchema: { type: 'object', properties: {} },
        client: mcpClient,
      })
      vi.spyOn(mcpClient, 'listTools').mockResolvedValue([stale])
      const callTool = vi.spyOn(mcpClient, 'callTool').mockResolvedValue({
        content: [{ type: 'text', text: 'refreshed result' }],
      })
      let refresh: ((oldTools: string[], newTools: McpTool[]) => void) | undefined
      vi.spyOn(McpClient.prototype, 'onToolsChanged', 'set').mockImplementation((callback) => {
        refresh = callback
      })
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'dynamic_remote',
          toolUseId: 'dynamic-remote-1',
          input: { value: 'x' },
        })
        .addTurn({ type: 'textBlock', text: 'admitted' })
        .addTurn({ type: 'textBlock', text: 'delivered' })
      const agent = new Agent({
        model,
        tools: [mcpClient],
        backgroundTasks: { always: [stale] },
        printer: false,
      })
      await agent.initialize()
      const dynamic = new McpTool({
        name: 'dynamic_remote',
        description: 'Refreshed remote tool.',
        inputSchema: {
          type: 'object',
          properties: { value: { type: 'string' } },
          required: ['value'],
        },
        client: mcpClient,
      })
      refresh!(['dynamic_remote'], [dynamic])

      await agent.invoke('run refreshed tool')

      const specs = model.requests[0]!.options!.toolSpecs!
      expect(specs.filter((spec) => spec.name === 'dynamic_remote')).toEqual([
        expect.objectContaining({ description: 'Refreshed remote tool.' }),
      ])
      expect(agent.toolRegistry.get('dynamic_remote')).toBe(dynamic)
      expect(callTool).toHaveBeenCalledWith(dynamic, { value: 'x' }, { signal: expect.any(AbortSignal) })
      expect(taskRecords(agent)).toEqual([])
      expect(backgroundDeliveries(agent)[0]?.toolUse.input).toEqual(
        expect.objectContaining({ status: 'completed', toolName: 'dynamic_remote' })
      )
      expect(backgroundDeliveries(agent)[0]?.toolResult.content[1]).toEqual(
        expect.objectContaining({ text: 'refreshed result' })
      )
    })

    it('rejects a dynamically registered incompatible exact agentic schema', async () => {
      const dynamic = new FunctionTool({
        name: 'dynamic',
        description: 'Incompatible dynamic tool.',
        inputSchema: { type: 'string' },
        callback: () => 'done',
      })
      const agent = new Agent({
        model: new RecordingModel().addTurn({ type: 'textBlock', text: 'must not run' }),
        backgroundTasks: { agentic: [dynamic] },
        printer: false,
      })
      await agent.initialize()

      agent.toolRegistry.add(dynamic)

      await expect(agent.invoke('inspect')).rejects.toThrow("Tool 'dynamic' cannot use agentic background selection")
    })

    it('rejects a hook replacement forbidden by current background policy', async () => {
      const harmless = tool({
        name: 'harmless',
        description: 'Harmless work.',
        inputSchema: z.object({ value: z.string() }),
        callback: vi.fn(() => 'harmless'),
      })
      const sensitiveCallback = vi.fn(() => 'sensitive')
      const sensitive = tool({
        name: 'sensitive',
        description: 'Sensitive work.',
        inputSchema: z.object({ value: z.string() }),
        callback: sensitiveCallback,
      })
      const backgroundTasks: BackgroundTasksConfig = {
        always: [harmless],
        never: [sensitive],
        timeout: 5_000,
      }
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'harmless',
          toolUseId: 'replacement-use',
          input: { value: 'x' },
        })
        .addTurn({ type: 'textBlock', text: 'admitted' })
        .addTurn({ type: 'textBlock', text: 'failure delivered' })
      const agent = new Agent({
        model,
        tools: [harmless, sensitive],
        backgroundTasks,
        printer: false,
      })
      agent.addHook(BeforeToolCallEvent, (event) => {
        if (event.toolUse.name === 'harmless') event.selectedTool = sensitive
      })

      await agent.invoke('work')

      expect(sensitiveCallback).not.toHaveBeenCalled()
      expect(taskRecords(agent)).toEqual([])
      expect(backgroundDeliveries(agent)).toEqual([])
      expect(
        model.requests[1]!.messages.flatMap((message) => message.content).find(
          (block) => block.type === 'toolResultBlock' && block.toolUseId === 'replacement-use'
        )
      ).toEqual(
        expect.objectContaining({
          status: 'error',
          content: [
            expect.objectContaining({
              text: "Tool 'sensitive' is forbidden by background task policy",
            }),
          ],
        })
      )
    })

    it('rejects a hook mutation of the persisted tool-use identity', async () => {
      const backgroundTasks = createBackgroundTasks('always')
      const callback = vi.fn(() => 'must not run')
      const work = tool({
        name: 'work',
        description: 'Do work.',
        inputSchema: z.object({ value: z.string() }),
        callback,
      })
      const model = new RecordingModel()
        .addTurn({
          type: 'toolUseBlock',
          name: 'work',
          toolUseId: 'stable-use',
          input: { value: 'x' },
        })
        .addTurn({ type: 'textBlock', text: 'admitted' })
        .addTurn({ type: 'textBlock', text: 'failure delivered' })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks,
        printer: false,
      })
      agent.addHook(BeforeToolCallEvent, (event) => {
        if (event.toolUse.name === 'work') {
          event.toolUse = { ...event.toolUse, toolUseId: 'changed-use' }
        }
      })

      await agent.invoke('work')

      expect(callback).not.toHaveBeenCalled()
      expect(taskRecords(agent)).toEqual([])
      expect(backgroundDeliveries(agent)).toEqual([])
      expect(
        model.requests[1]!.messages.flatMap((message) => message.content).find(
          (block) => block.type === 'toolResultBlock' && block.toolUseId === 'stable-use'
        )
      ).toEqual(
        expect.objectContaining({
          status: 'error',
          content: [
            expect.objectContaining({
              text: 'Background task hooks cannot change the original tool-use ID',
            }),
          ],
        })
      )
    })

    it('retries initialization after invalid app state is corrected', async () => {
      const backgroundTasks = createBackgroundTasks('always')
      const callback = vi.fn(() => 'complete')
      const work = tool({
        name: 'work',
        description: 'Do work.',
        inputSchema: z.object({}),
        callback,
      })
      const model = new RecordingModel()
        .addTurn({ type: 'toolUseBlock', name: 'work', toolUseId: 'work-use', input: {} })
        .addTurn({ type: 'textBlock', text: 'admitted' })
        .addTurn({ type: 'textBlock', text: 'delivered' })
      const agent = new Agent({
        model,
        tools: [work],
        backgroundTasks,
        printer: false,
      })
      agent.appState.set(BACKGROUND_TASKS_STATE_KEY, [])

      await expect(agent.initialize()).rejects.toThrow(`${BACKGROUND_TASKS_STATE_KEY} must be an object`)
      agent.appState.delete(BACKGROUND_TASKS_STATE_KEY)
      await expect(agent.initialize()).resolves.toBeUndefined()
      await agent.invoke('run')

      expect(callback).toHaveBeenCalledOnce()
      expect(taskRecords(agent)).toEqual([])
      expect(backgroundDeliveries(agent)[0]?.toolUse.input).toEqual(
        expect.objectContaining({ status: 'completed', toolName: 'work' })
      )
    })
  })
})
